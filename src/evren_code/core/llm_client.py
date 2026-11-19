"""Multi-Model Router module."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set

import httpx

from evren_code.config import AppConfig, ModelSpec
from evren_code.core.events import EventBus, EventType
from evren_code.core.parser import ParsedAssistantResponse, parse_assistant_message
from evren_code.core.stream_filter import StreamTagFilter


def calculate_retry_delay(
    attempt: int,
    response: Optional[httpx.Response] = None,
    base_delay: float = 1.5,
    max_delay: float = 60.0,
) -> float:
    """Calculate retry delay."""
    if response is not None:
        raw_header = response.headers.get("retry-after") or response.headers.get("Retry-After")
        if raw_header:
            try:
                hdr_val = float(raw_header.strip())
                if hdr_val > 0:
                    return min(hdr_val, max_delay)
            except ValueError:
                pass

    backoff = base_delay * (2 ** max(0, attempt))
    jitter = random.uniform(0.0, 1.5)
    return min(backoff + jitter, max_delay)


class EvrenAPIError(Exception):
    """Evren api error implementation."""


class EvrenLLMClient:
    """Evren llm client implementation."""

    def __init__(
        self,
        config: AppConfig,
        event_bus: Optional[EventBus] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus or EventBus()
        self._external_client = http_client
        self._owns_client = http_client is None
        self._persistent_client: Optional[httpx.Client] = http_client or httpx.Client(
            timeout=config.api.timeout_seconds,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=120.0),
        )
        self._active_model_id: str = config.models.default
        self._cancel_requested: bool = False
        self._model_cooldowns: Dict[str, float] = {}

    def _get_client(self) -> httpx.Client:
        """Get client (keep-alive)."""
        if self._persistent_client is None or self._persistent_client.is_closed:
            self._persistent_client = self._external_client or httpx.Client(
                timeout=self.config.api.timeout_seconds,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=120.0),
            )
        return self._persistent_client

    def close(self) -> None:
        """Close (connection pool)."""
        if self._persistent_client and not self._persistent_client.is_closed:
            if self._owns_client:
                self._persistent_client.close()
            self._persistent_client = None

    def __enter__(self) -> EvrenLLMClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def cancel(self) -> None:
        """Cancel."""
        self._cancel_requested = True

    def reset_cancel(self) -> None:
        self._cancel_requested = False

    @property
    def active_model(self) -> ModelSpec:
        return self.config.models.get_model(self._active_model_id)

    def set_model(self, model_or_role: str) -> ModelSpec:
        """Set model."""
        spec = self.config.models.get_model(model_or_role)
        self._active_model_id = spec.id
        self._model_cooldowns.pop(spec.id, None)
        self.event_bus.emit(
            EventType.MODEL_SWITCHED,
            model_id=spec.id,
            model_name=spec.name,
            tools=spec.tools,
            thinking=spec.thinking,
        )
        return spec

    def _parse_resets_at_cooldown(self, error_body: str, default_cooldown: float = 45.0) -> float:
        """Parse resets at cooldown."""
        try:
            data = json.loads(error_body)
            resets_at_str = (
                data.get("error", {}).get("evren", {}).get("resets_at")
                or data.get("evren", {}).get("resets_at")
            )
            if resets_at_str and isinstance(resets_at_str, str):
                norm_iso = resets_at_str.replace("Z", "+00:00")
                reset_dt = datetime.fromisoformat(norm_iso)
                now_dt = datetime.now(timezone.utc)
                remaining = (reset_dt - now_dt).total_seconds()
                if remaining > 0:
                    return max(5.0, min(remaining + 1.0, 180.0))
        except Exception:
            pass
        return default_cooldown

    def _mark_model_cooldown(self, model_id: str, error_body: str = "") -> float:
        """Mark model cooldown."""
        cooldown = self._parse_resets_at_cooldown(error_body)
        self._model_cooldowns[model_id] = time.time() + cooldown
        return cooldown

    def _is_model_on_cooldown(self, model_id: str) -> bool:
        exp = self._model_cooldowns.get(model_id, 0.0)
        if exp <= 0.0:
            return False
        if time.time() >= exp:
            self._model_cooldowns.pop(model_id, None)
            return False
        return True

    @staticmethod
    def _messages_contain_vision(messages: List[Dict[str, Any]]) -> bool:
        for msg in messages[-4:]:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                        return True
        return False

    def _pick_fallback_model(
        self,
        current_model_id: str,
        tried_models: Set[str],
        has_vision: bool = False,
    ) -> Optional[ModelSpec]:
        """Pick fallback model."""
        if has_vision:
            priority_chain = [
                "qwen3-vl-30b",
                "deepseek-v4.1-flash",
                "qwen3.8-flash-next",
                "gemma-4-31b",
            ]
        else:
            priority_chain = [
                "deepseek-v4.1-flash",
                "glm-5.3",
                "qwen3.8-flash-next",
                "gemma-4-31b",
                "auto",
            ]

        for cid in priority_chain:
            if cid == current_model_id or cid in tried_models:
                continue
            if cid in self.config.models.catalog and not self._is_model_on_cooldown(cid):
                return self.config.models.get_model(cid)

        untried = [
            cid
            for cid in priority_chain
            if cid != current_model_id and cid not in tried_models and cid in self.config.models.catalog
        ]
        if untried:
            best_cid = min(untried, key=lambda c: self._model_cooldowns.get(c, 0.0))
            return self.config.models.get_model(best_cid)

        return None

    @staticmethod
    def _inject_fallback_xml_instructions(
        payload: Dict[str, Any],
        fallback_tool_instructions: Optional[str],
    ) -> None:
        """Inject fallback xml instructions."""
        if not fallback_tool_instructions:
            return
        msgs = list(payload.get("messages") or [])
        if not msgs:
            return
        first = dict(msgs[0])
        if first.get("role") == "system" and isinstance(first.get("content"), str):
            if "<tool_call>" not in first["content"]:
                first["content"] = first["content"] + "\n\n" + fallback_tool_instructions
                msgs[0] = first
                payload["messages"] = msgs

    def _adapt_payload_for_model(
        self,
        payload: Dict[str, Any],
        spec: ModelSpec,
        tools_schema: Optional[List[Dict[str, Any]]],
        fallback_tool_instructions: Optional[str],
    ) -> None:
        """Adapt payload for model (failover)."""
        payload["model"] = spec.id
        payload["max_tokens"] = spec.max_output_tokens
        if tools_schema and spec.tools:
            payload["tools"] = tools_schema
            payload["tool_choice"] = "auto"
        else:
            payload.pop("tools", None)
            payload.pop("tool_choice", None)
            self._inject_fallback_xml_instructions(payload, fallback_tool_instructions)

    @staticmethod
    def _is_context_length_error(error_body: str) -> bool:
        """Is context length error."""
        lower = (error_body or "").lower()
        return (
            "context_length_exceeded" in lower
            or "bağlam penceresini" in lower
            or "maximum context length" in lower
            or "context window" in lower
        )

    @staticmethod
    def _parse_context_length_numbers(error_body: str) -> Optional[ tuple[int, int, int] ]:
        """Parse context length numbers (istem_token, cikti_tavani, baglam_penceresi)."""
        import re

        m = re.search(
            r"\((\d+)\)\s*\+\s*[^()]*\((\d+)\)[^()]*\((\d+)\)",
            error_body or "",
        )
        if m:
            try:
                return int(m.group(1)), int(m.group(2)), int(m.group(3))
            except ValueError:
                pass
        return None

    def _try_recover_context_overflow(
        self,
        error_body: str,
        payload: Dict[str, Any],
        spec: ModelSpec,
    ) -> bool:
        """Try recover context overflow."""
        parsed_nums = self._parse_context_length_numbers(error_body)
        if not parsed_nums:
            return False
        prompt_tokens, _req_max_out, ctx_window = parsed_nums
        if ctx_window > 0 and spec.context_length > ctx_window:
            spec.context_length = ctx_window

        available_out = ctx_window - prompt_tokens - 256
        current_max = int(payload.get("max_tokens") or spec.max_output_tokens)
        if available_out >= 1024 and current_max > available_out:
            payload["max_tokens"] = available_out
            self.event_bus.emit(
                EventType.WARNING,
                message=(
                    f"[EVREN API - Çıktı Tavanı Uyarlama]: Bağlam penceresine ({ctx_window}) uyum için "
                    f"`max_tokens` {current_max} -> {available_out} olarak daraltıldı."
                ),
            )
            return True
        return False

    @staticmethod
    def _has_tool_arg_repetition(arg_buf: str) -> bool:
        """Has tool arg repetition."""
        if len(arg_buf) < 50:
            return False

        tail = arg_buf[-800:]

        if re.search(r"(?:[ \t]*\r?\n[ \t]*){12,}", tail):
            return True

        candidates = [tail, tail.rstrip('"\t\r\n }')]
        for target in candidates:
            if not target:
                continue
            for k in range(3, 21):
                if len(target) < k * 8:
                    continue
                pat = target[-k:]
                if not pat.strip():
                    continue
                if target.endswith(pat * 8):
                    return True

            for k in range(21, 61):
                if len(target) < k * 5:
                    continue
                pat = target[-k:]
                if not pat.strip():
                    continue
                if target.endswith(pat * 5):
                    return True

            for k in range(30, 161):
                pat = target[-k:]
                if not pat.strip():
                    continue
                if (
                    target.endswith(pat * 3)
                    or target.endswith(pat + " " + pat + " " + pat)
                ):
                    return True

        return False

    def _get_headers(self) -> Dict[str, str]:
        api_key = self.config.api.api_key
        if not api_key:
            raise EvrenAPIError(
                "EVREN API anahtarı tanımlı değil. Lütfen `evren --setup` çalıştırın veya EVREN_API_KEY tanımlayın."
            )
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        model_override: Optional[str] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
        fallback_tool_instructions: Optional[str] = None,
        silent: bool = False,
        on_tool_speculate: Optional[Callable[[str, Dict[str, Any], str], None]] = None,
        cancel_event: Optional[Any] = None,
    ) -> ParsedAssistantResponse:
        """Chat."""
        if fallback_tool_instructions is None:
            fallback_tool_instructions = getattr(self, "_fallback_tool_instructions", None)
        spec = (
            self.config.models.get_model(model_override)
            if model_override
            else self.active_model
        )
        has_vision = self._messages_contain_vision(messages)

        if self._is_model_on_cooldown(spec.id):
            fb_spec = self._pick_fallback_model(spec.id, {spec.id}, has_vision=has_vision)
            if fb_spec is not None:
                rem = max(1, int(self._model_cooldowns.get(spec.id, 0.0) - time.time()))
                self.event_bus.emit(
                    EventType.WARNING,
                    message=(
                        f"[EVREN API - Yedek Model]: '{spec.name}' ({spec.id}) kapasite soğumasında (~{rem}sn kaldı); "
                        f"bu adım '{fb_spec.name}' ({fb_spec.id}) ile yürütülüyor."
                    ),
                )
                spec = fb_spec

        url = f"{self.config.api.base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": spec.id,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self.config.agent.temperature
            ),
            "max_tokens": spec.max_output_tokens,
            "stream": stream,
        }
        freq_pen = getattr(self.config.agent, "frequency_penalty", 0.0)
        payload["frequency_penalty"] = freq_pen if freq_pen > 0 else 0.15

        pres_pen = getattr(self.config.agent, "presence_penalty", 0.0)
        payload["presence_penalty"] = pres_pen if pres_pen > 0 else 0.05

        self._adapt_payload_for_model(payload, spec, tools_schema, fallback_tool_instructions)

        if stream:
            payload["stream_options"] = {"include_usage": True}
            return self._send_streaming(
                url,
                payload,
                spec,
                tools_schema=tools_schema,
                fallback_tool_instructions=fallback_tool_instructions,
                has_vision=has_vision,
                silent=silent,
                on_tool_speculate=on_tool_speculate,
                cancel_event=cancel_event,
            )
        return self._send_non_streaming(
            url,
            payload,
            spec,
            tools_schema=tools_schema,
            fallback_tool_instructions=fallback_tool_instructions,
            has_vision=has_vision,
            silent=silent,
            cancel_event=cancel_event,
        )

    def _send_non_streaming(
        self,
        url: str,
        payload: Dict[str, Any],
        spec: ModelSpec,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        fallback_tool_instructions: Optional[str] = None,
        has_vision: bool = False,
        silent: bool = False,
        cancel_event: Optional[Any] = None,
    ) -> ParsedAssistantResponse:
        """Send non streaming (non-streaming)."""
        client = self._get_client()
        should_close = False
        tried_models: Set[str] = {spec.id}
        same_model_retries = 0
        max_total_attempts = max(self.config.api.max_retries * 2, 6)

        try:
            for attempt in range(max_total_attempts):
                if (cancel_event and getattr(cancel_event, "is_set", lambda: False)()) or self._cancel_requested:
                    return ParsedAssistantResponse(content="İşlem kullanıcı/zamanlayıcı tarafından iptal edildi.")

                try:
                    resp = client.post(url, headers=self._get_headers(), json=payload)
                except (httpx.TransportError, httpx.TimeoutException) as err:
                    if (cancel_event and getattr(cancel_event, "is_set", lambda: False)()) or self._cancel_requested:
                        return ParsedAssistantResponse(content="İşlem iptal edildi.")

                    if attempt + 1 < max_total_attempts:
                        if isinstance(err, httpx.TimeoutException):
                            fb_spec = self._pick_fallback_model(spec.id, tried_models, has_vision=has_vision)
                            if fb_spec is not None:
                                tried_models.add(fb_spec.id)
                                same_model_retries = 0
                                self.event_bus.emit(
                                    EventType.WARNING,
                                    message=(
                                        f"[Ağ Zaman Aşımı]: '{spec.name}' ({spec.id}) yanıt vermedi ({err}). "
                                        f"İşlem kesintisiz olarak '{fb_spec.name}' ({fb_spec.id}) modeline aktarılıyor..."
                                    ),
                                )
                                spec = fb_spec
                                self._adapt_payload_for_model(payload, spec, tools_schema, fallback_tool_instructions)
                                continue

                        same_model_retries += 1
                        delay = calculate_retry_delay(
                            attempt=same_model_retries,
                            response=None,
                            base_delay=1.5,
                            max_delay=30.0,
                        )
                        self.event_bus.emit(
                            EventType.WARNING,
                            message=f"[Ağ Hatası]: {err}. {delay:.1f} saniye sonra tekrar deneniyor ({same_model_retries}/{self.config.api.max_retries})...",
                        )
                        time.sleep(delay)
                        continue
                    raise EvrenAPIError(f"Bağlantı hatası: {err}") from err

                if resp.status_code == 400:
                    err_body = resp.text
                    if "no user query found" in err_body.lower():
                        msgs = list(payload.get("messages") or [])
                        insert_pos = 1 if msgs and msgs[0].get("role") == "system" else 0
                        msgs.insert(insert_pos, {"role": "user", "content": "Lütfen mevcut göreve devam et ve adımları uygula."})
                        payload["messages"] = msgs
                        self.event_bus.emit(
                            EventType.WARNING,
                            message="[Bağlam Kurtarma]: Mesajlarda kullanıcı sorgusu eksikliği tespit edildi (HTTP 400). Çapa kullanıcı hedefi enjekte edildi ve adım tekrarlanıyor...",
                        )
                        continue
                    if self._is_context_length_error(err_body):
                        if self._try_recover_context_overflow(err_body, payload, spec):
                            continue
                        raise EvrenAPIError(
                            f"EVREN API Hatası (HTTP 400): {err_body[:400]}"
                        )
                    if "tools" in payload:
                        payload.pop("tools", None)
                        payload.pop("tool_choice", None)
                        spec.tools = False
                        self._inject_fallback_xml_instructions(payload, fallback_tool_instructions)
                        continue

                if resp.status_code in (429, 502, 503, 504):
                    err_body = resp.text
                    is_capacity_exhausted = "model_capacity_exhausted" in err_body or resp.status_code == 503

                    if is_capacity_exhausted or same_model_retries >= 1:
                        cooldown_sec = self._mark_model_cooldown(spec.id, err_body)
                        fb_spec = self._pick_fallback_model(spec.id, tried_models, has_vision=has_vision)
                        if fb_spec is not None:
                            tried_models.add(fb_spec.id)
                            same_model_retries = 0
                            self.event_bus.emit(
                                EventType.WARNING,
                                message=(
                                    f"[EVREN API - Otomatik Yedek Model]: '{spec.name}' ({spec.id}) şu anda yoğun/dolu "
                                    f"(HTTP {resp.status_code}, ~{int(cooldown_sec)}sn soğuma). "
                                    f"İşleminiz kesintisiz olarak '{fb_spec.name}' ({fb_spec.id}) modeline aktarıldı..."
                                ),
                            )
                            spec = fb_spec
                            self._adapt_payload_for_model(payload, spec, tools_schema, fallback_tool_instructions)
                            continue

                    if attempt + 1 < max_total_attempts:
                        same_model_retries += 1
                        delay = calculate_retry_delay(
                            attempt=same_model_retries,
                            response=resp,
                            base_delay=1.5,
                            max_delay=60.0,
                        )
                        self.event_bus.emit(
                            EventType.WARNING,
                            message=(
                                f"[EVREN API]: Model geçici olarak yoğun (HTTP {resp.status_code}). "
                                f"{delay:.1f} saniye içinde tekrar deneniyor ({same_model_retries}/{self.config.api.max_retries})..."
                            ),
                        )
                        time.sleep(delay)
                        continue

                if resp.status_code != 200:
                    raise EvrenAPIError(
                        f"EVREN API Hatası (HTTP {resp.status_code}): {resp.text[:400]}"
                    )

                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    return ParsedAssistantResponse(content="")
                message = choices[0].get("message") or {}
                finish_reason = choices[0].get("finish_reason")
                usage_dict = data.get("usage")
                parsed = parse_assistant_message(
                    message,
                    usage=usage_dict,
                    finish_reason=finish_reason,
                )
                if not silent:
                    if parsed.thinking:
                        self.event_bus.emit(EventType.THINKING_DELTA, delta=parsed.thinking)
                    if parsed.content:
                        self.event_bus.emit(EventType.TEXT_DELTA, delta=parsed.content)
                return parsed

            raise EvrenAPIError("Tüm yedek modeller denendi ancak EVREN API şu anda yanıt veremiyor.")
        finally:
            if should_close:
                client.close()

    def _send_streaming(
        self,
        url: str,
        payload: Dict[str, Any],
        spec: ModelSpec,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        fallback_tool_instructions: Optional[str] = None,
        has_vision: bool = False,
        silent: bool = False,
        on_tool_speculate: Optional[Callable[[str, Dict[str, Any], str], None]] = None,
        cancel_event: Optional[Any] = None,
    ) -> ParsedAssistantResponse:
        """Send streaming (Server-Sent Events)."""
        client = self._get_client()
        should_close = False
        tried_models: Set[str] = {spec.id}
        same_model_retries = 0
        max_total_attempts = max(self.config.api.max_retries * 2, 6)

        try:
            self._cancel_requested = False
            for attempt in range(max_total_attempts):
                content_chunks: List[str] = []
                reasoning_chunks: List[str] = []
                tool_calls_acc: Dict[int, Dict[str, Any]] = {}
                tool_arg_counters: Dict[int, int] = {}
                speculated_indices: Set[int] = set()
                accumulated_usage: Optional[Dict[str, int]] = None

                tag_filter = StreamTagFilter(
                    on_text=(lambda txt: None) if silent else (lambda txt: self.event_bus.emit(EventType.TEXT_DELTA, delta=txt)),
                    on_thinking=(lambda thk: None) if silent else (lambda thk: self.event_bus.emit(EventType.THINKING_DELTA, delta=thk)),
                    on_tool_preparing=(lambda name, bcount: None) if silent else (lambda name, bcount: self.event_bus.emit(EventType.TOOL_CALL_PREPARING, tool_name=name, bytes_received=bcount)),
                    repetition_guard=getattr(self.config.agent, "repetition_guard_enabled", True),
                    repetition_threshold=getattr(self.config.agent, "repetition_threshold_count", 3),
                )

                try:
                    with client.stream(
                        "POST", url, headers=self._get_headers(), json=payload
                    ) as resp:
                        if resp.status_code == 400:
                            err_body = resp.read().decode("utf-8", errors="replace")
                            if "no user query found" in err_body.lower():
                                msgs = list(payload.get("messages") or [])
                                insert_pos = 1 if msgs and msgs[0].get("role") == "system" else 0
                                msgs.insert(insert_pos, {"role": "user", "content": "Lütfen mevcut göreve devam et ve adımları uygula."})
                                payload["messages"] = msgs
                                self.event_bus.emit(
                                    EventType.WARNING,
                                    message="[Bağlam Kurtarma]: Mesajlarda kullanıcı sorgusu eksikliği tespit edildi (HTTP 400). Çapa kullanıcı hedefi enjekte edildi ve akış tekrarlanıyor...",
                                )
                                continue
                            if self._is_context_length_error(err_body):
                                if self._try_recover_context_overflow(err_body, payload, spec):
                                    continue
                                raise EvrenAPIError(
                                    f"EVREN API Akış Hatası (HTTP 400): {err_body[:400]}"
                                )
                            if "stream_options" in payload and "stream_options" in err_body.lower():
                                payload.pop("stream_options", None)
                                continue
                            if "tools" in payload:
                                payload.pop("tools", None)
                                payload.pop("tool_choice", None)
                                spec.tools = False
                                self._inject_fallback_xml_instructions(payload, fallback_tool_instructions)
                                continue

                        if resp.status_code in (429, 502, 503, 504):
                            err_body = resp.read().decode("utf-8", errors="replace")
                            is_capacity_exhausted = "model_capacity_exhausted" in err_body or resp.status_code == 503

                            if is_capacity_exhausted or same_model_retries >= 1:
                                cooldown_sec = self._mark_model_cooldown(spec.id, err_body)
                                fb_spec = self._pick_fallback_model(spec.id, tried_models, has_vision=has_vision)
                                if fb_spec is not None:
                                    tried_models.add(fb_spec.id)
                                    same_model_retries = 0
                                    self.event_bus.emit(
                                        EventType.WARNING,
                                        message=(
                                            f"[EVREN API - Otomatik Yedek Model]: '{spec.name}' ({spec.id}) şu anda yoğun/dolu "
                                            f"(HTTP {resp.status_code}, ~{int(cooldown_sec)}sn soğuma). "
                                            f"İşleminiz kesintisiz olarak '{fb_spec.name}' ({fb_spec.id}) modeline aktarıldı..."
                                        ),
                                    )
                                    spec = fb_spec
                                    self._adapt_payload_for_model(payload, spec, tools_schema, fallback_tool_instructions)
                                    continue

                            if attempt + 1 < max_total_attempts:
                                same_model_retries += 1
                                delay = calculate_retry_delay(
                                    attempt=same_model_retries,
                                    response=resp,
                                    base_delay=1.5,
                                    max_delay=60.0,
                                )
                                self.event_bus.emit(
                                    EventType.WARNING,
                                    message=(
                                        f"[EVREN API]: Model geçici olarak yoğun veya kapasite dolu (HTTP {resp.status_code}). "
                                        f"{delay:.1f} saniye içinde tekrar deneniyor ({same_model_retries}/{self.config.api.max_retries})..."
                                    ),
                                )
                                time.sleep(delay)
                                continue

                        if resp.status_code != 200:
                            body = resp.read().decode("utf-8", errors="replace")
                            raise EvrenAPIError(
                                f"EVREN API Akış Hatası (HTTP {resp.status_code}): {body[:400]}"
                            )

                        tool_arg_repetition = False
                        accumulated_finish_reason: Optional[str] = None
                        for line in resp.iter_lines():
                            if (cancel_event and getattr(cancel_event, "is_set", lambda: False)()) or self._cancel_requested:
                                self.event_bus.emit(EventType.WARNING, message="[İptal]: Kullanıcı/zamanlayıcı isteği ile akış durduruldu.")
                                break
                            if not line:
                                continue
                            if not line.startswith("data:"):
                                continue
                            data_str = line[len("data:") :].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            if "usage" in chunk and chunk["usage"]:
                                accumulated_usage = chunk["usage"]

                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            if choices[0].get("finish_reason"):
                                accumulated_finish_reason = choices[0].get("finish_reason")
                            delta = choices[0].get("delta") or {}

                            r_piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                            if r_piece:
                                reasoning_chunks.append(r_piece)
                                tag_filter.feed_thinking(r_piece)
                                if tag_filter.repetition_detected:
                                    break

                            c_piece = delta.get("content") or ""
                            if c_piece:
                                content_chunks.append(c_piece)
                                tag_filter.feed(c_piece)
                                if tag_filter.repetition_detected:
                                    break

                            tc_deltas = delta.get("tool_calls") or []
                            for tc_delta in tc_deltas:
                                idx = int(tc_delta.get("index", 0))
                                entry = tool_calls_acc.setdefault(
                                    idx,
                                    {
                                        "id": tc_delta.get("id") or "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                if tc_delta.get("id"):
                                    entry["id"] = tc_delta["id"]
                                fn_delta = tc_delta.get("function") or {}
                                if fn_delta.get("name"):
                                    entry["function"]["name"] += fn_delta["name"]
                                    if not silent and entry["function"]["name"]:
                                        self.event_bus.emit(
                                            EventType.TOOL_CALL_PREPARING,
                                            tool_name=entry["function"]["name"],
                                            bytes_received=len(entry["function"]["arguments"]),
                                        )
                                if fn_delta.get("arguments"):
                                    arg_chunk = fn_delta["arguments"]
                                    entry["function"]["arguments"] += arg_chunk
                                    cnt = tool_arg_counters.get(idx, 0) + 1
                                    tool_arg_counters[idx] = cnt
                                    if (
                                        self.config.agent.repetition_guard_enabled
                                        and ("\n" in arg_chunk or cnt % 25 == 0)
                                        and self._has_tool_arg_repetition(entry["function"]["arguments"])
                                    ):
                                        tool_arg_repetition = True
                                        break
                                    if not silent and (cnt <= 2 or cnt % 4 == 0):
                                        self.event_bus.emit(
                                            EventType.TOOL_CALL_PREPARING,
                                            tool_name=entry["function"]["name"],
                                            bytes_received=len(entry["function"]["arguments"]),
                                        )
                                if on_tool_speculate and idx not in speculated_indices:
                                    raw_args = entry["function"]["arguments"].strip()
                                    if raw_args.startswith("{") and raw_args.endswith("}"):
                                        try:
                                            parsed_args = json.loads(raw_args)
                                            if isinstance(parsed_args, dict) and entry["function"]["name"]:
                                                speculated_indices.add(idx)
                                                call_id = entry["id"] or f"call_spec_{idx}"
                                                on_tool_speculate(entry["function"]["name"], parsed_args, call_id)
                                        except Exception:
                                            pass
                            if tool_arg_repetition:
                                break
                        tag_filter.flush()

                        if on_tool_speculate and tool_calls_acc and not tool_arg_repetition:
                            for idx, entry in tool_calls_acc.items():
                                if idx not in speculated_indices and entry["function"]["name"]:
                                    raw_args = entry["function"]["arguments"].strip()
                                    try:
                                        parsed_args = json.loads(raw_args) if raw_args else {}
                                        if isinstance(parsed_args, dict):
                                            speculated_indices.add(idx)
                                            call_id = entry["id"] or f"call_spec_{idx}"
                                            on_tool_speculate(entry["function"]["name"], parsed_args, call_id)
                                    except Exception:
                                        pass

                        if tool_calls_acc and not tool_arg_repetition and self.config.agent.repetition_guard_enabled:
                            for entry in tool_calls_acc.values():
                                if self._has_tool_arg_repetition(entry["function"]["arguments"]):
                                    tool_arg_repetition = True
                                    break

                        assembled_message: Dict[str, Any] = {
                            "content": "".join(content_chunks),
                            "reasoning_content": "".join(reasoning_chunks),
                        }
                        if tool_calls_acc and not tool_arg_repetition:
                            assembled_message["tool_calls"] = [
                                tool_calls_acc[k] for k in sorted(tool_calls_acc.keys())
                            ]

                        return parse_assistant_message(
                            assembled_message,
                            repetition_detected=(tag_filter.repetition_detected or tool_arg_repetition),
                            usage=accumulated_usage,
                            finish_reason=accumulated_finish_reason,
                        )
                except (httpx.TransportError, httpx.TimeoutException) as err:
                    if attempt + 1 < max_total_attempts:
                        # Fallback to secondary model if timeout occurs
                        if isinstance(err, httpx.TimeoutException):
                            fb_spec = self._pick_fallback_model(spec.id, tried_models, has_vision=has_vision)
                            if fb_spec is not None:
                                tried_models.add(fb_spec.id)
                                self.event_bus.emit(
                                    EventType.WARNING,
                                    message=(
                                        f"[Network Timeout / Ağ Zaman Aşımı]: '{spec.name}' ({spec.id}) did not respond ({err}). "
                                        f"Seamlessly failing over to '{fb_spec.name}' ({fb_spec.id})..."
                                    ),
                                )
                                spec = fb_spec
                                self._adapt_payload_for_model(payload, spec, tools_schema, fallback_tool_instructions)
                                continue
                        delay = calculate_retry_delay(
                            attempt=attempt + 1,
                            response=None,
                            base_delay=2.0,
                            max_delay=30.0,
                        )
                        self.event_bus.emit(
                            EventType.WARNING,
                            message=f"[Network Error / Ağ Hatası]: {err}. Retrying in {delay:.1f}s...",
                        )
                        time.sleep(delay)
                        continue
                    raise EvrenAPIError(f"Connection error / Bağlantı hatası: {err}") from err

            raise EvrenAPIError("Maximum retry limit reached (Maksimum yeniden deneme sayısına ulaşıldı).")
        finally:
            if should_close:
                client.close()


