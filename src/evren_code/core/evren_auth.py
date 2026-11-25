"""EVREN AI Authentication, Terms of Service Acceptance, and Live Model Discovery Service (`evren_auth.py`).

Operates self-contained without external dependencies:
1. Validates `evren_llm_` API key format and persists credentials in `~/.evren-code/.evren-key`.
2. Verifies authentication via `GET /v1/models`.
3. Handles `403 terms_not_accepted` by fetching `/v1/terms/text` and posting approval to `/v1/terms/accept`.
4. Discovers active models and registers their capabilities (`tools`, `thinking`, `context_length`, `modalities`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from evren_code.config import AppConfig, ModelSpec

logger = logging.getLogger(__name__)


class EvrenAuthError(Exception):
    """EVREN authentication or API access error."""


class TermsNotAcceptedError(EvrenAuthError):
    """Raised when EVREN platform terms of service have not been accepted yet."""

    def __init__(self, terms_text: str, terms_file: Path) -> None:
        super().__init__("EVREN platform terms of service have not been accepted yet.")
        self.terms_text = terms_text
        self.terms_file = terms_file


class EvrenAuthManager:
    """Manages EVREN LLM API keys, terms agreement flows, and dynamic model discovery."""

    def __init__(
        self,
        config: AppConfig,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.config = config
        self._client = http_client

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=self.config.api.verify_timeout_seconds)

    @staticmethod
    def sanitize_key(raw_key: str) -> str:
        """Sanitize key."""
        cleaned = re.sub(r"\s+", "", raw_key or "")
        cleaned = cleaned.strip("\"'")
        return cleaned

    def validate_key_format(self, key: str) -> str:
        """Validate key format."""
        cleaned = self.sanitize_key(key)
        if not cleaned:
            raise EvrenAuthError("API anahtarı boş olamaz.")
        prefix = self.config.api.key_prefix
        if not cleaned.startswith(prefix):
            raise EvrenAuthError(
                f"API anahtarı '{prefix}' ile başlamalıdır. "
                "EVREN platformunda 'LLM Çıkarım -> API Anahtarı Oluştur' sayfasından alabilirsiniz."
            )
        return cleaned

    def mask_key(self, key: Optional[str] = None) -> str:
        """Return a masked representation of an API key safe for terminal display."""
        target = self.sanitize_key(key if key is not None else (self.config.api.api_key or ""))
        if not target:
            return "(tanımlı değil)"
        prefix = self.config.api.key_prefix
        if target.startswith(prefix):
            secret_part = target[len(prefix) :]
            if len(secret_part) > 6:
                return f"{prefix}****...{secret_part[-4:]}"
            return f"{prefix}****"
        if len(target) > 8:
            return f"{target[:4]}****...{target[-4:]}"
        return "****"

    def get_key_source(self) -> str:
        """Return human-readable source of the currently active API key."""
        if os.environ.get("EVREN_API_KEY"):
            return "EVREN_API_KEY (ortam değişkeni / .env)"
        key_file = self.config.api.key_file_path
        if key_file.is_file():
            return str(key_file)
        if self.config.api.api_key:
            return "oturum belleği"
        return "(kayıtlı anahtar yok)"

    def load_stored_key(self) -> Optional[str]:
        """Load stored key (.env)."""
        if self.config.api.api_key:
            try:
                return self.validate_key_format(self.config.api.api_key)
            except EvrenAuthError:
                pass

        key_file = self.config.api.key_file_path
        if key_file.is_file():
            content = key_file.read_text(encoding="utf-8").strip()
            if content:
                try:
                    key = self.validate_key_format(content)
                    self.config.api.api_key = key
                    return key
                except EvrenAuthError:
                    return None
        return None

    def save_key(self, key: str) -> Path:
        """Save key (0600)."""
        valid_key = self.validate_key_format(key)
        storage_dir = self.config.api.storage_dir
        storage_dir.mkdir(parents=True, exist_ok=True)

        key_file = self.config.api.key_file_path
        key_file.write_text(valid_key, encoding="utf-8")
        try:
            os.chmod(key_file, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        self.config.api.api_key = valid_key
        return key_file

    def fetch_terms_text(self, api_key: str) -> Tuple[str, Path]:
        """Fetch terms text."""
        url = f"{self.config.api.base_url}/terms/text"
        headers = {"Authorization": f"Bearer {api_key}"}

        client = self._get_client()
        should_close = self._client is None
        try:
            resp = client.get(url, headers=headers)
        finally:
            if should_close:
                client.close()

        if resp.status_code != 200:
            raise EvrenAuthError(
                f"Kullanım şartları metni alınamadı (HTTP {resp.status_code})."
            )

        data = resp.json()
        content = data.get("content", "")
        self.config.api.storage_dir.mkdir(parents=True, exist_ok=True)
        terms_path = self.config.api.terms_file_path
        terms_path.write_text(content, encoding="utf-8")
        return content, terms_path

    def accept_terms(self, api_key: str) -> None:
        """Accept terms."""
        url = f"{self.config.api.base_url}/terms/accept"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"version": self.config.api.terms_version}

        client = self._get_client()
        should_close = self._client is None
        try:
            resp = client.post(url, headers=headers, json=payload)
        finally:
            if should_close:
                client.close()

        if resp.status_code != 200:
            raise EvrenAuthError(
                f"Kullanım şartları onaylanamadı (HTTP {resp.status_code})."
            )

    def verify_and_discover_models(
        self,
        api_key: str,
        terms_acceptance_callback: Optional[Callable[[str, Path], bool]] = None,
    ) -> Dict[str, ModelSpec]:
        """Verify and discover models."""
        valid_key = self.validate_key_format(api_key)
        url = f"{self.config.api.base_url}/models"
        headers = {"Authorization": f"Bearer {valid_key}"}

        client = self._get_client()
        should_close = self._client is None
        try:
            resp = client.get(url, headers=headers)

            if resp.status_code == 401:
                raise EvrenAuthError(
                    "Anahtar reddedildi (401). Anahtarın tamamını kopyaladığınızdan ve "
                    "EVREN'de 'LLM Çıkarım' sınıfında oluşturulduğundan emin olun."
                )

            if resp.status_code == 403:
                err_code = ""
                try:
                    err_code = resp.json().get("error", {}).get("code", "")
                except (ValueError, json.JSONDecodeError) as exc:
                    logger.debug("403 yanıtından hata kodu parse edilemedi: %s", exc)

                if err_code == "terms_not_accepted":
                    terms_text, terms_path = self.fetch_terms_text(valid_key)
                    if terms_acceptance_callback is None:
                        raise TermsNotAcceptedError(terms_text, terms_path)
                    accepted = terms_acceptance_callback(terms_text, terms_path)
                    if not accepted:
                        raise EvrenAuthError(
                            "Kullanım şartları kabul edilmeden EVREN API kullanılamaz."
                        )
                    self.accept_terms(valid_key)
                    resp = client.get(url, headers=headers)
                else:
                    raise EvrenAuthError(
                        f"Erişim reddedildi (403, {err_code or 'bilinmeyen'})."
                    )

            if resp.status_code != 200:
                raise EvrenAuthError(
                    f"Model listesi alınamadı (HTTP {resp.status_code}). İnternet bağlantınızı kontrol edin."
                )

            payload = resp.json()
        finally:
            if should_close:
                client.close()

        discovered = self.parse_models_payload(payload)
        if discovered:
            self.config.models.catalog.update(discovered)
            self._save_models_cache(discovered)
        return discovered

    def parse_models_payload(self, payload: Any) -> Dict[str, ModelSpec]:
        """Parse models payload."""
        if isinstance(payload, dict):
            items = payload.get("data", [])
        elif isinstance(payload, list):
            items = payload
        else:
            items = []

        discovered: Dict[str, ModelSpec] = {}
        display_names = self.config.models.display_names
        default_ctx = self.config.models.default_context_length
        default_out = self.config.models.default_max_output_tokens

        for item in items:
            if not isinstance(item, dict):
                continue
            task = item.get("task", "chat")
            if task not in ("chat", "vision_chat"):
                continue
            mid = item.get("id")
            if not mid:
                continue

            if mid == "auto":
                discovered[mid] = ModelSpec(
                    id="auto",
                    name=display_names.get("auto", "EVREN Auto"),
                    task="chat",
                    context_length=default_ctx,
                    max_output_tokens=default_out,
                    tools=True,
                    thinking=True,
                    modalities=["text", "image"],
                )
                continue

            caps = item.get("capabilities") or {}
            modalities = item.get("modalities") or (
                ["text", "image"] if task == "vision_chat" else ["text"]
            )
            raw_max_out = (
                item.get("max_output_tokens")
                or item.get("default_max_output_tokens")
                or default_out
            )
            discovered[mid] = ModelSpec(
                id=mid,
                name=display_names.get(mid, item.get("name", mid)),
                task=task,
                context_length=int(item.get("context_length") or default_ctx),
                max_output_tokens=min(int(raw_max_out), 32768),
                tools=bool(caps.get("tools", True)),
                thinking=bool(caps.get("thinking", False)),
                modalities=list(modalities),
            )

        return discovered

    def _save_models_cache(self, models: Dict[str, ModelSpec]) -> None:
        """Save models cache."""
        try:
            self.config.api.storage_dir.mkdir(parents=True, exist_ok=True)
            serialized = {mid: asdict(spec) for mid, spec in models.items()}
            self.config.api.models_cache_path.write_text(
                json.dumps(serialized, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def load_models_cache(self) -> Dict[str, ModelSpec]:
        """Load models cache."""
        cache_path = self.config.api.models_cache_path
        if not cache_path.is_file():
            return {}
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            loaded: Dict[str, ModelSpec] = {}
            default_out = self.config.models.default_max_output_tokens
            for mid, data in raw.items():
                if not isinstance(data, dict):
                    continue
                if mid == "auto":
                    loaded["auto"] = ModelSpec(
                        id="auto",
                        name=self.config.models.display_names.get("auto", "EVREN Auto"),
                        task="chat",
                        context_length=self.config.models.default_context_length,
                        max_output_tokens=default_out,
                        tools=True,
                        thinking=True,
                        modalities=["text", "image"],
                    )
                    continue
                spec = ModelSpec(**data)
                cat_spec = self.config.models.catalog.get(mid)
                if cat_spec and spec.max_output_tokens < cat_spec.max_output_tokens:
                    spec.max_output_tokens = cat_spec.max_output_tokens
                loaded[mid] = spec
            self.config.models.catalog.update(loaded)
            return loaded
        except Exception as exc:
            logger.debug("Önbellekten model kataloğu yüklenirken hata: %s", exc)
            return {}
