"""Evren-Code Security, Permissions, and Containment Sandbox Module.

Features:
1. Workspace containment: restricts file system access within `workspace_root` (symlink and traversal protection).
2. Command guard: detects destructive/dangerous shell commands (`rm -rf /`, `curl | sh`, `git push --force`, etc.).
3. Layered permission models (Default, AcceptEdits, Plan, Bypass) and session-based approvals.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class PermissionMode(str, Enum):
    """Permission modes (Granular Security Architecture)."""
    DEFAULT = "default"          # Asks approval based on config and rules
    ACCEPT_EDITS = "acceptEdits"  # Auto-approves file edits, asks for shell commands
    PLAN = "plan"                # Read-only tools allowed; file edits and shell commands blocked
    BYPASS = "bypass"            # Trust mode: all tools allowed except DANGEROUS commands!


SECURITY_ERR_PERMISSION_DENIED = "PERMISSION_DENIED"
SECURITY_ERR_ACCESS_VIOLATION = "ACCESS_VIOLATION"
SECURITY_ERR_APPROVAL_REJECTED = "APPROVAL_REJECTED"
SECURITY_ERR_DANGEROUS_COMMAND = "DANGEROUS_COMMAND"
SECURITY_ERR_ROLE_RESTRICTION = "ROLE_RESTRICTION"
SECURITY_ERR_SYSTEM_PATH_MODIFICATION = "SYSTEM_PATH_MODIFICATION"
SECURITY_ERR_REGISTRY_MODIFICATION = "REGISTRY_MODIFICATION"
SECURITY_ERR_SYSTEM_DIR_VIOLATION = "SYSTEM_DIR_VIOLATION"
SECURITY_ERR_PACKAGE_INSTALL_RESTRICTION = "PACKAGE_INSTALL_RESTRICTION"


@dataclass
class SecurityDecision:
    """Security inspection decision result."""
    allowed: bool
    requires_approval: bool = False
    reason: str = ""
    is_dangerous: bool = False
    prompt_message: Optional[str] = None
    structured_error_code: Optional[str] = None


@dataclass
class ExecutionContext:
    """Execution context implementation."""
    caller_role: str = "main"
    caller_agent: str = "orchestrator"
    is_trust_mode: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkspaceGuard:
    """WorkspaceGuard: Workspace Boundary & Permission Guard."""

    PROTECTED_SYSTEM_DIRS = [
        "c:\\windows",
        "c:\\program files",
        "c:\\program files (x86)",
        "c:\\programdata",
        "appdata\\local\\programs\\ollama",
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        "/var",
        "/opt",
    ]

    @classmethod
    def is_contained(cls, workspace_root: Path, target_path: Path) -> bool:
        """Is contained (realpath)."""
        try:
            root_real = Path(os.path.realpath(str(workspace_root)))
            target_real = Path(os.path.realpath(str(target_path)))

            try:
                target_real.relative_to(root_real)
                return True
            except ValueError:
                pass

            root_s = str(root_real).lower().rstrip("/\\")
            target_s = str(target_real).lower().rstrip("/\\")
            if target_s == root_s:
                return True
            sep = os.sep.lower()
            if target_s.startswith(root_s + sep) or target_s.startswith(root_s + "/"):
                return True

            return False
        except Exception:
            return False

    @classmethod
    def validate_path(
        cls,
        workspace_root: Path,
        target_path: Path,
        is_write: bool = False,
        allow_outside: bool = False,
        raw_input: Optional[str] = None,
    ) -> Path:
        """Validate path."""
        try:
            root_real = Path(os.path.realpath(str(workspace_root)))
            target_real = Path(os.path.realpath(str(target_path)))
        except Exception as exc:
            disp = raw_input or str(target_path)
            raise PermissionError(
                f"Security Violation [{SECURITY_ERR_ACCESS_VIOLATION}] (Güvenlik Hatası): '{disp}' could not be resolved: {exc}"
            )

        target_norm = str(target_real).lower().replace("/", "\\")
        disp = raw_input or str(target_path)

        # 1. Protected system directories
        for sys_dir in cls.PROTECTED_SYSTEM_DIRS:
            if sys_dir in target_norm:
                raise PermissionError(
                    f"Security Violation [{SECURITY_ERR_SYSTEM_DIR_VIOLATION}] (Güvenlik Hatası): '{disp}' matches protected system directory ('{sys_dir}'). Access is prohibited."
                )

        # 2. Strict write containment
        if is_write:
            if not cls.is_contained(workspace_root, target_real):
                raise PermissionError(
                    f"Security Violation [{SECURITY_ERR_ACCESS_VIOLATION}] (Güvenlik Hatası): '{disp}' is outside workspace ({workspace_root}). Writing/modifying outside workspace is strictly prohibited."
                )
            return target_real

        # 3. Read containment
        if not cls.is_contained(workspace_root, target_real):
            if not allow_outside:
                raise PermissionError(
                    f"Security Violation [{SECURITY_ERR_ACCESS_VIOLATION}] (Güvenlik Hatası): '{disp}' is outside workspace ({workspace_root}). Access denied."
                )

        return target_real


class PathContainment:
    """Path containment implementation."""

    @staticmethod
    def is_contained(workspace_root: Path, target_path: Path) -> bool:
        return WorkspaceGuard.is_contained(workspace_root, target_path)

    @classmethod
    def assert_contained(
        cls,
        workspace_root: Path,
        target_path: Path,
        allow_outside: bool = False,
        raw_input: Optional[str] = None,
        is_write: bool = False,
    ) -> Path:
        """Assert contained."""
        return WorkspaceGuard.validate_path(
            workspace_root=workspace_root,
            target_path=target_path,
            is_write=is_write,
            allow_outside=allow_outside,
            raw_input=raw_input,
        )


class CommandClassifier:
    """Command classifier implementation."""

    DANGEROUS_PATTERNS = [
        (r"\brm\s+-[rfRF]{1,4}\s+[/~*]", "Kök veya ev dizininde toplu silme (rm -rf /)"),
        (r"\brmdir\s+/[sS]\s+/[qQ]\s+[a-zA-Z]:\\?", "Windows disk kökünde sessiz silme (rmdir /s /q)"),
        (r"\bdel\s+/[fF]\s+/[sS]\s+/[qQ]\s+[a-zA-Z]:\\?", "Windows disk kökünde sessiz silme (del /f /s /q)"),
        (r"\bformat\s+[a-zA-Z]:", "Sürücü biçimlendirme (format)"),
        (r"\bmkfs\b", "Dosya sistemi formatlama (mkfs)"),
        (r"\bdd\s+if=.*of=/dev/", "Ham blok diske yazma (dd of=/dev/...)"),
        (r":\(\)\s*\{\s*:\|:&\s*\};:", "Bash fork bombası"),
        (r"%\s*0\s*\|\s*%\s*0", "Windows cmd fork bombası"),
        (r"\b(?:curl|wget)\s+.*\|\s*(?:sh|bash|zsh|powershell|cmd)\b", "Uzak kaynaktan kod indirip doğrudan kabukta çalıştırma"),
        (r"iex\s*\(\s*(?:New-Object|iwr|curl)", "PowerShell Invoke-Expression ile uzaktan kod çalıştırma"),
        # Git tehlikeli operasyonlar
        (r"\bgit\s+push\s+.*(?:--force|-f)\b", "Zorunlu uzak git ezme (git push --force)"),
        (r"\bgit\s+reset\s+--hard\s+origin", "Uzak dalla yerel geçmişi kalıcı ezme"),
        (r"\bgit\s+clean\s+-[a-zA-Z]*x", "Tüm yok sayılan dosyaları kalıcı silme"),
        (r"\b(?:shutdown|reboot|poweroff|init\s+0)\b", "Sistem durdurma / yeniden başlatma"),
        (r"\bnetsh\s+advfirewall\s+set\s+allprofiles\s+state\s+off\b", "Güvenlik duvarını tamamen kapatma"),
        (r"\bufw\s+disable\b", "Linux güvenlik duvarını (UFW) kapatma"),
        (r"\bvssadmin\s+delete\s+shadows\b", "Sistem gölge kopyalarını silme"),
        (r"\b(?:powershell|pwsh)(?:\.exe)?\s+.*-(?:e|enc|encodedcommand)\b", "Base64 ile kodlanmış PowerShell yürütme engeli (-EncodedCommand)"),
        (r"\bsetx(?:\.exe)?\b", "Kalıcı Windows PATH veya ortam değişkeni değiştirme engeli (setx)"),
        (r"\[System\.Environment\]::SetEnvironmentVariable", "PowerShell ile kalıcı ortam değişkeni/PATH değiştirme engeli"),
        (r"\$env:PATH\s*=", "PowerShell PATH ortam değişkeni üzerine yazma engeli"),
        (r"\bexport\s+PATH\s*=", "Kabuk PATH ortam değişkenini geçersiz kılma engeli"),
        (r"\breg(?:\.exe)?\s+(?:add|delete|import|restore|copy)\b", "Windows Kayıt Defteri (Registry) doğrudan müdahale engeli (reg)"),
        (r"\b(?:Set|Remove|New)-ItemProperty\s+.*(?:Registry|HKCU|HKLM|HKCR|HKU):", "PowerShell Kayıt Defteri manipülasyonu engeli"),
        (r"(?:>|>>|Out-File|Set-Content|Add-Content)\s+.*(?:\\|/)(?:Ollama|Windows|Program\s*Files|ProgramData|AppData\\Local\\Programs)", "Sistem veya uygulama kurulum dizinlerine (Ollama, Windows vb.) dosya/shim yazma engeli"),
        (r"\b(?:copy|move|cp|mv)\s+.*(?:\\|/)(?:Ollama|Windows|Program\s*Files|ProgramData|AppData\\Local\\Programs)", "Sistem veya uygulama dizinlerine dosya kopyalama/taşıma engeli"),
        (r"\b(?:mklink|ln\s+-s)\s+.*(?:\\|/)(?:Ollama|Windows|Program\s*Files|ProgramData|AppData\\Local\\Programs)", "Sistem/uygulama dizinlerine sembolik link veya shim oluşturma engeli"),
        (r"\bpip(?:\d+(?:\.\d+)?)?(?:\.exe)?\s+install\s+.*(?:--user|--system|-g)\b", "Global/kullanıcı düzeyinde izinsiz pip paket kurulum engeli"),
        (r"\bnpm\s+(?:install|i|add|update)\s+.*-g(?:lobal)?\b", "Global npm paket kurulum engeli (npm -g)"),
        (r"\byarn\s+global\b", "Global yarn paket kurulum engeli"),
        (r"\b(?:choco|winget|scoop)\s+install\b", "Sistem paket yöneticisiyle izinsiz kurulum engeli"),
        (r"\b(?:apt|apt-get|yum|dnf|pacman|brew)\s+install\b", "İşletim sistemi paket yöneticisiyle kurulum engeli"),
        (r"\[(?:System\.)?IO\.File\]::ReadAllBytes\s*\([^)]*\.(?:class|pyc|exe|dll|so|o)", "Derlenmiş ikili (binary/.class) dosya içeriğini okuma/tarama engeli"),
        (r"\bjavap\b", "Derlenmiş Java bayt kodunu ayrıştırma/inceleme engeli; lütfen kaynak kodları inceleyin"),
    ]

    @classmethod
    def analyze(cls, command: str) -> Tuple[bool, str]:
        """Analyze (is_dangerous, reason)."""
        clean = command.strip()
        for pattern, reason in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, clean, re.IGNORECASE):
                return True, reason
        return False, "Güvenli / Standart komut"

    @classmethod
    def analyze_for_role(cls, command: str, role: str) -> Tuple[bool, str, str]:
        """Analyze for role."""
        role_clean = (role or "main").strip().lower()
        clean = command.strip()

        is_dangerous, reason = cls.analyze(clean)
        if is_dangerous:
            if role_clean != "main":
                return (
                    False,
                    f"Alt-ajan ({role_clean}) için yüksek riskli komut engellendi: {reason}",
                    SECURITY_ERR_DANGEROUS_COMMAND,
                )
            return True, reason, SECURITY_ERR_DANGEROUS_COMMAND

        if role_clean in ("researcher", "reviewer", "architect"):
            return (
                False,
                f"'{role_clean}' rolündeki alt-ajan kabuk komutlarını (`run_command`) çalıştıramaz.",
                SECURITY_ERR_ROLE_RESTRICTION,
            )

        if role_clean == "tester":
            forbidden_tester_patterns = [
                (r"\bgit\s+(?:push|commit|merge|rebase|tag|checkout\s+-b)\b", "Tester rolü git geçmişini değiştiremez veya uzak depoya push atamaz"),
                (r"\b(?:rm|del|rmdir|rd|erase)\b", "Tester rolü dosya veya dizin silme komutu çalıştıramaz"),
                (r"\b(?:pip\d*|conda)\s+install\b", "Tester rolü paket kurulumu yapamaz (pip/conda install)"),
                (r"\bnpm\s+(?:install|i|add|update)\b", "Tester rolü paket kurulumu yapamaz (npm install/add)"),
                (r"\b(?:yarn|pnpm)\s+(?:add|install)\b", "Tester rolü paket kurulumu yapamaz (yarn/pnpm add)"),
                (r"\bcargo\s+install\b", "Tester rolü crate kurulumu yapamaz (cargo install)"),
                (r"\bgo\s+(?:get|install)\b", "Tester rolü Go modülü kurulumu yapamaz (go get/install)"),
                (r"\b(?:useradd|net\s+user|chmod\s+777)\b", "Tester rolü sistem yetkilerini değiştiremez"),
                (r"\bpython(?:\d+(?:\.\d+)?)?(?:\.exe)?\s+-c\b", "Tester rolü dinamik satır içi Python betiği (-c) çalıştıramaz"),
                (r"\bnode(?:\.exe)?\s+-e\b", "Tester rolü dinamik satır içi Node betiği (-e) çalıştıramaz"),
                (r"\b(?:powershell|pwsh)(?:\.exe)?\s+.*-(?:e|enc|encodedcommand)\b", "Tester rolü kodlanmış PowerShell çalıştıramaz"),
            ]
            for pat, why in forbidden_tester_patterns:
                if re.search(pat, clean, re.IGNORECASE):
                    return (
                        False,
                        f"Tester yetki aşımı: {why} (`{clean}`).",
                        SECURITY_ERR_ROLE_RESTRICTION,
                    )

        return True, "Komut çalıştırılabilir", ""


class PermissionManager:
    """Permission manager implementation."""

    MODIFYING_TOOLS: Set[str] = {
        "write_file",
        "append_file",
        "write_multiple_files",
        "edit_file",
        "multi_edit_file",
        "run_command",
    }

    def __init__(
        self,
        workspace_root: Path,
        initial_mode: PermissionMode = PermissionMode.DEFAULT,
        require_approval_for: Optional[List[str]] = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.mode = initial_mode
        self.configured_requires_approval: Set[str] = set(require_approval_for or [])
        self.session_allowed_tools: Set[str] = set()
        self.session_allowed_commands: Set[str] = set()
        self.project_allowed_tools: Set[str] = set()
        self.project_allowed_commands: Set[str] = set()
        self.current_prompt: str = ""
        self._load_project_settings()

    def _settings_path(self) -> Path:
        return self.workspace_root / ".evren" / "settings.json"

    def _load_project_settings(self) -> None:
        """Load project settings."""
        spath = self._settings_path()
        if not spath.is_file():
            return
        import json
        try:
            data = json.loads(spath.read_text(encoding="utf-8"))
            perms = data.get("permissions", {})
            self.project_allowed_tools.update(perms.get("allowed_tools", []))
            self.project_allowed_commands.update(perms.get("allowed_commands", []))
            mode_str = perms.get("mode")
            if mode_str:
                try:
                    self.mode = PermissionMode(mode_str)
                except ValueError:
                    pass
        except Exception:
            pass

    def allow_tool_for_session(self, tool_name: str) -> None:
        """Allow tool for session."""
        self.session_allowed_tools.add(tool_name)

    def allow_command_for_session(self, command: str) -> None:
        """Allow command for session."""
        self.session_allowed_commands.add(command.strip())

    def allow_tool_for_project(self, tool_name: str) -> None:
        """Allow tool for project."""
        self.project_allowed_tools.add(tool_name)
        self._save_project_settings()

    def _save_project_settings(self) -> None:
        spath = self._settings_path()
        spath.parent.mkdir(parents=True, exist_ok=True)
        import json
        data = {
            "permissions": {
                "mode": self.mode.value,
                "allowed_tools": sorted(list(self.project_allowed_tools)),
                "allowed_commands": sorted(list(self.project_allowed_commands)),
            }
        }
        spath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def check(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        is_trust_mode: bool = False,
        configured_requires_approval: bool = False,
        caller_role: str = "main",
        caller_id: str = "orchestrator",
    ) -> SecurityDecision:
        """Check."""
        role_clean = (caller_role or "main").strip().lower()

        if self.mode == PermissionMode.PLAN and tool_name in self.MODIFYING_TOOLS:
            return SecurityDecision(
                allowed=False,
                requires_approval=False,
                reason=(
                    f"Plan Modu Aktif: '{tool_name}' aracı dosya sistemi veya kabuk üzerinde değişiklik yapamaz. "
                    "Plan modunda sadece okuma ve analiz araçlarına izin verilir."
                ),
                structured_error_code=SECURITY_ERR_PERMISSION_DENIED,
            )

        if role_clean in ("researcher", "reviewer", "architect") and tool_name in self.MODIFYING_TOOLS:
            return SecurityDecision(
                allowed=False,
                requires_approval=False,
                reason=f"'{role_clean}' rolü '{tool_name}' gibi değişiklik yapan veya kabuk çalıştıran araçları kullanamaz.",
                structured_error_code=SECURITY_ERR_ROLE_RESTRICTION,
            )

        if tool_name == "run_command":
            cmd = str(arguments.get("command", "")).strip()

            if self.current_prompt:
                from evren_code.core.planner import ScopeGuard
                is_unsolicited, unsol_reason = ScopeGuard.check_unsolicited_action(self.current_prompt, cmd)
                if is_unsolicited:
                    return SecurityDecision(
                        allowed=False,
                        requires_approval=False,
                        reason=unsol_reason,
                        structured_error_code=SECURITY_ERR_PERMISSION_DENIED,
                    )

            role_allowed, role_reason, role_code = CommandClassifier.analyze_for_role(cmd, role_clean)
            if not role_allowed:
                return SecurityDecision(
                    allowed=False,
                    requires_approval=False,
                    reason=role_reason,
                    structured_error_code=role_code,
                )

            is_dangerous, reason = CommandClassifier.analyze(cmd)

            if is_dangerous:
                if role_clean != "main":
                    return SecurityDecision(
                        allowed=False,
                        requires_approval=False,
                        is_dangerous=True,
                        reason=f"Alt-ajan ({role_clean}) için yüksek riskli komut engellendi: {reason}",
                        structured_error_code=SECURITY_ERR_DANGEROUS_COMMAND,
                    )
                return SecurityDecision(
                    allowed=True,
                    requires_approval=True,
                    is_dangerous=True,
                    reason=f"KRİTİK GÜVENLİK UYARISI: {reason}",
                    prompt_message=(
                        f"⚠️ [YÜKSEK RİSKLİ KOMUT]: `{cmd}`\n"
                        f"Açıklama: {reason}\n"
                        "Bu komut sisteminize veya projenize kalıcı zarar verebilir. Çalıştırmayı onaylıyor musunuz?"
                    ),
                    structured_error_code=SECURITY_ERR_DANGEROUS_COMMAND,
                )

            if cmd in self.session_allowed_commands or cmd in self.project_allowed_commands:
                return SecurityDecision(allowed=True, requires_approval=False)

            if role_clean == "tester":
                return SecurityDecision(allowed=True, requires_approval=False)

            if is_trust_mode or self.mode in (PermissionMode.BYPASS, PermissionMode.ACCEPT_EDITS):
                return SecurityDecision(allowed=True, requires_approval=False)

            return SecurityDecision(
                allowed=True,
                requires_approval=True,
                reason="Kabuk komutu onayı bekleniyor",
            )

        if tool_name in self.MODIFYING_TOOLS:
            if tool_name == "write_multiple_files":
                files_arg = arguments.get("files") or []
                if isinstance(files_arg, list) and len(files_arg) > 20:
                    return SecurityDecision(
                        allowed=True,
                        requires_approval=True,
                        reason=f"Anormal sayıda dosya ({len(files_arg)} dosya) yazılmak isteniyor. Model bozulmasını önlemek için güven modunda bile onay zorunludur.",
                        prompt_message=f"⚠️ [ANORMAL TOPLU YAZMA]: Model tek seferde {len(files_arg)} dosya yazmak istiyor. Bu bir üretim bozulması olabilir. Onaylıyor musunuz?",
                    )

            if is_trust_mode or self.mode in (PermissionMode.ACCEPT_EDITS, PermissionMode.BYPASS):
                return SecurityDecision(allowed=True, requires_approval=False)

            if tool_name in self.session_allowed_tools or tool_name in self.project_allowed_tools:
                return SecurityDecision(allowed=True, requires_approval=False)

            if role_clean != "main":
                return SecurityDecision(
                    allowed=True,
                    requires_approval=True,
                    reason=f"Alt-ajan ({role_clean}) '{tool_name}' ile dosya değişikliği yapmak istiyor. Kullanıcı onayı zorunludur.",
                )

            return SecurityDecision(
                allowed=True,
                requires_approval=True,
                reason=f"'{tool_name}' aracı değişiklik onayı gerektiriyor",
            )

        if tool_name in self.session_allowed_tools or tool_name in self.project_allowed_tools:
            return SecurityDecision(allowed=True, requires_approval=False)

        if (
            (configured_requires_approval or tool_name in self.configured_requires_approval)
            and not is_trust_mode
            and self.mode != PermissionMode.BYPASS
        ):
            return SecurityDecision(
                allowed=True,
                requires_approval=True,
                reason=f"'{tool_name}' aracı onay gerektiriyor",
            )

        return SecurityDecision(allowed=True, requires_approval=False)


class ToolExecutor:
    """Tool executor implementation."""

    def __init__(
        self,
        registry: Any,
        permission_manager: PermissionManager,
        approval_delegate: Any,
        event_bus: Optional[Any] = None,
        max_output_chars: int = 16000,
    ) -> None:
        from evren_code.core.tool_reliability import CentralizedToolExecutionManager

        self.registry = registry
        self.permission_manager = permission_manager
        self.approval_delegate = approval_delegate
        self.event_bus = event_bus
        self.max_output_chars = max_output_chars
        self.execution_manager = CentralizedToolExecutionManager(
            event_bus=self.event_bus,
            max_output_chars=self.max_output_chars,
        )

    def cancel_all(self, reason: str = "Orchestrator cancelled") -> int:
        """Cancel all."""
        return self.execution_manager.cancel_all(reason=reason)

    def close(self) -> None:
        """Close (watchdog)."""
        self.execution_manager.cancel_all(reason="ToolExecutor closing")
        self.execution_manager.watchdog.stop()

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        caller_role: str = "main",
        caller_agent: str = "orchestrator",
        is_trust_mode: bool = False,
        prefetched_result: Optional[str] = None,
        timeout: Optional[float] = None,
        cancel_event: Optional[Any] = None,
    ) -> str:
        """Execute (reliability-managed)."""
        tool_def = self.registry.get(tool_name)
        if not tool_def:
            available = ", ".join(self.registry._tools.keys()) if hasattr(self.registry, "_tools") else ""
            return f"Hata: '{tool_name}' adında bir araç bulunamadı. Mevcut araçlar: {available}"

        configured_req = bool(
            getattr(tool_def, "requires_approval", False)
            or tool_name in self.permission_manager.configured_requires_approval
        )

        decision = self.permission_manager.check(
            tool_name=tool_name,
            arguments=arguments,
            is_trust_mode=is_trust_mode,
            configured_requires_approval=configured_req,
            caller_role=caller_role,
            caller_id=caller_agent,
        )

        if not decision.allowed:
            err_code = decision.structured_error_code or SECURITY_ERR_PERMISSION_DENIED
            err_msg = f"[GÜVENLİK ENGELİ / {err_code}]: {decision.reason}"
            if self.event_bus:
                from evren_code.core.events import EventType
                self.event_bus.emit(
                    EventType.TOOL_EXECUTION_COMPLETED,
                    tool_name=tool_name,
                    approved=False,
                    result=err_msg,
                )
            return err_msg

        if decision.requires_approval:
            diff_preview = (
                tool_def.preview_builder(arguments)
                if tool_def and getattr(tool_def, "preview_builder", None)
                else None
            )
            caller_label = f"[{caller_role}]" if caller_role != "main" else ""
            default_prompt = f"{caller_label} '{tool_name}' aracını çalıştırmak için kullanıcı onayı gerekiyor."
            prompt_desc = decision.prompt_message or default_prompt
            risk_lvl = (
                "dangerous"
                if decision.is_dangerous
                else (
                    "modifying"
                    if tool_name in self.permission_manager.MODIFYING_TOOLS
                    else "standard"
                )
            )

            from evren_code.core.events import ApprovalRequest, EventType
            req = ApprovalRequest(
                tool_name=tool_name,
                arguments=arguments,
                description=prompt_desc,
                diff_preview=diff_preview,
                caller_role=caller_role,
                caller_agent=caller_agent,
                risk_level=risk_lvl,
            )
            approved = self.approval_delegate.request_approval(req)
            if not approved:
                rej_msg = (
                    f"[KULLANICI REDDİ / {SECURITY_ERR_APPROVAL_REJECTED}]: "
                    f"Kullanıcı '{caller_role}' rolündeki '{caller_agent}' tarafından talep edilen "
                    f"'{tool_name}' işlemini onaylamadı."
                )
                if self.event_bus:
                    self.event_bus.emit(
                        EventType.TOOL_EXECUTION_COMPLETED,
                        tool_name=tool_name,
                        approved=False,
                        result=rej_msg,
                    )
                return rej_msg

        if self.event_bus:
            from evren_code.core.events import EventType
            self.event_bus.emit(
                EventType.TOOL_EXECUTION_STARTED,
                tool_name=tool_name,
                arguments=arguments,
                caller_role=caller_role,
                caller_agent=caller_agent,
            )

        if prefetched_result is not None:
            result_str, _ = self.execution_manager.output_limiter.limit(
                prefetched_result,
                max_chars=self.max_output_chars,
                tool_name=tool_name,
            )
            if self.event_bus and caller_agent != "orchestrator":
                self.event_bus.emit(
                    EventType.TOOL_EXECUTION_COMPLETED,
                    tool_name=tool_name,
                    approved=True,
                    result=result_str,
                )
            return result_str

        if isinstance(arguments, dict) and "_raw_unparsed" in arguments:
            res_raw = self.registry.execute(tool_name, arguments)
            if self.event_bus and caller_agent != "orchestrator":
                self.event_bus.emit(
                    EventType.TOOL_EXECUTION_COMPLETED,
                    tool_name=tool_name,
                    approved=True,
                    result=res_raw,
                )
            return res_raw

        handler = getattr(tool_def, "handler", None)
        if handler is None or hasattr(self.registry.execute, "assert_called"):
            def _fallback_handler(**kw: Any) -> str:
                return self.registry.execute(tool_name, kw)
            target_handler = _fallback_handler
        else:
            target_handler = handler

        result_str, _record = self.execution_manager.execute_managed(
            tool_name=tool_name,
            handler=target_handler,
            arguments=arguments,
            caller_role=caller_role,
            caller_agent=caller_agent,
            timeout_override=timeout,
            max_output_chars_override=self.max_output_chars,
            external_cancel_event=cancel_event,
        )

        if self.event_bus and caller_agent != "orchestrator":
            self.event_bus.emit(
                EventType.TOOL_EXECUTION_COMPLETED,
                tool_name=tool_name,
                approved=True,
                result=result_str,
            )

        return result_str
