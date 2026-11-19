"""Autonomous Self-Healing Module (`self_healing.py`).

Runs compilers, linters, and test suites automatically after code generation or editing.
If an error or compilation failure occurs, feeds the diagnostics back into the model to
autonomously repair the code up to max_attempts (default: 3) before returning to the user.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple


class ErrorCategory(str, Enum):
    """Self-healing and loop-breaker error categories."""
    COMMAND_NOT_FOUND = "command_not_found"
    COMPILATION_ERROR = "compilation_error"
    TEST_FAILURE = "test_failure"
    FILE_OPERATION_ERROR = "file_operation_error"
    PERMISSION_DENIED = "permission_denied"
    NETWORK_API_ERROR = "network_api_error"
    MODEL_TOOL_ERROR = "model_tool_error"
    TIMEOUT = "timeout"
    REPEATED_EXECUTION_ERROR = "repeated_execution_error"


class ErrorClassifier:
    """Classifies error outputs and command responses into 9 structured categories."""

    @classmethod
    def classify(cls, output: str, exit_code: Optional[int] = None, command: str = "") -> Tuple[ErrorCategory, str]:
        out_lower = (output or "").lower()
        cmd_lower = (command or "").lower()

        # 1. PERMISSION_DENIED
        if any(term in out_lower for term in (
            "güvenlik engeli", "güvenlik ihlali", "access_violation", "permission denied",
            "system_dir_violation", "kullanıcı reddi", "erişim engellendi", "unauthorized",
            "operation not permitted"
        )):
            return ErrorCategory.PERMISSION_DENIED, "Güvenlik politikası veya dosya/kabuk izin engeli."

        # 2. COMMAND_NOT_FOUND
        if any(term in out_lower for term in (
            "is not recognized as an internal or external command",
            "command not found",
            "komut yok",
            "terim bir cmdlet, işlev",
            "bulunamıyor",
            "no such file or directory: 'mvn",
            "no such file or directory: 'gradle",
            "no such file or directory: 'npm",
            "no such file or directory: 'pytest",
        )):
            return ErrorCategory.COMMAND_NOT_FOUND, "Komut veya yürütülebilir dosya sistemde/PATH içinde bulunamadı."

        # 3. TIMEOUT
        if any(term in out_lower for term in ("timed out", "zaman aşımı", "timeoutexpired", "etimedout")):
            return ErrorCategory.TIMEOUT, "Komut yürütme zaman aşımına uğradı."

        # 4. REPEATED_EXECUTION_ERROR
        if any(term in out_lower for term in ("döngü koruması", "loop breaker", "hard circuit breaker", "otonom rota değiştirme")):
            return ErrorCategory.REPEATED_EXECUTION_ERROR, "Aynı komut veya aracın kilitlenen mükerrer döngüsü."

        # 5. COMPILATION_ERROR
        if any(term in out_lower for term in (
            "syntaxerror", "compilation error", "error ts", "cannot find symbol",
            "javac:", "derleme hatası", "sözdizimi hatası", "cargo check failed",
            "failed to compile", "parse error", "indentationerror"
        )):
            return ErrorCategory.COMPILATION_ERROR, "Kod derleme, sözdizimi veya tip kontrolü hatası."

        # 6. TEST_FAILURE
        if any(term in out_lower for term in (
            "assertionerror", "failures=", "failures:", "failed (failures=",
            "tests failed", "<<< failure!", "[error] <<< failure!", "fail: test",
            "✕ test", "failing tests", "failed:", "assertions failed"
        )):
            return ErrorCategory.TEST_FAILURE, "Birim veya entegrasyon test iddiası (assertion) başarısızlığı."

        # 7. FILE_OPERATION_ERROR
        if any(term in out_lower for term in (
            "filenotfounderror", "dosyası bulunamadı", "dosya bulunamadı", "hedef blok bulunamadı",
            "target bloğu bulunamadı", "aranan target", "aranan blok bulunamadı", "target not found",
            "ambiguous target", "çoklu eşleşme", "dosya bütünlük hatası", "güvenilirlik hatası",
            "bir dizindir", "no such file"
        )):
            return ErrorCategory.FILE_OPERATION_ERROR, "Dosya okuma, yazma, arama veya cerrahi blok hedefleme hatası."

        # 8. NETWORK_API_ERROR
        if any(term in out_lower for term in (
            "connectionrefused", "connection refused", "econnrefused", "502 bad gateway",
            "503 service unavailable", "network error", "could not resolve host"
        )):
            return ErrorCategory.NETWORK_API_ERROR, "Harici ağ veya API servis bağlantı hatası."

        # 9. MODEL_TOOL_ERROR
        if any(term in out_lower for term in (
            "jsondecodeerror", "invalid tool_call", "şemasına uygun değil", "argümanında eksik",
            "geçerli bir json nesnesi"
        )):
            return ErrorCategory.MODEL_TOOL_ERROR, "Model araç çağrısı veya JSON argüman biçim hatası."

        if exit_code is not None and exit_code != 0:
            return ErrorCategory.COMPILATION_ERROR, f"Komut sıfır dışı çıkış kodu ile bitti ({exit_code})."

        return ErrorCategory.TEST_FAILURE, "Genel doğrulama başarısızlığı."


@dataclass
class VerificationTask:
    """Verification task implementation."""

    command: str
    cwd: Path
    description: str


class SelfHealingManager:
    """Self healing manager implementation."""

    DEFAULT_CATEGORY_LIMITS: Dict[ErrorCategory, int] = {
        ErrorCategory.COMMAND_NOT_FOUND: 1,
        ErrorCategory.COMPILATION_ERROR: 3,
        ErrorCategory.TEST_FAILURE: 3,
        ErrorCategory.FILE_OPERATION_ERROR: 2,
        ErrorCategory.PERMISSION_DENIED: 0,
        ErrorCategory.NETWORK_API_ERROR: 1,
        ErrorCategory.MODEL_TOOL_ERROR: 2,
        ErrorCategory.TIMEOUT: 1,
        ErrorCategory.REPEATED_EXECUTION_ERROR: 0,
    }

    def __init__(
        self,
        workspace_root: Path,
        max_attempts: int = 3,
        enabled: bool = True,
        auto_detect: bool = True,
        category_limits: Optional[Dict[ErrorCategory, int]] = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.max_attempts = max_attempts
        self.enabled = enabled
        self.auto_detect = auto_detect
        self._current_attempt = 0
        self._active_task: Optional[VerificationTask] = None
        self.category_limits = dict(self.DEFAULT_CATEGORY_LIMITS)
        if category_limits:
            self.category_limits.update(category_limits)
        self.category_attempts: Dict[ErrorCategory, int] = {cat: 0 for cat in ErrorCategory}
        self.history: List[Dict[str, Any]] = []

    def reset_turn(self) -> None:
        """Reset turn."""
        self._current_attempt = 0
        self.category_attempts = {cat: 0 for cat in ErrorCategory}
        self.history.clear()

    @property
    def current_attempt(self) -> int:
        return self._current_attempt

    @property
    def has_attempts_left(self) -> bool:
        return self._current_attempt < self.max_attempts

    def check_workspace_wrapper(self, command: str, cwd: Optional[Path] = None) -> Optional[str]:
        """Check workspace wrapper (mvnw, gradlew vb.)."""
        cmd_stripped = command.strip()
        search_dirs = [cwd, self.workspace_root] if cwd else [self.workspace_root]

        if cmd_stripped.startswith("mvn ") or cmd_stripped == "mvn":
            for d in search_dirs:
                if not d:
                    continue
                if (d / "mvnw.cmd").is_file():
                    return cmd_stripped.replace("mvn", "mvnw.cmd", 1) if d == cwd else cmd_stripped.replace("mvn", f'"{d / "mvnw.cmd"}"', 1)
                if (d / "mvnw").is_file():
                    return cmd_stripped.replace("mvn", "./mvnw", 1) if d == cwd else cmd_stripped.replace("mvn", f'"{d / "mvnw"}"', 1)

        if cmd_stripped.startswith("gradle ") or cmd_stripped == "gradle":
            for d in search_dirs:
                if not d:
                    continue
                if (d / "gradlew.bat").is_file():
                    return cmd_stripped.replace("gradle", "gradlew.bat", 1) if d == cwd else cmd_stripped.replace("gradle", f'"{d / "gradlew.bat"}"', 1)
                if (d / "gradlew").is_file():
                    return cmd_stripped.replace("gradle", "./gradlew", 1) if d == cwd else cmd_stripped.replace("gradle", f'"{d / "gradlew"}"', 1)

        return None

    def should_retry(self, category: ErrorCategory, error_sig: str) -> Tuple[bool, str]:
        """Should retry."""
        cat_limit = self.category_limits.get(category, 1)
        used = self.category_attempts.get(category, 0)

        if used >= cat_limit:
            return False, (
                f"Kategori Tavanı Aşıldı: '{category.value}' türü hatalar için izin verilen azami deneme ({cat_limit}) "
                "sayısına ulaşıldı. Gereksiz token harcamasını ve sonsuz döngüyü önlemek için durduruldu."
            )

        if not self.has_attempts_left:
            return False, f"Genel Self-Healing tavanı ({self.max_attempts} deneme) dolduruldu."

        identical_count = sum(1 for h in self.history if h.get("category") == category and h.get("error_sig") == error_sig)
        if identical_count >= 2:
            return False, (
                f"Tekrarlayan Hata Döngüsü: '{category.value}' hatası önceki adımlarla birebir aynı şekilde "
                f"({identical_count} kez) tekrar etti ve çözüm stratejisi başarısız oldu. Rota kilitlenmesini önlemek için durduruldu."
            )

        return True, "Deneme yapılabilir."

    def get_strategy_recommendation(
        self,
        category: ErrorCategory,
        failed_command: str,
        error_snippet: str,
    ) -> str:
        """Get strategy recommendation."""
        if re.search(r"\*\*\*MASKED", error_snippet):
            return (
                "🛡️ [Test Harness Maskeleme Uyarısı]: Çıktıda görünen '***MASKED***' veya '***' kalıpları test ortamının "
                "güvenlik filtresidir. Bu bir uygulama veya token hatası DEĞİLDİR! Asla bu kalıbı kaynak kodda aramayın, "
                "derlenmiş .class veya ikili dosyaları incelemeyin. Doğrudan kaynak kod mantığına odaklanın."
            )

        if category == ErrorCategory.COMMAND_NOT_FOUND:
            wrapper = self.check_workspace_wrapper(failed_command)
            wrapper_note = f" Workspace içinde yerel wrapper tespit edildi: `{wrapper}`. Lütfen bu komutu kullanın." if wrapper else ""
            return (
                f"⚠️ [Eksik Komut Uyarısı]: `{failed_command}` komutu sistemde/PATH içinde bulunamadı.{wrapper_note}\n"
                "GÜVENLİK KURALI: Asla Windows PATH değişkenini değiştirmeyin, Ollama veya sistem dizinlerine shim dosyası yazmayın "
                "ve global paket kurmayın. Varsa çalışma alanı içindeki wrapper'ı (`mvnw` / `gradlew`) kullanın; "
                "yoksa durumu kullanıcıya açıkça raporlayın."
            )

        if category == ErrorCategory.PERMISSION_DENIED:
            return (
                "🛑 [Güvenlik ve Yetki Engeli]: Bu eylem workspace containment veya güvenlik politikası tarafından durduruldu. "
                "Güvenlik sınırlarını aşmaya çalışmayın; sadece çalışma alanı içindeki kaynak kodlara odaklanın."
            )

        if category == ErrorCategory.COMPILATION_ERROR:
            return (
                "💡 [Derleme / Sözdizimi Düzeltme Önerisi]: Derleyicinin işaret ettiği satır numarasını ve eksik sembolü inceleyin. "
                "Tüm dosyayı yeniden yazmak yerine `read_file` ile güncel satırları kontrol edip `edit_file` ile cerrahi düzeltme yapın."
            )

        if category == ErrorCategory.TEST_FAILURE:
            return (
                "💡 [Test Başarısızlığı Düzeltme Önerisi]: Kırılan testin beklediği girdi/çıktı sözleşmesini (assertion) inceleyin. "
                "Fonksiyon mantığındaki uç durumları (edge-cases) veya dönüş tipini kontrol edin."
            )

        if category == ErrorCategory.TIMEOUT:
            return (
                "⏳ [Zaman Aşımı Uyarısı]: Komut çok uzun sürdü. Tüm test paketi yerine sadece ilgili tekil test dosyasını çalıştırın."
            )

        return "Farklı bir çözüm stratejisi uygulayın; önceki hataya yol açan aynı adımı tekrarlamayın."

    def record_attempt(
        self,
        category: ErrorCategory,
        command: str,
        error_sig: str,
        notes: str = "",
    ) -> None:
        """Record attempt."""
        self.category_attempts[category] = self.category_attempts.get(category, 0) + 1
        self.history.append({
            "attempt": self._current_attempt,
            "category": category,
            "command": command,
            "error_sig": error_sig,
            "notes": notes,
        })

    def detect_verification_task(
        self,
        modified_paths: Optional[List[str]] = None,
    ) -> Optional[VerificationTask]:
        """Detect verification task."""
        if self._active_task is not None and not modified_paths:
            return self._active_task

        if not self.auto_detect:
            return self._active_task

        candidate_dirs: List[Path] = []
        if modified_paths:
            for mp in modified_paths:
                parts = Path(mp.replace("\\", "/")).parts
                if len(parts) > 1:
                    sub_cand = self.workspace_root / parts[0]
                    if sub_cand.is_dir() and sub_cand not in candidate_dirs:
                        candidate_dirs.append(sub_cand)

        if self.workspace_root not in candidate_dirs:
            candidate_dirs.append(self.workspace_root)

        try:
            for item in self.workspace_root.iterdir():
                if (
                    item.is_dir()
                    and not item.name.startswith((".", "_"))
                    and item.name not in ("node_modules", "venv")
                    and item not in candidate_dirs
                ):
                    candidate_dirs.append(item)
        except OSError:
            pass

        for cdir in candidate_dirs:
            # 1. Node / TypeScript / Next.js
            pkg_file = cdir / "package.json"
            if pkg_file.is_file():
                try:
                    data = json.loads(pkg_file.read_text(encoding="utf-8", errors="ignore"))
                    scripts = data.get("scripts", {})
                    if "typecheck" in scripts:
                        self._active_task = VerificationTask("npm run typecheck", cdir, "TypeScript Tip Kontrolü")
                        return self._active_task
                    if "test" in scripts and "no test specified" not in scripts.get("test", ""):
                        self._active_task = VerificationTask("npm test -- --passWithNoTests", cdir, "Birim Testler")
                        return self._active_task
                    if "build" in scripts:
                        self._active_task = VerificationTask("npm run build", cdir, "Derleme Kontrolü")
                        return self._active_task
                except Exception:
                    pass

            # 2. Python (pytest / unittest)
            if (cdir / "pytest.ini").is_file() or (cdir / "pyproject.toml").is_file():
                if (cdir / "tests").is_dir() or (cdir / "test").is_dir():
                    target_test_file = None
                    if modified_paths:
                        for mp in modified_paths:
                            p_obj = Path(mp)
                            cand_name = f"test_{p_obj.stem}.py"
                            for t_dir in (cdir / "tests", cdir / "test"):
                                if (t_dir / cand_name).is_file():
                                    target_test_file = f"{t_dir.name}/{cand_name}"
                                    break
                            if target_test_file:
                                break
                    if target_test_file:
                        py_cmd = f'"{sys.executable}" -m pytest {target_test_file} -q'
                        self._active_task = VerificationTask(py_cmd, cdir, f"Python Hedefli Test ({target_test_file})")
                    else:
                        py_cmd = f'"{sys.executable}" -m unittest discover'
                        self._active_task = VerificationTask(py_cmd, cdir, "Python Test Paketi")
                    return self._active_task

            # 3. Java (Maven / Gradle)
            if (cdir / "pom.xml").is_file():
                mvn_cmd = "mvnw.cmd test-compile -q" if (cdir / "mvnw.cmd").is_file() else "mvn test-compile -q"
                self._active_task = VerificationTask(mvn_cmd, cdir, "Java Maven Derleme Kontrolü")
                return self._active_task

            if (cdir / "build.gradle").is_file() or (cdir / "build.gradle.kts").is_file():
                gradle_cmd = "gradlew.bat compileJava -q" if (cdir / "gradlew.bat").is_file() else "gradle compileJava -q"
                self._active_task = VerificationTask(gradle_cmd, cdir, "Java/Kotlin Gradle Derleme Kontrolü")
                return self._active_task

            # 4. Flutter / Dart
            if (cdir / "pubspec.yaml").is_file():
                self._active_task = VerificationTask("flutter analyze --no-pub", cdir, "Flutter Statik Analiz")
                return self._active_task

            # 5. Rust (Cargo)
            if (cdir / "Cargo.toml").is_file():
                self._active_task = VerificationTask("cargo check", cdir, "Rust Derleme ve Tip Kontrolü")
                return self._active_task

            # 6. Go
            if (cdir / "go.mod").is_file():
                self._active_task = VerificationTask("go test ./...", cdir, "Go Test Paketi")
                return self._active_task

        return None

    def execute_verification(
        self,
        command_executor: Callable[[str, Optional[str]], str],
        task: Optional[VerificationTask] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Execute verification (False, command, output)."""
        vtask = task or self.detect_verification_task()
        if not vtask:
            return True, "", None

        self._current_attempt += 1
        try:
            rel_cwd = vtask.cwd.relative_to(self.workspace_root).as_posix()
            cwd_arg = rel_cwd if rel_cwd != "." else None
        except ValueError:
            cwd_arg = str(vtask.cwd)

        eff_command = self.check_workspace_wrapper(vtask.command, vtask.cwd) or vtask.command
        raw_output = command_executor(eff_command, cwd_arg)
        is_success = "(Exit Code): 0" in raw_output or "Exit Code: 0" in raw_output or "Exit Code: None" in raw_output

        if is_success:
            return True, eff_command, raw_output
        return False, eff_command, raw_output
