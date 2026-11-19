"""Classifier module."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from evren_code.core.verification.models import (
    FailureCategory,
    FailureClassification,
    TestExecutionResult,
    TestFailure,
)


class FailureClassifier:
    """Failure classifier implementation."""

    ENV_PATTERNS = [
        re.compile(r"permission denied", re.IGNORECASE),
        re.compile(r"address already in use", re.IGNORECASE),
        re.compile(r"connection refused", re.IGNORECASE),
        re.compile(r"command not found:? (\S+)", re.IGNORECASE),
        re.compile(r"'(\S+)' is not recognized as an internal or external command", re.IGNORECASE),
        re.compile(r"no such file or directory:? '?(?:[a-zA-Z]:|\/)(?:usr|bin|etc|tmp)", re.IGNORECASE),
        re.compile(r"docker daemon is not running", re.IGNORECASE),
        re.compile(r"out of memory", re.IGNORECASE),
    ]

    DEP_PATTERNS = [
        re.compile(r"ModuleNotFoundError: No module named '([^']+)'", re.IGNORECASE),
        re.compile(r"ImportError: cannot import name '([^']+)'", re.IGNORECASE),
        re.compile(r"ClassNotFoundException: (\S+)", re.IGNORECASE),
        re.compile(r"NoClassDefFoundError: (\S+)", re.IGNORECASE),
        re.compile(r"Cannot find module '([^']+)'", re.IGNORECASE),
        re.compile(r"npm ERR! code (?:ENOENT|E404|ERESOLVE)", re.IGNORECASE),
        re.compile(r"Could not resolve dependencies", re.IGNORECASE),
        re.compile(r"pip install", re.IGNORECASE),
    ]

    CONFIG_PATTERNS = [
        re.compile(r"usage: pytest", re.IGNORECASE),
        re.compile(r"pytest: error: unrecognized arguments", re.IGNORECASE),
        re.compile(r"jest-haste-map: Haste module naming collision", re.IGNORECASE),
        re.compile(r"configuration error", re.IGNORECASE),
        re.compile(r"Invalid test pattern", re.IGNORECASE),
        re.compile(r"unrecognized option", re.IGNORECASE),
        re.compile(r"Failed to execute goal.*surefire", re.IGNORECASE),
    ]

    PYTEST_FAIL_RE = re.compile(r"(?:FAILED|ERROR)\s+([^\s:]+)::([^\s\-]+)(?:\s+-\s+(.+))?")
    # Python unittest: FAIL: test_name (tests.test_mod.TestCase)
    UNITTEST_FAIL_RE = re.compile(r"(?:FAIL|ERROR):\s+([^\s]+)\s+\(([^)]+)\)")
    # Jest/Vitest: ✕ test description
    JEST_FAIL_RE = re.compile(r"(?:✕|FAIL)\s+(.+)")
    # Java JUnit: [ERROR] testMethod(com.example.TestClass)  Time elapsed: ... <<< FAILURE!
    JUNIT_FAIL_RE = re.compile(r"\[ERROR\]\s+([a-zA-Z0-9_]+)\(([^)]+)\)")

    def __init__(
        self,
        changed_files: Optional[List[str]] = None,
        baseline_failures: Optional[Set[str]] = None,
    ) -> None:
        self.changed_files = [f.replace("\\", "/").lower() for f in (changed_files or [])]
        self.baseline_failures = baseline_failures or set()

    def parse_and_classify(self, exec_result: TestExecutionResult) -> List[TestFailure]:
        """Parse and classify."""
        raw_text = f"{exec_result.stdout}\n{exec_result.stderr}"
        failures = self._extract_individual_failures(raw_text)

        if not failures and exec_result.exit_code != 0:
            failures.append(
                TestFailure(
                    test_name=exec_result.step_name or exec_result.command,
                    failure_message=exec_result.stderr.strip() or exec_result.stdout[-1000:].strip(),
                    stack_trace=raw_text[-3000:],
                )
            )

        for f in failures:
            f.classification = self.classify_failure(f, raw_text)

        return failures

    def classify_failure(self, failure: TestFailure, full_output: str = "") -> FailureClassification:
        """Classify failure."""
        fail_text = f"{failure.test_name} {failure.failure_message} {failure.stack_trace} {full_output}"

        if failure.test_name in self.baseline_failures:
            return FailureClassification(
                category=FailureCategory.PRE_EXISTING,
                confidence=0.95,
                evidence=f"Test '{failure.test_name}' ajan değişikliklerinden önceki temel (baseline) koşumda da başarısızdı.",
                related_files=[failure.file_path] if failure.file_path else [],
                suggested_action="Bu hata mevcut kod tabanından kaynaklanıyor. Ajanın yeni değişiklikleriyle doğrudan ilgili olmayabilir.",
            )

        for pat in self.ENV_PATTERNS:
            m = pat.search(fail_text)
            if m:
                return FailureClassification(
                    category=FailureCategory.ENVIRONMENT,
                    confidence=0.90,
                    evidence=f"Ortam hatası tespit edildi: '{m.group(0)}'",
                    related_files=[],
                    suggested_action="Çalışma ortamı bağımlılıklarını (servisler, izinler, portlar, CLI araçları) kontrol edin.",
                )

        for pat in self.DEP_PATTERNS:
            m = pat.search(fail_text)
            if m:
                return FailureClassification(
                    category=FailureCategory.DEPENDENCY,
                    confidence=0.88,
                    evidence=f"Eksik bağımlılık tespit edildi: '{m.group(0)}'",
                    related_files=[],
                    suggested_action="Eksik kütüphane veya modülü paket yöneticisi ile projeye ekleyin.",
                )

        for pat in self.CONFIG_PATTERNS:
            m = pat.search(fail_text)
            if m:
                return FailureClassification(
                    category=FailureCategory.TEST_CONFIGURATION,
                    confidence=0.85,
                    evidence=f"Test yapılandırma hatası: '{m.group(0)}'",
                    related_files=[],
                    suggested_action="Test komutunun bayraklarını veya test ayar dosyasını (pytest.ini, jest.config vb.) inceleyin.",
                )

        related_changed: List[str] = []
        fail_text_low = fail_text.lower().replace("\\", "/")

        for cf in self.changed_files:
            cf_stem = Path(cf).stem.lower()
            if cf in fail_text_low or cf_stem in fail_text_low:
                related_changed.append(cf)

        if related_changed or (failure.file_path and failure.file_path.lower().replace("\\", "/") in self.changed_files):
            return FailureClassification(
                category=FailureCategory.AGENT_INTRODUCED,
                confidence=0.90 if related_changed else 0.75,
                evidence=(
                    f"Ajan tarafından değiştirilen dosyalar hata izinde yer alıyor: {', '.join(related_changed)}"
                    if related_changed
                    else f"Hata veren dosya ajanın değiştirdiği dosyalar arasında: {failure.file_path}"
                ),
                related_files=related_changed or ([failure.file_path] if failure.file_path else []),
                suggested_action="Son yapılan kod değişikliklerini ve ilgili sembol implementasyonunu inceleyip düzeltin.",
            )

        if "assertionerror" in fail_text_low or "syntaxerror" in fail_text_low:
            return FailureClassification(
                category=FailureCategory.AGENT_INTRODUCED,
                confidence=0.70,
                evidence="Doğrulama iddiası (Assertion) veya sözdizimi (Syntax) başarısız oldu.",
                related_files=[failure.file_path] if failure.file_path else [],
                suggested_action="Testin beklediği dönüş değeri veya davranış ile fonksiyon çıktısını karşılaştırın.",
            )

        # 6. UNKNOWN Fallback
        return FailureClassification(
            category=FailureCategory.UNKNOWN,
            confidence=0.30,
            evidence="Hatanın kök nedeni kesin olarak belirlenemedi.",
            related_files=[failure.file_path] if failure.file_path else [],
            suggested_action="Hata yığın izini (stack trace) ve stderr çıktılarını manuel olarak inceleyin.",
        )

    def _extract_individual_failures(self, text: str) -> List[TestFailure]:
        """Extract individual failures."""
        failures: List[TestFailure] = []
        seen_names: Set[str] = set()

        # 1. Pytest regex
        for m in self.PYTEST_FAIL_RE.finditer(text):
            fpath, tname, msg = m.groups()
            full_tname = f"{fpath}::{tname}"
            if full_tname not in seen_names:
                seen_names.add(full_tname)
                failures.append(
                    TestFailure(
                        test_name=full_tname,
                        file_path=fpath,
                        failure_message=msg or "Pytest failure",
                        stack_trace=self._extract_surrounding_trace(text, full_tname),
                    )
                )

        # 2. Unittest regex
        for m in self.UNITTEST_FAIL_RE.finditer(text):
            tname, tcase = m.groups()
            full_tname = f"{tcase}.{tname}"
            if full_tname not in seen_names:
                seen_names.add(full_tname)
                failures.append(
                    TestFailure(
                        test_name=full_tname,
                        file_path=tcase.replace(".", "/") + ".py",
                        failure_message="Unittest assertion failure or error",
                        stack_trace=self._extract_surrounding_trace(text, tname),
                    )
                )

        # 3. JUnit regex
        for m in self.JUNIT_FAIL_RE.finditer(text):
            tname, tclass = m.groups()
            full_tname = f"{tclass}#{tname}"
            if full_tname not in seen_names:
                seen_names.add(full_tname)
                failures.append(
                    TestFailure(
                        test_name=full_tname,
                        file_path=tclass.replace(".", "/") + ".java",
                        failure_message="JUnit test failure",
                        stack_trace=self._extract_surrounding_trace(text, tname),
                    )
                )

        # 4. Jest / Vitest regex
        for m in self.JEST_FAIL_RE.finditer(text):
            tname = m.group(1).strip()
            if tname and tname not in seen_names and len(tname) < 100:
                seen_names.add(tname)
                failures.append(
                    TestFailure(
                        test_name=tname,
                        failure_message="Jest/Vitest assertion failed",
                        stack_trace=self._extract_surrounding_trace(text, tname),
                    )
                )

        return failures

    def _extract_surrounding_trace(self, text: str, anchor: str, max_lines: int = 25) -> str:
        """Extract surrounding trace."""
        lines = text.splitlines()
        anchor_idx = -1
        for i, l in enumerate(lines):
            if anchor in l:
                anchor_idx = i
                break

        if anchor_idx == -1:
            return ""

        start = max(0, anchor_idx - 5)
        end = min(len(lines), anchor_idx + max_lines)
        return "\n".join(lines[start:end])
