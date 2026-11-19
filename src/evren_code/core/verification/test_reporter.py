"""Test reporter module."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class TestVerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    UNVERIFIED_CLAIMS = "UNVERIFIED_CLAIMS"
    INCONSISTENT = "INCONSISTENT"
    NOT_RUN = "NOT_RUN"


@dataclass
class ParsedTestMetrics:
    """Parsed test metrics implementation."""
    framework: str  # "pytest", "unittest", "maven", "gradle", "jest", "cargo", "go", "unknown"
    tests_executed: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    exit_code: int = 0
    build_status: str = "UNKNOWN"  # "SUCCESS", "FAILED", "NOT_RUN"
    report_location: Optional[str] = None
    raw_summary: str = ""


@dataclass
class VerificationSummaryReport:
    """Verification summary report implementation."""
    build_status: str
    tests_executed: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    test_report_location: Optional[str]
    verification_status: TestVerificationStatus
    unverified_claims: List[str] = field(default_factory=list)
    remaining_issues: List[str] = field(default_factory=list)
    framework: str = "unknown"

    def to_markdown(self) -> str:
        """Returns human-readable Markdown output."""
        unverified_block = (
            "\n".join(f"  • ⚠️ {c}" for c in self.unverified_claims)
            if self.unverified_claims
            else "  • None (Verified / Doğrulandı)"
        )
        issues_block = (
            "\n".join(f"  • ❌ {i}" for i in self.remaining_issues)
            if self.remaining_issues
            else "  • None (All checks passed / Tüm kontroller başarılı)"
        )

        return (
            "### 📊 Verification & Test Report / Doğrulama ve Test Raporu\n"
            f"- **Build Status**: `{self.build_status}`\n"
            f"- **Test Framework**: `{self.framework}`\n"
            f"- **Tests Executed**: `{self.tests_executed}`\n"
            f"- **Tests Passed**: `{self.tests_passed}`\n"
            f"- **Tests Failed**: `{self.tests_failed}`\n"
            f"- **Tests Skipped**: `{self.tests_skipped}`\n"
            f"- **Test Report Location**: `{self.test_report_location or 'Console Standard Output (stdout)'}`\n"
            f"- **Verification Status**: `{self.verification_status.value}`\n"
            f"- **Unverified Claims (Kanıtlanamayan İddialar)**:\n{unverified_block}\n"
            f"- **Remaining Issues (Kalan Sorunlar)**:\n{issues_block}\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "build_status": self.build_status,
            "framework": self.framework,
            "tests_executed": self.tests_executed,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "test_report_location": self.test_report_location,
            "verification_status": self.verification_status.value,
            "unverified_claims": list(self.unverified_claims),
            "remaining_issues": list(self.remaining_issues),
        }


class TestOutputParser:
    """Test output parser implementation."""

    # 1. Pytest: "5 passed, 1 failed, 2 skipped in 1.23s"
    PYTEST_SUMMARY_RE = re.compile(
        r"=+\s*(?:(?P<passed>\d+)\s+passed)?(?:,\s*)?(?:(?P<failed>\d+)\s+failed)?(?:,\s*)?(?:(?P<skipped>\d+)\s+skipped)?(?:,\s*)?(?:(?P<error>\d+)\s+error(?:s)?)?.*in\s+[\d\.]+s\s*=+",
        re.IGNORECASE,
    )

    UNITTEST_RAN_RE = re.compile(r"Ran\s+(?P<total>\d+)\s+tests?\s+in\s+[\d\.]+s", re.IGNORECASE)
    UNITTEST_FAIL_RE = re.compile(r"FAILED\s*\((?:failures=(?P<failures>\d+))?(?:, )?(?:errors=(?P<errors>\d+))?(?:, )?(?:skipped=(?P<skipped>\d+))?\)", re.IGNORECASE)

    # 3. Maven Surefire: "Tests run: 10, Failures: 0, Errors: 0, Skipped: 0"
    MAVEN_SUREFIRE_RE = re.compile(
        r"Tests run:\s*(?P<total>\d+),\s*Failures:\s*(?P<failures>\d+),\s*Errors:\s*(?P<errors>\d+),\s*Skipped:\s*(?P<skipped>\d+)",
        re.IGNORECASE,
    )

    # 4. Gradle: "10 tests completed, 2 failed, 1 skipped"
    GRADLE_RE = re.compile(
        r"(?P<total>\d+)\s+tests?\s+completed,\s*(?P<failed>\d+)\s+failed(?:,\s*(?P<skipped>\d+)\s+skipped)?",
        re.IGNORECASE,
    )

    # 5. Jest / Vitest: "Tests: 2 failed, 1 skipped, 10 passed, 13 total"
    JEST_RE = re.compile(
        r"Tests:\s+(?:(?P<failed>\d+)\s+failed,?\s*)?(?:(?P<skipped>\d+)\s+skipped,?\s*)?(?:(?P<passed>\d+)\s+passed,?\s*)?(?P<total>\d+)\s+total",
        re.IGNORECASE,
    )

    # 6. Cargo test: "test result: ok. 15 passed; 0 failed; 2 ignored;"
    CARGO_RE = re.compile(
        r"test result:\s*(?P<status>ok|FAILED)\.\s*(?P<passed>\d+)\s+passed;\s*(?P<failed>\d+)\s+failed;\s*(?P<ignored>\d+)\s+ignored",
        re.IGNORECASE,
    )

    GO_PASS_RE = re.compile(r"^--- PASS:\s*(\S+)", re.MULTILINE)
    GO_FAIL_RE = re.compile(r"^--- FAIL:\s*(\S+)", re.MULTILINE)
    GO_SKIP_RE = re.compile(r"^--- SKIP:\s*(\S+)", re.MULTILINE)

    @classmethod
    def parse_output(
        cls,
        output: str,
        exit_code: int = 0,
        workspace_root: Optional[Path] = None,
    ) -> ParsedTestMetrics:
        """Parse output."""
        if workspace_root:
            xml_metrics = cls._parse_surefire_xmls(workspace_root, exit_code)
            if xml_metrics:
                return xml_metrics

        m_mvn = list(cls.MAVEN_SUREFIRE_RE.finditer(output))
        if m_mvn:
            tot = sum(int(m.group("total") or 0) for m in m_mvn)
            fails = sum(int(m.group("failures") or 0) for m in m_mvn)
            errs = sum(int(m.group("errors") or 0) for m in m_mvn)
            skips = sum(int(m.group("skipped") or 0) for m in m_mvn)
            total_failed = fails + errs
            passed = max(0, tot - total_failed - skips)
            build_st = "SUCCESS" if exit_code == 0 and total_failed == 0 else "FAILED"
            return ParsedTestMetrics(
                framework="maven",
                tests_executed=tot,
                tests_passed=passed,
                tests_failed=total_failed,
                tests_skipped=skips,
                exit_code=exit_code,
                build_status=build_st,
                raw_summary=f"Maven Surefire: {tot} run, {passed} passed, {total_failed} failed, {skips} skipped",
            )

        m_py = cls.PYTEST_SUMMARY_RE.search(output)
        if m_py:
            p = int(m_py.group("passed") or 0)
            f = int(m_py.group("failed") or 0)
            s = int(m_py.group("skipped") or 0)
            e = int(m_py.group("error") or 0)
            total_f = f + e
            tot = p + total_f + s
            build_st = "SUCCESS" if exit_code == 0 and total_f == 0 else "FAILED"
            return ParsedTestMetrics(
                framework="pytest",
                tests_executed=tot,
                tests_passed=p,
                tests_failed=total_f,
                tests_skipped=s,
                exit_code=exit_code,
                build_status=build_st,
                raw_summary=f"pytest: {tot} executed, {p} passed, {total_f} failed, {s} skipped",
            )

        m_unit_ran = cls.UNITTEST_RAN_RE.search(output)
        if m_unit_ran:
            tot = int(m_unit_ran.group("total") or 0)
            m_unit_fail = cls.UNITTEST_FAIL_RE.search(output)
            if m_unit_fail:
                f = int(m_unit_fail.group("failures") or 0)
                e = int(m_unit_fail.group("errors") or 0)
                s = int(m_unit_fail.group("skipped") or 0)
                total_f = f + e
                p = max(0, tot - total_f - s)
                build_st = "FAILED"
            else:
                total_f = 0
                s = output.count("... skipped")
                p = max(0, tot - s)
                build_st = "SUCCESS" if exit_code == 0 else "FAILED"

            return ParsedTestMetrics(
                framework="unittest",
                tests_executed=tot,
                tests_passed=p,
                tests_failed=total_f,
                tests_skipped=s,
                exit_code=exit_code,
                build_status=build_st,
                raw_summary=f"unittest: {tot} run, {p} passed, {total_f} failed, {s} skipped",
            )

        # 4. Jest / Vitest
        m_jest = cls.JEST_RE.search(output)
        if m_jest:
            tot = int(m_jest.group("total") or 0)
            p = int(m_jest.group("passed") or 0)
            f = int(m_jest.group("failed") or 0)
            s = int(m_jest.group("skipped") or 0)
            build_st = "SUCCESS" if exit_code == 0 and f == 0 else "FAILED"
            return ParsedTestMetrics(
                framework="jest",
                tests_executed=tot,
                tests_passed=p,
                tests_failed=f,
                tests_skipped=s,
                exit_code=exit_code,
                build_status=build_st,
                raw_summary=f"Jest/Vitest: {tot} total, {p} passed, {f} failed, {s} skipped",
            )

        # 5. Gradle
        m_grad = cls.GRADLE_RE.search(output)
        if m_grad:
            tot = int(m_grad.group("total") or 0)
            f = int(m_grad.group("failed") or 0)
            s = int(m_grad.group("skipped") or 0)
            p = max(0, tot - f - s)
            build_st = "SUCCESS" if exit_code == 0 and f == 0 else "FAILED"
            return ParsedTestMetrics(
                framework="gradle",
                tests_executed=tot,
                tests_passed=p,
                tests_failed=f,
                tests_skipped=s,
                exit_code=exit_code,
                build_status=build_st,
                raw_summary=f"Gradle: {tot} completed, {p} passed, {f} failed, {s} skipped",
            )

        # 6. Cargo test
        m_cargo = cls.CARGO_RE.search(output)
        if m_cargo:
            p = int(m_cargo.group("passed") or 0)
            f = int(m_cargo.group("failed") or 0)
            s = int(m_cargo.group("ignored") or 0)
            tot = p + f + s
            build_st = "SUCCESS" if exit_code == 0 and f == 0 else "FAILED"
            return ParsedTestMetrics(
                framework="cargo",
                tests_executed=tot,
                tests_passed=p,
                tests_failed=f,
                tests_skipped=s,
                exit_code=exit_code,
                build_status=build_st,
                raw_summary=f"Cargo: {tot} total, {p} passed, {f} failed, {s} ignored",
            )

        # 7. Go test
        go_passes = len(cls.GO_PASS_RE.findall(output))
        go_fails = len(cls.GO_FAIL_RE.findall(output))
        go_skips = len(cls.GO_SKIP_RE.findall(output))
        if go_passes > 0 or go_fails > 0 or go_skips > 0:
            tot = go_passes + go_fails + go_skips
            build_st = "SUCCESS" if exit_code == 0 and go_fails == 0 else "FAILED"
            return ParsedTestMetrics(
                framework="go",
                tests_executed=tot,
                tests_passed=go_passes,
                tests_failed=go_fails,
                tests_skipped=go_skips,
                exit_code=exit_code,
                build_status=build_st,
                raw_summary=f"Go: {tot} tests, {go_passes} passed, {go_fails} failed, {go_skips} skipped",
            )

        has_test_keyword = any(k in output.lower() for k in ("test", "passed", "failed", "assertions", "specs"))
        if has_test_keyword:
            is_success = exit_code == 0 and not any(k in output.lower() for k in ("failed", "error", "hata"))
            return ParsedTestMetrics(
                framework="generic",
                tests_executed=1 if is_success else 0,
                tests_passed=1 if is_success else 0,
                tests_failed=0 if is_success else 1,
                tests_skipped=0,
                exit_code=exit_code,
                build_status="SUCCESS" if is_success else "FAILED",
                raw_summary="Genel test göstergesi tespit edildi (exit code baz alındı)",
            )

        return ParsedTestMetrics(
            framework="unknown",
            tests_executed=0,
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            exit_code=exit_code,
            build_status="UNKNOWN",
            raw_summary="Test çıktısı tespit edilmedi",
        )

    @classmethod
    def _parse_surefire_xmls(cls, workspace_root: Path, exit_code: int) -> Optional[ParsedTestMetrics]:
        """Parse surefire xmls."""
        surefire_dirs = list(workspace_root.glob("**/target/surefire-reports"))
        if not surefire_dirs:
            return None

        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        xml_count = 0

        for sdir in surefire_dirs:
            for xml_file in sdir.glob("TEST-*.xml"):
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    total_tests += int(root.attrib.get("tests", 0))
                    total_failures += int(root.attrib.get("failures", 0))
                    total_errors += int(root.attrib.get("errors", 0))
                    total_skipped += int(root.attrib.get("skipped", 0))
                    xml_count += 1
                except Exception:
                    pass

        if xml_count == 0:
            return None

        total_failed = total_failures + total_errors
        passed = max(0, total_tests - total_failed - total_skipped)
        build_st = "SUCCESS" if exit_code == 0 and total_failed == 0 else "FAILED"
        first_rep = str(surefire_dirs[0].relative_to(workspace_root))

        return ParsedTestMetrics(
            framework="maven_surefire_xml",
            tests_executed=total_tests,
            tests_passed=passed,
            tests_failed=total_failed,
            tests_skipped=total_skipped,
            exit_code=exit_code,
            build_status=build_st,
            report_location=first_rep,
            raw_summary=f"Surefire XML ({xml_count} rapor): {total_tests} test, {passed} passed, {total_failed} failed, {total_skipped} skipped",
        )


class TestReportingEngine:
    """Test reporting engine implementation."""

    @classmethod
    def generate_report(
        cls,
        metrics: ParsedTestMetrics,
        agent_claimed_executed: Optional[int] = None,
        agent_claimed_passed: Optional[int] = None,
        remaining_issues: Optional[List[str]] = None,
    ) -> VerificationSummaryReport:
        unverified_claims: List[str] = []
        issues = list(remaining_issues or [])

        if agent_claimed_executed is not None and agent_claimed_executed != metrics.tests_executed:
            unverified_claims.append(
                f"Ajan {agent_claimed_executed} test çalıştırıldığını iddia etti, ancak test aracı gerçekte "
                f"{metrics.tests_executed} test çalıştırdı (Fark: {abs(agent_claimed_executed - metrics.tests_executed)})."
            )

        if agent_claimed_passed is not None and agent_claimed_passed != metrics.tests_passed:
            unverified_claims.append(
                f"Ajan {agent_claimed_passed} testin geçtiğini iddia etti, ancak test aracı gerçekte "
                f"{metrics.tests_passed} testin geçtiğini raporladı."
            )

        if unverified_claims:
            v_status = TestVerificationStatus.UNVERIFIED_CLAIMS
        elif metrics.tests_failed > 0 or metrics.build_status != "SUCCESS" or metrics.exit_code != 0:
            v_status = TestVerificationStatus.FAILED
            if not issues and metrics.tests_failed > 0:
                issues.append(f"{metrics.tests_failed} adet test başarısız oldu (Exit code: {metrics.exit_code}).")
        elif metrics.tests_executed == 0:
            v_status = TestVerificationStatus.NOT_RUN
        else:
            v_status = TestVerificationStatus.VERIFIED

        return VerificationSummaryReport(
            build_status=metrics.build_status,
            tests_executed=metrics.tests_executed,
            tests_passed=metrics.tests_passed,
            tests_failed=metrics.tests_failed,
            tests_skipped=metrics.tests_skipped,
            test_report_location=metrics.report_location,
            verification_status=v_status,
            unverified_claims=unverified_claims,
            remaining_issues=issues,
            framework=metrics.framework,
        )
