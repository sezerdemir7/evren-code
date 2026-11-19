"""Regression module."""

from __future__ import annotations

from typing import List, Optional, Set

from evren_code.core.verification.models import (
    RegressionReport,
    TestExecutionResult,
    VerificationResult,
)


class RegressionDetector:
    """Regression detector implementation."""

    def __init__(self, baseline_result: Optional[VerificationResult] = None) -> None:
        self.baseline_result = baseline_result
        self._baseline_failed_tests: Set[str] = set()
        self._baseline_passed_tests: Set[str] = set()

        if baseline_result:
            self._extract_baseline_stats(baseline_result)

    def _extract_baseline_stats(self, res: VerificationResult) -> None:
        for step in res.steps:
            for f in step.failures:
                self._baseline_failed_tests.add(f.test_name)
            if step.exit_code == 0:
                self._baseline_passed_tests.add(step.step_name or step.command)

    def detect_regressions(
        self,
        current_result: VerificationResult,
    ) -> RegressionReport:
        """Detect regressions (baseline)."""
        if self.baseline_result is None:
            current_failures = []
            for step in current_result.steps:
                for f in step.failures:
                    current_failures.append(f.test_name)

            return RegressionReport(
                new_regressions=[],
                resolved_failures=[],
                remaining_failures=current_failures,
                is_regression_detected=False,
                is_baseline_missing=True,
            )

        current_failed_tests: Set[str] = set()
        for step in current_result.steps:
            for f in step.failures:
                current_failed_tests.add(f.test_name)

        new_regressions: List[str] = []
        for tf in current_failed_tests:
            if tf not in self._baseline_failed_tests:
                new_regressions.append(tf)

        resolved_failures: List[str] = []
        for bf in self._baseline_failed_tests:
            if bf not in current_failed_tests:
                resolved_failures.append(bf)

        remaining_failures: List[str] = []
        for cf in current_failed_tests:
            if cf in self._baseline_failed_tests:
                remaining_failures.append(cf)

        is_regression = len(new_regressions) > 0

        return RegressionReport(
            new_regressions=sorted(new_regressions),
            resolved_failures=sorted(resolved_failures),
            remaining_failures=sorted(remaining_failures),
            is_regression_detected=is_regression,
            is_baseline_missing=False,
        )
