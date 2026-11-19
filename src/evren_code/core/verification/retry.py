"""Autonomous Verification Retry & Loop Controller (`retry.py`).

Manages boundaries for verification and self-healing cycles:
- Maximum attempt limit (default: 3)
- Maximum cycle limit (default: 4)
- Category-based policy controls
- Deadlock / infinite loop detection
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from evren_code.core.verification.models import (
    CorrectionAttempt,
    FailureCategory,
    FailureClassification,
    RegressionReport,
    VerificationStatus,
)


class RetryController:
    """Coordinates verification attempts, cycles, and loop limits."""

    def __init__(
        self,
        max_attempts: int = 3,
        max_cycles: int = 4,
        min_confidence: float = 0.5,
    ) -> None:
        self.max_attempts = max_attempts
        self.max_cycles = max_cycles
        self.min_confidence = min_confidence
        self.current_attempt: int = 0
        self.current_cycle: int = 0
        self.attempts: List[CorrectionAttempt] = []
        self._history_failures: List[List[str]] = []

    def record_attempt(self, attempt: CorrectionAttempt) -> None:
        """Records a new correction attempt."""
        self.attempts.append(attempt)
        self.current_attempt += 1

    def record_cycle(self) -> None:
        """Records a new verification cycle."""
        self.current_cycle += 1

    def should_retry(
        self,
        status: VerificationStatus,
        classifications: List[FailureClassification],
        regression: Optional[RegressionReport] = None,
    ) -> Tuple[bool, str]:
        """Evaluates whether another correction step should be attempted."""
        # 1. Already verified
        if status == VerificationStatus.VERIFIED:
            return False, "All verification steps and tests passed cleanly (VERIFIED / başarıyla tamamlandı)."

        # 2. Blocked by security policy or user rejection
        if status == VerificationStatus.BLOCKED:
            return False, "Verification operation blocked by security policy or user refusal (BLOCKED / Güvenlik)."

        # 3. Maximum cycle limit reached
        if self.current_cycle >= self.max_cycles:
            return (
                False,
                f"Maximum verification cycle limit reached ({self.max_cycles}) (Maksimum doğrulama döngü sınırına ulaşıldı). Terminating.",
            )

        # 4. Maximum attempt limit reached
        if self.current_attempt >= self.max_attempts:
            return (
                False,
                f"Maximum attempt limit reached ({self.max_attempts}) (Maksimum deneme sınırına ulaşıldı). Terminating.",
            )

        if not classifications:
            return False, "No active classified failure detected."

        # 5. Category-based policy controls
        if all(c.category == FailureCategory.ENVIRONMENT for c in classifications):
            return (
                False,
                "Failures are entirely environment-related (ENVIRONMENT). Human intervention required.",
            )

        if all(c.category == FailureCategory.PRE_EXISTING for c in classifications):
            return (
                False,
                "Failures are pre-existing issues prior to agent modifications (PRE_EXISTING).",
            )

        if all(c.category == FailureCategory.DEPENDENCY for c in classifications):
            return (
                False,
                "Failures stem from missing dependencies (DEPENDENCY / Bağımlılık). Cannot install automatically.",
            )

        if all(c.category == FailureCategory.TEST_CONFIGURATION for c in classifications):
            return (
                False,
                "Failures stem from test configuration (TEST_CONFIGURATION). Test suite cannot be altered blindly.",
            )

        if all(c.category == FailureCategory.UNKNOWN for c in classifications):
            return (
                False,
                "Root cause could not be verified (UNKNOWN). Safe automated code edit halted.",
            )

        agent_introduced_clfs = [
            c for c in classifications if c.category == FailureCategory.AGENT_INTRODUCED
        ]
        if not agent_introduced_clfs:
            return (
                False,
                "No agent-introduced (AGENT_INTRODUCED) fixable failure found.",
            )

        max_conf = max(c.confidence for c in agent_introduced_clfs)
        if max_conf < self.min_confidence:
            return (
                False,
                f"Agent failure confidence score ({max_conf:.2f}) below safe threshold ({self.min_confidence:.2f}) (güven skoru eşiğin altında).",
            )

        # 6. Infinite loop / deadlock check
        current_fails = sorted([c.evidence for c in classifications])
        self._history_failures.append(current_fails)
        if len(self._history_failures) >= 3:
            if self._history_failures[-1] == self._history_failures[-2] == self._history_failures[-3]:
                return (
                    False,
                    "No improvement detected across recent attempts (kısırdöngü durduruldu).",
                )

        # 7. Critical regression check
        if regression and regression.is_regression_detected and len(regression.new_regressions) > 3:
            return (
                False,
                f"Critical regression detected ({len(regression.new_regressions)} new failing tests). Stopped for safety.",
            )

        return True, f"Correction attempt allowed / Düzeltme denemesi yapılabilir (Attempt #{self.current_attempt + 1}/{self.max_attempts})."

    def reset(self) -> None:
        """Resets attempt and cycle counters."""
        self.current_attempt = 0
        self.current_cycle = 0
        self.attempts.clear()
        self._history_failures.clear()
