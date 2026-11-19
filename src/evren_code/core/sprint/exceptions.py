"""Sprint Exceptions module."""

from __future__ import annotations


class SprintError(Exception):
    """Sprint error implementation."""
    pass


class SprintPlanningError(SprintError):
    """Sprint planning error implementation."""
    pass


class SprintValidationError(SprintError):
    """Sprint validation error implementation."""
    pass


class SprintExecutionError(SprintError):
    """Sprint execution error implementation."""
    pass


class AcceptanceVerificationError(SprintError):
    """Acceptance verification error implementation."""
    pass


class ApprovalRequiredError(SprintError):
    """Approval required error implementation."""
    pass


class SprintInterruptedError(SprintError):
    """Sprint interrupted error implementation."""
    pass


class SprintRecoveryError(SprintError):
    """Sprint recovery error implementation."""
    pass
