"""Models module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from evren_code.tools.shell import mask_sensitive_data


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


class FailureCategory(str, Enum):
    AGENT_INTRODUCED = "AGENT_INTRODUCED"
    PRE_EXISTING = "PRE_EXISTING"
    ENVIRONMENT = "ENVIRONMENT"
    DEPENDENCY = "DEPENDENCY"
    TEST_CONFIGURATION = "TEST_CONFIGURATION"
    UNKNOWN = "UNKNOWN"


@dataclass
class FailureClassification:
    """Failure classification implementation."""

    category: FailureCategory
    confidence: float  # 0.0 - 1.0
    evidence: str
    related_files: List[str] = field(default_factory=list)
    suggested_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "confidence": round(self.confidence, 3),
            "evidence": mask_sensitive_data(self.evidence),
            "related_files": list(self.related_files),
            "suggested_action": self.suggested_action,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FailureClassification:
        return cls(
            category=FailureCategory(data.get("category", FailureCategory.UNKNOWN.value)),
            confidence=float(data.get("confidence", 0.0)),
            evidence=data.get("evidence", ""),
            related_files=data.get("related_files", []),
            suggested_action=data.get("suggested_action", ""),
        )


@dataclass
class TestFailure:
    """Test failure implementation."""

    test_name: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    failure_message: str = ""
    stack_trace: str = ""
    classification: Optional[FailureClassification] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "failure_message": mask_sensitive_data(self.failure_message[:1000]),
            "stack_trace": mask_sensitive_data(self.stack_trace[:3000]),
            "classification": self.classification.to_dict() if self.classification else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestFailure:
        clf = FailureClassification.from_dict(data["classification"]) if data.get("classification") else None
        return cls(
            test_name=data.get("test_name", ""),
            file_path=data.get("file_path"),
            line_number=data.get("line_number"),
            failure_message=data.get("failure_message", ""),
            stack_trace=data.get("stack_trace", ""),
            classification=clf,
        )


@dataclass
class VerificationStep:
    """Verification step implementation."""

    name: str  # "pytest:unit", "mvn:test", "npm:typecheck"
    command: str  # "pytest tests/test_auth.py"
    cwd: str = "."
    target_type: str = "test"  # "test", "build", "lint", "typecheck"
    description: str = ""
    why_selected: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationStep:
        return cls(
            name=data.get("name", ""),
            command=data.get("command", ""),
            cwd=data.get("cwd", "."),
            target_type=data.get("target_type", "test"),
            description=data.get("description", ""),
            why_selected=data.get("why_selected", ""),
        )


@dataclass
class TestExecutionResult:
    """Test execution result implementation."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timestamp: str
    step_name: str = ""
    target_type: str = "test"
    passed_count: int = 0
    failed_count: int = 0
    failures: List[TestFailure] = field(default_factory=list)
    status: VerificationStatus = VerificationStatus.NOT_RUN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": mask_sensitive_data(self.stdout[:4000]),
            "stderr": mask_sensitive_data(self.stderr[:4000]),
            "duration_seconds": round(self.duration_seconds, 3),
            "timestamp": self.timestamp,
            "step_name": self.step_name,
            "target_type": self.target_type,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "failures": [f.to_dict() for f in self.failures],
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestExecutionResult:
        fails = [TestFailure.from_dict(f) for f in data.get("failures", [])]
        return cls(
            command=data.get("command", ""),
            exit_code=int(data.get("exit_code", -1)),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            timestamp=data.get("timestamp", ""),
            step_name=data.get("step_name", ""),
            target_type=data.get("target_type", "test"),
            passed_count=int(data.get("passed_count", 0)),
            failed_count=int(data.get("failed_count", 0)),
            failures=fails,
            status=VerificationStatus(data.get("status", VerificationStatus.NOT_RUN.value)),
        )


@dataclass
class VerificationResult:
    """Verification result implementation."""

    steps: List[TestExecutionResult] = field(default_factory=list)
    overall_status: VerificationStatus = VerificationStatus.NOT_RUN
    total_duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "overall_status": self.overall_status.value,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationResult:
        steps = [TestExecutionResult.from_dict(s) for s in data.get("steps", [])]
        return cls(
            steps=steps,
            overall_status=VerificationStatus(data.get("overall_status", VerificationStatus.NOT_RUN.value)),
            total_duration_seconds=float(data.get("total_duration_seconds", 0.0)),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class CorrectionAttempt:
    """Correction attempt implementation."""

    attempt_number: int
    failure_addressed: str
    plan_description: str
    modified_files: List[str] = field(default_factory=list)
    success: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CorrectionAttempt:
        return cls(
            attempt_number=int(data.get("attempt_number", 1)),
            failure_addressed=data.get("failure_addressed", ""),
            plan_description=data.get("plan_description", ""),
            modified_files=data.get("modified_files", []),
            success=bool(data.get("success", False)),
            notes=data.get("notes", ""),
        )


@dataclass
class RegressionReport:
    """Regression report implementation."""

    new_regressions: List[str] = field(default_factory=list)
    resolved_failures: List[str] = field(default_factory=list)
    remaining_failures: List[str] = field(default_factory=list)
    is_regression_detected: bool = False
    is_baseline_missing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RegressionReport:
        return cls(
            new_regressions=data.get("new_regressions", []),
            resolved_failures=data.get("resolved_failures", []),
            remaining_failures=data.get("remaining_failures", []),
            is_regression_detected=bool(data.get("is_regression_detected", False)),
            is_baseline_missing=bool(data.get("is_baseline_missing", False)),
        )


@dataclass
class VerificationReport:
    """Verification report implementation."""

    task_description: str
    implementation_status: str  # "SUCCESS", "FAILED", "PARTIAL"
    changed_files: List[str] = field(default_factory=list)
    impact_summary: List[str] = field(default_factory=list)
    step_results: List[TestExecutionResult] = field(default_factory=list)
    correction_attempts: List[CorrectionAttempt] = field(default_factory=list)
    failure_analysis: List[Dict[str, Any]] = field(default_factory=list)
    regression_analysis: Optional[RegressionReport] = None
    final_status: VerificationStatus = VerificationStatus.NOT_RUN
    total_duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_description": self.task_description,
            "implementation_status": self.implementation_status,
            "changed_files": list(self.changed_files),
            "impact_summary": list(self.impact_summary),
            "step_results": [s.to_dict() for s in self.step_results],
            "correction_attempts": [c.to_dict() for c in self.correction_attempts],
            "failure_analysis": self.failure_analysis,
            "regression_analysis": self.regression_analysis.to_dict() if self.regression_analysis else None,
            "final_status": self.final_status.value,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationReport:
        steps = [TestExecutionResult.from_dict(s) for s in data.get("step_results", [])]
        attempts = [CorrectionAttempt.from_dict(c) for c in data.get("correction_attempts", [])]
        reg = RegressionReport.from_dict(data["regression_analysis"]) if data.get("regression_analysis") else None
        return cls(
            task_description=data.get("task_description", ""),
            implementation_status=data.get("implementation_status", "UNKNOWN"),
            changed_files=data.get("changed_files", []),
            impact_summary=data.get("impact_summary", []),
            step_results=steps,
            correction_attempts=attempts,
            failure_analysis=data.get("failure_analysis", []),
            regression_analysis=reg,
            final_status=VerificationStatus(data.get("final_status", VerificationStatus.NOT_RUN.value)),
            total_duration_seconds=float(data.get("total_duration_seconds", 0.0)),
            timestamp=data.get("timestamp", ""),
        )
