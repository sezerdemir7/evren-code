"""Change Validation and Security Auditor (`validator.py`).

Prevents autonomous remediation and code changes from weakening tests, deleting assertions,
injecting trivial assertions, or escaping workspace boundaries using AST and pattern rules.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple


@dataclass
class ValidationOutcome:
    """Code change validation outcome."""

    is_valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ChangeValidator:
    """Validates code modifications against safety rules and test preservation policies."""

    # Test weakening and skipping patterns (Python, Java, Jest, Vitest, Mocha)
    TEST_WEAKENING_PATTERNS = [
        re.compile(r"^\+.*@pytest\.mark\.skip", re.MULTILINE),
        re.compile(r"^\+.*@pytest\.mark\.skipif", re.MULTILINE),
        re.compile(r"^\+.*@pytest\.mark\.xfail", re.MULTILINE),
        re.compile(r"^\+.*@Disabled", re.MULTILINE),
        re.compile(r"^\+.*@Ignore", re.MULTILINE),
        re.compile(r"^\+.*\b(?:describe|it|test|context)\.skip\(", re.MULTILINE),
        re.compile(r"^\+.*\b(?:xit|xtest|xdescribe)\(", re.MULTILINE),
        re.compile(r"^\+.*\btest\.todo\(", re.MULTILINE),
        re.compile(r"^\+.*\.skip\(", re.MULTILINE),
        re.compile(r"^\+.*unittest\.skip", re.MULTILINE),
    ]

    # Test deletion or assert removal patterns
    TEST_DELETION_PATTERNS = [
        re.compile(r"^-.*def\s+test_", re.MULTILINE),
        re.compile(r"^-.*@(?:Test|ParameterizedTest|RepeatedTest)\b", re.MULTILINE),
        re.compile(r"^-.*(?:public\s+)?void\s+test", re.MULTILINE),
        re.compile(r"^-.*\b(?:describe|it|test)\s*\(", re.MULTILINE),
        re.compile(r"^-.*assert\s+", re.MULTILINE),
        re.compile(r"^-.*assertEquals|assertTrue|assertFalse", re.MULTILINE),
        re.compile(r"^-.*expect\(.*(?:\.to|\.not)", re.MULTILINE),
    ]

    # Trivial / Fake assertions
    TRIVIAL_ASSERT_PATTERNS = [
        re.compile(r"^\+.*assert\s+True\b", re.MULTILINE),
        re.compile(r"^\+.*assert\s+1\s*==\s*1\b", re.MULTILINE),
        re.compile(r"^\+.*assert\s+1\b", re.MULTILINE),
        re.compile(r"^\+.*assertTrue\s*\(\s*(?:true|True|1)\s*\)", re.MULTILINE),
        re.compile(r"^\+.*assertEquals\s*\(\s*1\s*,\s*1\s*\)", re.MULTILINE),
        re.compile(r"^\+.*expect\s*\(\s*true\s*\)\s*\.toBe\s*\(\s*true\s*\)", re.MULTILINE),
    ]

    # Test config narrowing patterns
    CONFIG_NARROWING_PATTERNS = [
        re.compile(r"^\+.*testPathIgnorePatterns", re.MULTILINE),
        re.compile(r"^\+.*<exclude>.*(?:Test|\*).*\.java</exclude>", re.MULTILINE),
        re.compile(r"^\+.*ignoreFailures\s*=\s*true", re.MULTILINE),
        re.compile(r"^\+.*addopts\s*=.*(?:-k\s*[\"']not|-m\s*[\"']not)", re.MULTILINE),
    ]

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    def validate_patch(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        diff_text: Optional[str] = None,
    ) -> ValidationOutcome:
        """Validates patch against test weakening and containment violations."""
        violations: List[str] = []
        warnings: List[str] = []

        # 1. Path Containment Check
        try:
            target_path = (self.workspace_root / file_path).resolve()
            if not target_path.is_relative_to(self.workspace_root):
                violations.append(f"Workspace boundary violation / sınır ihlali: '{file_path}' is outside workspace.")
        except Exception as e:
            violations.append(f"Invalid path / Geçersiz dosya yolu '{file_path}': {e}")

        patch_text = diff_text or self._generate_simple_diff(old_content, new_content)

        # 2. Config narrowing check
        if self._is_test_config_file(file_path):
            for pat in self.CONFIG_NARROWING_PATTERNS:
                if pat.search(patch_text):
                    violations.append(
                        f"Test configuration narrowing detected (daraltma tespit edildi) in '{file_path}': Cannot add rules skipping test scope."
                    )

        # 3. Test weakening & deletion checks
        is_test = self._is_test_file(file_path)

        for pat in self.TEST_WEAKENING_PATTERNS:
            if pat.search(patch_text):
                violations.append(
                    f"Test weakening detected (zayıflatma): Cannot add skip/disabled decorators in '{file_path}'."
                )

        # Trivial assertion check
        for pat in self.TRIVIAL_ASSERT_PATTERNS:
            if pat.search(patch_text):
                violations.append(
                    f"Test weakening detected (zayıflatma): Added fake or trivial assertion / önemsiz iddia ('{file_path}')."
                )

        if is_test:
            if not new_content.strip() and old_content.strip():
                violations.append(
                    f"Test file deletion violation: '{file_path}' cannot be completely deleted or emptied (tamamen silinemez veya boşaltılamaz)."
                )

            for pat in self.TEST_DELETION_PATTERNS:
                matches = pat.findall(patch_text)
                if matches:
                    violations.append(
                        f"Assertion or test deletion detected (silme tespit edildi / zayıflatma) ({len(matches)} lines). Deleting tests to pass is prohibited."
                    )

            if file_path.endswith(".py"):
                ast_violations = self._validate_python_test_ast(old_content, new_content)
                violations.extend(ast_violations)

        lines_changed = len([l for l in patch_text.splitlines() if l.startswith("+") or l.startswith("-")])
        if lines_changed > 150:
            warnings.append(f"Large patch detected ({lines_changed} lines). Follow minimal change principle.")

        is_valid = len(violations) == 0
        return ValidationOutcome(is_valid=is_valid, violations=violations, warnings=warnings)

    def _validate_python_test_ast(self, old_code: str, new_code: str) -> List[str]:
        """Performs structural AST test integrity analysis for Python test files."""
        violations: List[str] = []
        try:
            old_tree = ast.parse(old_code)
            new_tree = ast.parse(new_code)
        except SyntaxError:
            return violations

        # 1. Assert count check
        old_asserts = sum(1 for n in ast.walk(old_tree) if isinstance(n, ast.Assert))
        new_asserts = sum(1 for n in ast.walk(new_tree) if isinstance(n, ast.Assert))
        if new_asserts < old_asserts:
            violations.append(
                f"Test integrity violation / Test bütünlüğü ihlali: Reduced number of assertions in test file (assert sayısı azaltıldı) ({old_asserts} -> {new_asserts})."
            )

        # 2. Test function count check
        old_tests = sum(
            1 for n in ast.walk(old_tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")
        )
        new_tests = sum(
            1 for n in ast.walk(new_tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")
        )
        if new_tests < old_tests:
            violations.append(
                f"Test deletion violation / Test silme ihlali: Reduced number of test functions (test fonksiyonu sayısı azaltıldı) ({old_tests} -> {new_tests})."
            )

        # 3. Trivial Assert AST Check (assert True, assert 1, etc.)
        for n in ast.walk(new_tree):
            if isinstance(n, ast.Assert):
                if isinstance(n.test, ast.Constant) and bool(n.test.value) is True:
                    violations.append(
                        "Test weakening violation (zayıflatma): Constant 'assert True / assert 1' / önemsiz iddia detected in code."
                    )
                elif isinstance(n.test, ast.Compare):
                    if (
                        isinstance(n.test.left, ast.Constant)
                        and len(n.test.comparators) == 1
                        and isinstance(n.test.comparators[0], ast.Constant)
                        and n.test.left.value == n.test.comparators[0].value
                    ):
                        violations.append(
                            "Test weakening violation (zayıflatma): Constant equality assertion ('assert 1 == 1' / önemsiz iddia) detected."
                        )

        for n in ast.walk(new_tree):
            if isinstance(n, ast.Try):
                for h in n.handlers:
                    is_catch_all = False
                    if h.type is None:
                        is_catch_all = True
                    elif isinstance(h.type, ast.Name) and h.type.id in ("AssertionError", "Exception", "BaseException"):
                        is_catch_all = True

                    if is_catch_all:
                        body_is_empty = all(isinstance(s, (ast.Pass, ast.Expr)) for s in h.body)
                        has_raise = any(isinstance(s, ast.Raise) for s in h.body)
                        if body_is_empty and not has_raise:
                            violations.append(
                                "Test atlatma ihlali: AssertionError/Exception yutan boş 'except: pass' tespit edildi."
                            )

        return violations

    def _generate_simple_diff(self, old: str, new: str) -> str:
        """Generate simple diff."""
        import difflib
        return "\n".join(difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        ))

    def _is_test_config_file(self, file_path: str) -> bool:
        """Is test config file."""
        name = Path(file_path).name.lower()
        if name in ("pytest.ini", "setup.cfg", "pom.xml", "build.gradle", "build.gradle.kts"):
            return True
        if name.startswith("jest.config.") or name in (".mocharc.json", ".mocharc.js"):
            return True
        return False

    def _is_test_file(self, file_path: str) -> bool:
        """Is test file."""
        p_low = file_path.lower().replace("\\", "/")
        name = Path(file_path).name.lower()

        if any(seg in p_low.split("/") for seg in ("test", "tests", "__tests__", "spec", "specs")):
            return True

        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        if name.endswith("test.java") or name.endswith("tests.java") or name.endswith("testcase.java"):
            return True
        if name.endswith(".test.ts") or name.endswith(".spec.ts"):
            return True
        if name.endswith(".test.js") or name.endswith(".spec.js"):
            return True
        if name.endswith(".test.tsx") or name.endswith(".spec.tsx"):
            return True
        if name.endswith(".test.jsx") or name.endswith(".spec.jsx"):
            return True
        if name.endswith("_test.go") or name.endswith("_test.rs"):
            return True
        return False
