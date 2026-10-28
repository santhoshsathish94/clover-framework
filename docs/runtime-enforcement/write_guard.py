from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path, PurePosixPath

PROTECTED_NAMES = {"test", "tests", "spec", "specs"}
PROTECTED_SUFFIXES = (
    ".test.js",
    ".test.ts",
    ".test.tsx",
    ".spec.js",
    ".spec.ts",
    ".spec.tsx",
)


class VerificationPolicy:
    """Small runtime policy for protecting trusted verification artifacts.

    A trusted verification file may be read and executed but not modified or deleted.
    New verification files can be created only when explicit creation mode is enabled.
    """

    def __init__(self, *, protected_root: str | None = None, allow_new_tests: bool | None = None) -> None:
        self.root = Path(protected_root).resolve() if protected_root else None
        self.allow_new_tests = (
            self._env_flag("CLOVER_ALLOW_NEW_TESTS") if allow_new_tests is None else allow_new_tests
        )

    @staticmethod
    def _env_flag(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    def _resolve(self, path: str) -> Path:
        return Path(path).resolve(strict=False)

    def _relative(self, path: str) -> PurePosixPath:
        candidate = self._resolve(path)
        if self.root is None:
            return PurePosixPath(candidate.as_posix())
        try:
            return PurePosixPath(candidate.relative_to(self.root).as_posix())
        except ValueError:
            # Outside the root. Match on the whole path rather than reporting "not verification",
            # so a caller using is_verification_path on its own is not told a test file is safe.
            return PurePosixPath(candidate.as_posix())

    def is_outside_root(self, path: str) -> bool:
        """Return True when a resolved path escapes the protected root."""
        if self.root is None:
            return False
        try:
            self._resolve(path).relative_to(self.root)
        except ValueError:
            return True
        return False

    @staticmethod
    def _fold(value: str) -> str:
        # NFC first: on macOS a decomposed spelling is the same file but a different string.
        return unicodedata.normalize("NFC", value).lower()

    def is_verification_path(self, path: str) -> bool:
        # Case-folded: on Windows and macOS, TESTS/A.TEST.JS names the same file as tests/a.test.js.
        relative = self._relative(path)
        parts = {self._fold(part) for part in relative.parts}
        name = self._fold(relative.name)
        return bool(parts & PROTECTED_NAMES) or name.endswith(PROTECTED_SUFFIXES)

    def _identity(self, path: Path) -> tuple[int, int] | None:
        """Return a filesystem identity for an existing file, or None."""
        try:
            info = path.stat()
        except (OSError, ValueError):
            return None
        if info.st_ino == 0:  # Some filesystems do not report inodes; identity is unavailable.
            return None
        return (info.st_dev, info.st_ino)

    def shares_identity_with_verification(self, path: str) -> bool:
        """Return True when a path is another name for an existing verification artifact.

        resolve() follows symlinks, but a hard link is not a link to a file, it is the file under a
        second name. Comparing identities is what catches src/notatest.js pointing at tests/a.test.js.
        """
        if self.root is None:
            return False
        resolved = self._resolve(path)
        try:
            info = resolved.stat()
        except (OSError, ValueError):
            return False
        # A file with one name cannot be a second name for something else, and this is the case for
        # almost every write. Without it, every write costs a full walk of the protected root.
        if info.st_nlink < 2:
            return False
        target = self._identity(resolved)
        if target is None:
            return False
        for candidate in self.root.rglob("*"):
            try:
                if not candidate.is_file() or not self.is_verification_path(str(candidate)):
                    continue
            except OSError:  # Unreadable entry, or a symlinked directory cycle.
                continue
            if self._identity(candidate) == target:
                return True
        return False

    def is_protected(self, path: str) -> bool:
        """Return True when an existing verification artifact is protected."""
        target = self._resolve(path)
        if self.is_verification_path(path) and target.exists():
            return True
        return self.shares_identity_with_verification(path)

    def authorize_write(self, path: str) -> str:
        """Return the policy decision and raise on a denied write."""
        if self.is_outside_root(path):
            self.audit("write_denied", path, reason="outside_protected_root")
            raise PermissionError(
                f"Clover runtime policy rejected write outside the protected root: {path}"
            )

        verification = self.is_verification_path(path)
        exists = self._resolve(path).exists()

        if not verification and self.shares_identity_with_verification(path):
            self.audit("write_denied", path, reason="hard_link_to_trusted_verification")
            raise PermissionError(
                f"Clover runtime policy rejected write to another name for a trusted verification artifact: {path}"
            )

        if verification and exists:
            self.audit("write_denied", path, reason="trusted_verification_exists")
            raise PermissionError(
                f"Clover runtime policy rejected write to trusted verification artifact: {path}"
            )

        if verification and not self.allow_new_tests:
            self.audit("write_denied", path, reason="new_verification_creation_disabled")
            raise PermissionError(
                f"Clover runtime policy rejected new verification creation: {path}"
            )

        decision = "write_allowed_new_verification" if verification else "write_allowed"
        self.audit(decision, path)
        return decision

    def audit(self, event: str, path: str, *, reason: str | None = None) -> None:
        """Emit one structured security event without recording file contents."""
        record = {
            "event": event,
            "path": str(self._resolve(path)),
            "verification_path": self.is_verification_path(path),
            "reason": reason,
        }
        print(json.dumps(record, sort_keys=True), flush=True)


def is_protected(path: str, *, protected_root: str | None = None) -> bool:
    """Compatibility helper for the default trusted-verification policy."""
    return VerificationPolicy(protected_root=protected_root).is_protected(path)


def guarded_write(
    path: str,
    content: str,
    *,
    protected_root: str | None = None,
    allow_new_tests: bool | None = None,
) -> None:
    """Authorize a write before it reaches the filesystem."""
    policy = VerificationPolicy(
        protected_root=protected_root,
        allow_new_tests=allow_new_tests,
    )
    policy.authorize_write(path)

    target = policy._resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(content)


if __name__ == "__main__":
    policy = VerificationPolicy()
    for example in [
        "src/app.js",
        "tests/example.test.js",
        "src/account/spec.ts",
    ]:
        status = "protected" if policy.is_protected(example) else "not-protected"
        print(f"{example}: {status}")
