from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from .config import load_settings

DANGEROUS_SYSTEM_ROOTS = (
    Path("/"),
    Path("/etc"),
    Path("/usr"),
    Path("/bin"),
    Path("/var"),
    Path("/root"),
    Path("C:/Windows"),
    Path("C:/Program Files"),
)

EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".mypy_cache",
    ".next",
    ".nox",
    ".nuxt",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "build",
    "coverage",
    "dist",
    "env",
    "htmlcov",
    "node_modules",
    "out",
    "target",
    "venv",
}

SENSITIVE_PATH_PATTERNS = (
    ".env",
    ".env.*",
    ".ssh/*",
    ".aws/*",
    ".config/gcloud/*",
    ".kube/*",
    "credentials",
    "*credentials*",
    "id_rsa",
    "id_ed25519",
    "*.pem",
    "*.key",
)

SENSITIVE_ROOT_DIRS = {
    ".aws",
    ".gnupg",
    ".kube",
    ".ssh",
}

# Directory names that must never appear as a component of a requested path,
# at any depth, regardless of where the workspace root is anchored.
SENSITIVE_DIR_COMPONENTS = {
    ".aws",
    ".gnupg",
    ".kube",
    ".ssh",
}

# Consecutive directory components that together identify a sensitive location.
SENSITIVE_DIR_SEQUENCES = ((".config", "gcloud"),)

MAX_FILE_BYTES = 1_000_000

# Bounds on directory expansion so a huge workspace cannot burn CPU/memory
# building a file list that would be discarded by the context-size limit.
MAX_WALK_FILES = 2_000
MAX_WALK_ENTRIES = 50_000


def build_context(
    relevant_files: list[str],
    continuation_context: str = "",
    workspace_root: str | None = None,
) -> str:
    parts: list[str] = []
    if continuation_context:
        parts.append("=== CONTINUATION CONTEXT ===\n" + continuation_context)

    file_context = read_relevant_files(relevant_files, workspace_root)
    if file_context:
        parts.append("=== RELEVANT FILES ===\n" + file_context)

    return "\n\n".join(parts)


def read_relevant_files(paths: list[str], workspace_root: str | None = None) -> str:
    if not paths:
        return ""
    if not workspace_root:
        return "[blocked: workspace_root is required when relevant_files are provided]"

    try:
        root = _resolve_workspace_root(workspace_root)
    except ValueError as exc:
        return f"[blocked: invalid workspace root: {exc}]"

    max_chars = load_settings().max_context_chars
    chunks: list[str] = []
    used = 0
    truncation_marker = "\n[truncated]"

    for path, marker in _expand_relevant_paths(paths, root):
        remaining = max_chars - used
        if remaining <= 0:
            chunks.append("[context truncated]")
            break

        chunk = marker if marker is not None else _read_file(path, root)
        if len(chunk) > remaining:
            if remaining <= len(truncation_marker):
                chunks.append("[context truncated]")
            else:
                chunks.append(chunk[: remaining - len(truncation_marker)] + truncation_marker)
            break
        used += len(chunk)
        chunks.append(chunk)
    return "\n\n".join(chunks)


def _resolve_workspace_root(raw_root: str) -> Path:
    root = Path(raw_root).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"{root} does not exist")
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")
    if _is_dangerous_system_root(root):
        raise ValueError(f"{root} is a protected system path")
    if _is_home_root(root):
        raise ValueError(f"{root} is a home directory root; choose a project subdirectory")
    if _is_sensitive_workspace_root(root):
        raise ValueError(f"{root} is a sensitive directory; choose a project subdirectory")
    return root


def _expand_relevant_paths(paths: list[str], root: Path) -> list[tuple[Path, str | None]]:
    expanded: list[tuple[Path, str | None]] = []
    seen: set[Path] = set()
    budget = MAX_WALK_FILES
    for raw_path in paths:
        path, marker = _resolve_requested_path(raw_path, root)
        if marker:
            expanded.append((path, marker))
            continue
        if path in seen:
            continue
        if path.is_file():
            expanded.append((path, None))
            seen.add(path)
            continue
        if path.is_dir():
            walked, truncated = _walk_directory(path, root, budget)
            budget -= len(walked)
            for file_path in walked:
                if file_path not in seen:
                    expanded.append((file_path, None))
                    seen.add(file_path)
            if truncated:
                expanded.append(
                    (
                        path,
                        f"--- {raw_path} ---\n"
                        f"[truncated: directory expansion limit of "
                        f"{MAX_WALK_FILES} files reached]",
                    )
                )
                break
    return sorted(expanded, key=lambda item: str(item[0]))


def _resolve_requested_path(raw_path: str, root: Path) -> tuple[Path, str | None]:
    requested = Path(raw_path).expanduser()
    candidate = requested if requested.is_absolute() else root / requested
    try:
        path = candidate.resolve()
    except OSError as exc:
        return candidate, f"--- {raw_path} ---\n[unreadable: {exc}]"

    if not _is_inside(path, root):
        return path, f"--- {raw_path} ---\n[blocked: outside workspace root]"
    if _is_sensitive_path(path, root):
        return path, f"--- {raw_path} ---\n[blocked: sensitive file]"
    if not path.exists():
        return path, f"--- {raw_path} ---\n[missing]"
    if not path.is_file() and not path.is_dir():
        return path, f"--- {raw_path} ---\n[not a file or directory]"
    if _has_excluded_component(path, root):
        return path, f"--- {raw_path} ---\n[blocked: excluded directory]"
    return path, None


def _walk_directory(path: Path, workspace_root: Path, limit: int) -> tuple[list[Path], bool]:
    """Collect files under ``path``, stopping once ``limit`` files are gathered.

    Returns the files found and whether the walk was cut short by a budget.
    """
    files: list[Path] = []
    if limit <= 0:
        return files, True
    scanned = 0
    for root, dirnames, filenames in os.walk(path):
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if not dirname.startswith(".") and dirname not in EXCLUDED_DIRS
        )
        for filename in sorted(filenames):
            scanned += 1
            if scanned > MAX_WALK_ENTRIES:
                return files, True
            if filename.startswith("."):
                continue
            file_path = (Path(root) / filename).resolve()
            if not _is_inside(file_path, workspace_root):
                continue
            if _is_sensitive_path(file_path, workspace_root):
                continue
            files.append(file_path)
            if len(files) >= limit:
                return files, True
    return files, False


def _read_file(path: Path, root: Path) -> str:
    try:
        stat_result = path.stat()
    except OSError as exc:
        return f"--- {path} ---\n[unreadable: {exc}]"
    if stat_result.st_size > MAX_FILE_BYTES:
        return (
            f"--- {path.relative_to(root)} ---\n"
            f"[file too large: {stat_result.st_size} bytes, max: {MAX_FILE_BYTES}]"
        )
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"--- {path.relative_to(root)} ---\n[unreadable: {exc}]"
    return f"--- {path.relative_to(root)} ---\n{text}"


def _is_inside(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _is_dangerous_system_root(path: Path) -> bool:
    if path.parent == path:
        return True
    for dangerous_root in DANGEROUS_SYSTEM_ROOTS:
        if str(dangerous_root) in {"/", "."}:
            continue
        try:
            resolved_dangerous = dangerous_root.resolve()
        except OSError:
            resolved_dangerous = dangerous_root
        if str(dangerous_root) in {"/var", "var"}:
            if path == resolved_dangerous:
                return True
            continue
        if path == resolved_dangerous or path.is_relative_to(resolved_dangerous):
            return True
    return False


def _is_home_root(path: Path) -> bool:
    try:
        home = Path.home().resolve()
    except OSError:
        return False
    return path == home


def _is_sensitive_workspace_root(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    if set(lowered_parts) & SENSITIVE_ROOT_DIRS:
        return True
    return _contains_sensitive_sequence(lowered_parts)


def _contains_sensitive_sequence(lowered_parts: list[str]) -> bool:
    for sequence in SENSITIVE_DIR_SEQUENCES:
        width = len(sequence)
        for index in range(len(lowered_parts) - width + 1):
            if tuple(lowered_parts[index : index + width]) == sequence:
                return True
    return False


def _has_excluded_component(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    parts = relative.parts if path.is_dir() else relative.parts[:-1]
    return any(part in EXCLUDED_DIRS for part in parts)


def _is_sensitive_path(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    lowered_parts = [part.lower() for part in relative.parts]
    if SENSITIVE_DIR_COMPONENTS.intersection(lowered_parts):
        return True
    if _contains_sensitive_sequence(lowered_parts):
        return True
    relative_posix = relative.as_posix().lower()
    name = path.name.lower()
    for pattern in SENSITIVE_PATH_PATTERNS:
        if fnmatch.fnmatch(relative_posix, pattern) or fnmatch.fnmatch(name, pattern):
            return True
    return False
