from __future__ import annotations

from pathlib import Path

from .config import load_settings


def build_context(relevant_files: list[str], continuation_context: str = "") -> str:
    parts: list[str] = []
    if continuation_context:
        parts.append("=== CONTINUATION CONTEXT ===\n" + continuation_context)

    file_context = read_relevant_files(relevant_files)
    if file_context:
        parts.append("=== RELEVANT FILES ===\n" + file_context)

    return "\n\n".join(parts)


def read_relevant_files(paths: list[str]) -> str:
    if not paths:
        return ""

    max_chars = load_settings().max_context_chars
    chunks: list[str] = []
    used = 0
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            chunks.append(f"--- {path} ---\n[missing]")
            continue
        if path.is_dir():
            chunks.append(f"--- {path} ---\n[directory]")
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            chunks.append(f"--- {path} ---\n[unreadable: {exc}]")
            continue
        remaining = max_chars - used
        if remaining <= 0:
            chunks.append("[context truncated]")
            break
        if len(text) > remaining:
            text = text[:remaining] + "\n[truncated]"
        used += len(text)
        chunks.append(f"--- {path} ---\n{text}")
    return "\n\n".join(chunks)
