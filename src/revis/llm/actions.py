"""Action parsing and application."""

import fnmatch
import re
from pathlib import Path

from revis.config import ActionBoundsConfig
from revis.types import Action, FileEdit


class ActionParseError(Exception):
    """Error parsing LLM action output."""

    pass


class ActionValidationError(Exception):
    """Error validating action against bounds."""

    pass


class ActionApplyError(Exception):
    """Error applying action to files."""

    pass


def parse_action(response: str) -> Action:
    """
    Parse LLM response into an Action.

    Expected format:
    <edit>
    <path>file/path</path>
    <search>text to find</search>  <!-- optional -->
    <replace>replacement text</replace>
    </edit>

    <rationale>explanation</rationale>
    <significant/>  <!-- optional -->
    <escalate/>  <!-- if cannot proceed -->
    """
    # Check for escalation
    if "<escalate/>" in response or "<escalate>" in response:
        rationale = _extract_tag(response, "rationale") or "LLM requested escalation"
        return Action(type="escalate", edits=[], rationale=rationale)

    # Parse edit blocks
    edits = _parse_edit_blocks(response)

    # Parse rationale
    rationale = _extract_tag(response, "rationale") or "No rationale provided"

    # Check for significant marker
    significant = "<significant/>" in response or "<significant>" in response

    return Action(
        type="code_patch",
        edits=edits,
        rationale=rationale,
        significant=significant,
    )


def _parse_edit_blocks(response: str) -> list[FileEdit]:
    """Parse all <edit> blocks from response."""
    edits = []

    # Find all edit blocks
    edit_pattern = r"<edit>(.*?)</edit>"
    matches = re.findall(edit_pattern, response, re.DOTALL)

    for block in matches:
        path = _extract_tag(block, "path")
        if not path:
            raise ActionParseError("Edit block missing <path> tag")

        search = _extract_tag(block, "search")  # Optional
        replace = _extract_tag(block, "replace")

        if replace is None:
            raise ActionParseError(f"Edit block for {path} missing <replace> tag")

        edits.append(FileEdit(
            path=path.strip(),
            search=search.strip() if search else None,
            replace=replace,
        ))

    return edits


def _extract_tag(text: str, tag: str) -> str | None:
    """Extract content between XML-style tags."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


def validate_action(action: Action, bounds: ActionBoundsConfig) -> tuple[bool, str | None]:
    """
    Validate action against bounds.

    Returns (valid, error_message).
    Deny patterns win over allow patterns on conflict.
    """
    if action.type == "escalate":
        return True, None

    for edit in action.edits:
        # Check deny patterns first (deny wins)
        for pattern in bounds.deny:
            if fnmatch.fnmatch(edit.path, pattern):
                return False, f"File '{edit.path}' matches denied pattern '{pattern}'"

        # Check allow patterns if specified
        if bounds.allow:
            allowed = any(fnmatch.fnmatch(edit.path, p) for p in bounds.allow)
            if not allowed:
                return False, f"File '{edit.path}' not in allowed patterns"

    return True, None


def apply_action(action: Action, repo_root: Path) -> tuple[bool, str | None]:
    """
    Apply action to files atomically.

    All edits must succeed or none are applied.
    Returns (success, error_message).
    """
    if action.type == "escalate":
        return True, None

    if not action.edits:
        return True, None

    # First pass: validate all edits can be applied
    pending_changes: list[tuple[Path, str]] = []

    for edit in action.edits:
        file_path = repo_root / edit.path

        if edit.search is None:
            # Full file replacement or new file
            new_content = edit.replace
        else:
            # Search and replace
            if not file_path.exists():
                return False, f"File not found: {edit.path}"

            content = file_path.read_text()
            normalized_search = _normalize_whitespace(edit.search)
            normalized_content = _normalize_whitespace(content)

            # Find the match position in normalized content
            match_pos = normalized_content.find(normalized_search)
            if match_pos == -1:
                # Try to find a close match for better error message
                suggestion = _find_similar(content, edit.search)
                error = f"Search text not found in {edit.path}"
                if suggestion:
                    error += f"\n\nDid you mean:\n{suggestion[:200]}..."
                return False, error

            # Find the actual position in original content
            actual_pos = _find_actual_position(content, edit.search)
            if actual_pos == -1:
                return False, f"Could not locate search text in {edit.path}"

            # Calculate end position
            end_pos = actual_pos + len(edit.search)

            # Apply replacement
            new_content = content[:actual_pos] + edit.replace + content[end_pos:]

        pending_changes.append((file_path, new_content))

    # Second pass: apply all changes
    for file_path, new_content in pending_changes:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(new_content)

    return True, None


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for matching."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    # Strip leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _find_actual_position(content: str, search: str) -> int:
    """Find actual position of search text, handling whitespace differences."""
    # Try exact match first
    pos = content.find(search)
    if pos != -1:
        return pos

    # Try with normalized whitespace
    normalized_search = _normalize_whitespace(search)
    normalized_content = _normalize_whitespace(content)

    norm_pos = normalized_content.find(normalized_search)
    if norm_pos == -1:
        return -1

    # Map normalized position back to original
    # Count characters in original that correspond to normalized position
    orig_pos = 0
    norm_count = 0
    content_clean = content.replace("\r\n", "\n").replace("\r", "\n")

    i = 0
    while i < len(content_clean) and norm_count < norm_pos:
        # Skip leading whitespace in original that was stripped
        if norm_count == 0:
            while i < len(content_clean) and content_clean[i] in " \t\n":
                i += 1
                orig_pos += 1

        if i >= len(content_clean):
            break

        orig_pos = i
        norm_count += 1
        i += 1

        # Handle trailing whitespace on lines
        if i < len(content_clean) and content_clean[i - 1] != "\n":
            while i < len(content_clean) and content_clean[i] in " \t" and content_clean[i] != "\n":
                i += 1

    return orig_pos


def _find_similar(content: str, search: str, max_len: int = 500) -> str | None:
    """Try to find similar text for error message."""
    search_lines = search.strip().split("\n")
    if not search_lines:
        return None

    # Try to find the first line
    first_line = search_lines[0].strip()
    if not first_line:
        return None

    for i, line in enumerate(content.split("\n")):
        if first_line in line or line.strip() == first_line:
            # Found potential match, extract context
            lines = content.split("\n")
            start = max(0, i)
            end = min(len(lines), i + len(search_lines) + 2)
            return "\n".join(lines[start:end])

    return None


def format_action_for_commit(action: Action) -> str:
    """Format action for commit message."""
    if action.type == "escalate":
        return "Escalated to human review"

    files = [e.path for e in action.edits]
    files_str = ", ".join(files[:3])
    if len(files) > 3:
        files_str += f" (+{len(files) - 3} more)"

    return f"Revis: {action.rationale}\n\nFiles: {files_str}"
