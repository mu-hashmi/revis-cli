"""Tool definitions and executor for agentic file editing."""

import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. For large files, use start_line/end_line to read specific sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to repo root",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-indexed, optional)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (inclusive, optional)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist. Always write the complete file content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to repo root",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a path. Returns file names with '/' suffix for directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to repo root. Use '.' for root.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true, list all files recursively (default: false)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": "Search for a pattern across all files in the repo. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (supports regex)",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Only search files matching this glob pattern (e.g., '*.py')",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_definition",
            "description": "Find where a function, class, or variable is defined in the codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the function, class, or variable to find",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the repo directory. Use for linting, type checking, or running tests. Only safe commands are allowed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30)",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


class ToolExecutor:
    """Execute tools for agentic file editing."""

    def __init__(self, repo_root: Path, deny_patterns: list[str]):
        self.repo_root = repo_root
        self.deny_patterns = deny_patterns
        self.files_modified: list[str] = []

    def is_denied(self, path: str) -> bool:
        """Check if path matches any deny pattern."""
        for pattern in self.deny_patterns:
            if fnmatch(path, pattern) or fnmatch(Path(path).name, pattern):
                return True
            # Handle ** patterns
            if "**" in pattern:
                # Convert glob ** to regex
                regex_pattern = pattern.replace("**", ".*").replace("*", "[^/]*")
                if re.match(regex_pattern, path):
                    return True
        return False

    def execute(self, tool_name: str, args: dict) -> str:
        """Execute a tool and return result as string."""
        method = getattr(self, f"tool_{tool_name}", None)
        if not method:
            return f"Unknown tool: {tool_name}"
        try:
            return method(**args)
        except Exception as e:
            return f"Error: {e}"

    def tool_read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        if self.is_denied(path):
            return f"Access denied: {path}"

        full_path = self.repo_root / path
        if not full_path.exists():
            return f"File not found: {path}"
        if not full_path.is_file():
            return f"Not a file: {path}"

        try:
            content = full_path.read_text()
        except UnicodeDecodeError:
            return f"Cannot read binary file: {path}"

        lines = content.splitlines()

        if start_line or end_line:
            start = (start_line or 1) - 1
            end = end_line or len(lines)
            lines = lines[start:end]
            numbered = [f"{i + start + 1}: {line}" for i, line in enumerate(lines)]
            return "\n".join(numbered)

        return content

    def tool_write_file(self, path: str, content: str) -> str:
        if self.is_denied(path):
            return f"Access denied: {path}"

        full_path = self.repo_root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

        if path not in self.files_modified:
            self.files_modified.append(path)

        return f"Successfully wrote {len(content)} bytes to {path}"

    def tool_list_directory(self, path: str, recursive: bool = False) -> str:
        full_path = self.repo_root / path
        if not full_path.exists():
            return f"Directory not found: {path}"
        if not full_path.is_dir():
            return f"Not a directory: {path}"

        results = []
        if recursive:
            for item in sorted(full_path.rglob("*")):
                try:
                    rel_path = str(item.relative_to(self.repo_root))
                except ValueError:
                    continue
                if self.is_denied(rel_path):
                    continue
                suffix = "/" if item.is_dir() else ""
                results.append(f"{rel_path}{suffix}")
        else:
            for item in sorted(full_path.iterdir()):
                try:
                    rel_path = str(item.relative_to(self.repo_root))
                except ValueError:
                    continue
                if self.is_denied(rel_path):
                    continue
                suffix = "/" if item.is_dir() else ""
                results.append(f"{item.name}{suffix}")

        return "\n".join(results[:500]) if results else "(empty)"

    def tool_search_codebase(
        self,
        pattern: str,
        file_pattern: str | None = None,
    ) -> str:
        results = []
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        for filepath in self.repo_root.rglob("*"):
            if not filepath.is_file():
                continue

            try:
                rel_path = str(filepath.relative_to(self.repo_root))
            except ValueError:
                continue

            if self.is_denied(rel_path):
                continue

            if file_pattern and not fnmatch(filepath.name, file_pattern):
                continue

            try:
                content = filepath.read_text()
            except (UnicodeDecodeError, PermissionError):
                continue

            for i, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    results.append(f"{rel_path}:{i}: {line.strip()}")

        return "\n".join(results[:50]) if results else "No matches found"

    def tool_find_definition(self, name: str) -> str:
        patterns = [
            rf"^class\s+{re.escape(name)}\b",
            rf"^def\s+{re.escape(name)}\b",
            rf"^{re.escape(name)}\s*=",
            rf"^\s+def\s+{re.escape(name)}\b",
        ]
        combined = "|".join(f"({p})" for p in patterns)
        return self.tool_search_codebase(combined, "*.py")

    def tool_run_command(self, command: str, timeout: int = 30) -> str:
        allowed_prefixes = [
            "python -m py_compile",
            "python3 -m py_compile",
            "ruff check",
            "ruff format --check",
            "black --check",
            "pytest -x",
            "mypy",
        ]
        if not any(command.startswith(p) for p in allowed_prefixes):
            return f"Command not allowed. Allowed prefixes: {allowed_prefixes}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            return output.strip() if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
