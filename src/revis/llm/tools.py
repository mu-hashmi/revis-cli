"""Tool definitions and executor for config-only changes."""

from __future__ import annotations

import json
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from revis.executor.base import Executor

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
            "name": "get_training_logs",
            "description": "Get training logs from the current run. Use to understand training dynamics, debug errors, or see loss progression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "enum": ["all", "errors", "metrics"],
                        "description": "Filter: 'all' = last 200 lines, 'errors' = ERROR/WARNING lines, 'metrics' = loss/accuracy lines.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_config",
            "description": "Modify a value in a config file (YAML, JSON, or TOML). Only modifies existing keys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Config file path relative to repo root",
                    },
                    "key": {
                        "type": "string",
                        "description": "Dot-separated key path (e.g., 'training.learning_rate' or 'optimizer.params.lr')",
                    },
                    "value": {
                        "type": "string",
                        "description": "New value (will be parsed as appropriate type: number, bool, string)",
                    },
                },
                "required": ["path", "key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_next_command",
            "description": "Set the CLI command for the next training run. Use to change hyperparameters passed via CLI args.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Full training command (e.g., 'python train.py --lr 1e-5 --epochs 100')",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_code_change",
            "description": "Request a code change that cannot be done via config modification. This will hand off to a coding agent (Claude Code, Aider, etc.) or pause for manual intervention.",
            "parameters": {
                "type": "object",
                "properties": {
                    "suggestion": {
                        "type": "string",
                        "description": "Detailed description of the code change needed",
                    },
                    "hypothesis": {
                        "type": "string",
                        "description": "Why this change should improve metrics",
                    },
                    "relevant_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of files that likely need modification",
                    },
                },
                "required": ["suggestion", "hypothesis", "relevant_files"],
            },
        },
    },
]


class ToolExecutor:
    """Execute tools for config-only changes."""

    def __init__(
        self,
        repo_root: Path,
        deny_patterns: list[str],
        executor: Executor | None = None,
        run_output_dir: str | None = None,
    ):
        self.repo_root = repo_root
        self.deny_patterns = deny_patterns
        self._executor = executor
        self._run_output_dir = run_output_dir
        self.config_changes: list[dict] = []
        self.next_command: str | None = None
        self.code_change_request: dict | None = None

    def is_denied(self, path: str) -> bool:
        """Check if path matches any deny pattern."""
        for pattern in self.deny_patterns:
            if fnmatch(path, pattern) or fnmatch(Path(path).name, pattern):
                return True
            if "**" in pattern:
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

    def tool_get_training_logs(self, filter: str = "all") -> str:
        """Get training logs with optional filtering."""
        if self._executor is None or self._run_output_dir is None:
            return "Training logs not available (no active run)"

        log_path = f"{self._run_output_dir}/train.log"
        log_content = self._executor.get_log_tail(log_path, lines=500)

        if not log_content.strip():
            return "(no training logs found)"

        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        log_content = ansi_escape.sub('', log_content)

        lines = log_content.strip().split("\n")

        if filter == "errors":
            patterns = ["error", "warning", "exception", "traceback", "failed", "oom", "nan", "cuda"]
            lines = [line for line in lines if any(p in line.lower() for p in patterns)]
        elif filter == "metrics":
            patterns = ["loss", "accuracy", "acc", "epoch", "step", "lr=", "learning_rate", "val_", "train_"]
            lines = [line for line in lines if any(p in line.lower() for p in patterns)]
            if len(lines) > 50:
                step = len(lines) // 50
                lines = lines[::step]

        if len(lines) > 200:
            lines = lines[-200:]

        result = "\n".join(lines)
        if len(result) > 30000:
            result = result[-30000:]

        return result if result.strip() else "(no matching log lines)"

    def tool_modify_config(self, path: str, key: str, value: str) -> str:
        """Modify a config file value."""
        if self.is_denied(path):
            return f"Access denied: {path}"

        full_path = self.repo_root / path
        if not full_path.exists():
            return f"Config file not found: {path}"

        suffix = full_path.suffix.lower()
        if suffix not in (".yaml", ".yml", ".json", ".toml"):
            return f"Unsupported config format: {suffix}"

        try:
            content = full_path.read_text()

            if suffix in (".yaml", ".yml"):
                data = yaml.safe_load(content)
            elif suffix == ".json":
                data = json.loads(content)
            elif suffix == ".toml":
                try:
                    import tomllib
                    data = tomllib.loads(content)
                except ImportError:
                    return "TOML support requires Python 3.11+"
            else:
                return f"Unsupported format: {suffix}"

            keys = key.split(".")
            current = data
            for k in keys[:-1]:
                if k not in current:
                    return f"Key not found: {key}"
                current = current[k]

            final_key = keys[-1]
            if final_key not in current:
                return f"Key not found: {key}"

            old_value = current[final_key]
            new_value = self._parse_value(value, type(old_value))
            current[final_key] = new_value

            self.config_changes.append({
                "path": path,
                "key": key,
                "old_value": old_value,
                "new_value": new_value,
            })

            if suffix in (".yaml", ".yml"):
                new_content = yaml.dump(data, default_flow_style=False, sort_keys=False)
            elif suffix == ".json":
                new_content = json.dumps(data, indent=2)
            else:
                return "Writing TOML not supported"

            full_path.write_text(new_content)
            return f"Modified {path}: {key} = {old_value} → {new_value}"

        except Exception as e:
            return f"Error modifying config: {e}"

    def _parse_value(self, value: str, target_type: type):
        """Parse string value to target type."""
        if target_type is bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif target_type is int:
            if "e" in value.lower() or "." in value:
                return int(float(value))
            return int(value)
        elif target_type is float:
            return float(value)
        elif target_type is list:
            return json.loads(value)
        elif target_type is dict:
            return json.loads(value)
        else:
            return value

    def tool_set_next_command(self, command: str) -> str:
        """Set the CLI command for the next training run."""
        self.next_command = command
        return f"Next training command set to: {command}"

    def tool_request_code_change(
        self,
        suggestion: str,
        hypothesis: str,
        relevant_files: list[str],
    ) -> str:
        """Request a code change to be handled by a coding agent."""
        self.code_change_request = {
            "suggestion": suggestion,
            "hypothesis": hypothesis,
            "relevant_files": relevant_files,
        }
        files_str = ", ".join(relevant_files)
        return (
            f"Code change requested. This will be handed off to a coding agent.\n\n"
            f"Suggestion: {suggestion}\nHypothesis: {hypothesis}\nFiles: {files_str}"
        )
