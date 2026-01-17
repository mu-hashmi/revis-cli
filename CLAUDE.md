# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Constraints

**This is a public open-source tool** intended for use by ML engineers and research teams. When making changes:
- Never hard-code values specific to any particular ML repo or user environment
- All features must work universally across different ML projects
- Avoid workarounds that wouldn't scale as a general-purpose tool

## Build & Development Commands

```bash
uv run pytest                    # Run all tests
uv run pytest tests/test_X.py    # Run single test file
uv run pytest -k "test_name"     # Run tests matching pattern
uv run ruff check src            # Lint
uv run ruff format src           # Format
uv run revis <command>           # Run CLI
```

## Architecture Overview

Revis is an autonomous ML iteration engine. The core loop:
1. Run training script (local tmux or remote SSH)
2. Collect metrics from `eval.json`
3. LLM agent analyzes results and proposes code changes
4. Commit changes and repeat until budget exhausted or target achieved

### Core Subsystems

```
src/revis/
├── cli.py           # Typer CLI - all user-facing commands
├── loop.py          # RevisLoop - main orchestration
├── config.py        # Pydantic config models from revis.yaml
├── types.py         # Core types: Budget, Session, Run, Decision
│
├── llm/
│   ├── agent.py     # Tool-calling agentic loop (run_agent)
│   ├── tools.py     # Tool definitions & ToolExecutor
│   ├── client.py    # LLM API wrapper via litellm
│   └── prompts.py   # System prompts & context building
│
├── executor/
│   ├── local.py     # LocalExecutor - tmux on local machine
│   └── ssh.py       # SSHExecutor - paramiko + remote tmux
│
├── analyzer/
│   ├── compare.py   # RunAnalyzer - compares runs, calculates deltas
│   └── detectors.py # GuardrailChecker - NaN, divergence, plateau
│
├── evaluator/
│   └── harness.py   # Parses eval.json from training output
│
├── store/
│   ├── base.py      # RunStore protocol (interface for storage backends)
│   └── sqlite.py    # SQLiteRunStore implementation (default)
│
└── github/
    └── pr.py        # Git operations & GitHub PR creation
```

### Key Data Flow

- **Config**: `revis.yaml` → `config.py` → Pydantic models
- **Storage**: `RunStore` protocol in `store/base.py` defines the interface; SQLite implementation in `store/sqlite.py` (`.revis/revis.db`). Designed to support other backends (S3, GCS, etc.)
- **Logs**: `.revis/logs/{session_name}.log` for persistent output
- **Branches**: `revis/{session_name}` created from base SHA

### LLM Agent Tools

The agent in `llm/agent.py` uses these tools defined in `llm/tools.py`:
- `read_file`: Read files with optional line ranges
- `write_file`: Complete file content writes (never partial)
- `list_directory`: List files with optional recursion
- `search_codebase`: Regex search across codebase
- `find_definition`: Find function/class definitions
- `run_command`: Execute shell commands (limited to safe commands like linters/tests)
- `get_training_logs`: Read training output logs with filters (all/errors/metrics)

File operations respect deny patterns from config (e.g., `.git/**`, `**/__pycache__/**`, `revis.yaml`).

### Training Command Override

The `entry.train` command in `revis.yaml` is used for the baseline run. The LLM agent cannot modify `revis.yaml` directly. Instead, if the agent needs to change CLI arguments (e.g., learning rate passed via command line), it can specify `NEXT_COMMAND:` in its response to override the training command for subsequent iterations. This keeps `revis.yaml` as the user-controlled config while allowing the agent to experiment with CLI arguments.

### Session Lifecycle

```
init → loop (running) → [stop signal] → stopped
                     → [budget exhausted] → completed
                     → [target achieved] → completed
                     → [error] → error
```

Sessions can be resumed with `revis resume <name>`.

## Configuration (revis.yaml)

Key sections:
- `executor`: `type: local` or `type: ssh` with host/port
- `entry_point`: Command to run training (e.g., `python train.py`)
- `metrics.primary`: Which metric to optimize
- `metrics.minimize`: Whether lower is better
- `metrics.target`: Optional target value to stop early
- `guardrails`: plateau window, timeout, retry budget
- `context.deny_patterns`: Files agent cannot read

## CLI Commands

| Command | Purpose |
|---------|---------|
| `init` | Initialize `.revis/` and `revis.yaml` |
| `loop` | Start iteration loop (supports `--background`) |
| `resume` | Resume stopped session |
| `stop` | Graceful stop via signal file |
| `status` | Show progress (supports `--watch`) |
| `show` | Show session details (`--trace` for agent tool calls) |
| `logs` | View session logs |
| `watch` | Attach to tmux session |
| `export` | Push branch & create GitHub PR |
| `delete` | Delete session and optionally branch |

## Code Style

- Python 3.11+
- Ruff: line-length=100, rules E,F,I,N,W
- Pydantic for config validation
- Rich for terminal output
- Type hints on public APIs
