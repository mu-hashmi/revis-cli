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
2. Collect metrics (from W&B API or `eval.json`)
3. LLM agent analyzes results and proposes changes (config or code handoff)
4. Commit changes and repeat until budget exhausted or target achieved

### Core Subsystems

```
src/revis/
├── cli.py           # Typer CLI - all user-facing commands
├── loop.py          # RevisLoop - main orchestration
├── config.py        # Pydantic config models from revis.yaml
├── types.py         # Core types: Budget, Session, Run, Decision, Suggestion
│
├── init/            # Interactive init flow
│   ├── prompts.py   # InquirerPy interactive prompts
│   ├── ssh_config.py # Parse ~/.ssh/config for host detection
│   └── metrics/     # Metrics source detection for init
│       ├── base.py      # MetricsSource protocol
│       ├── wandb.py     # W&B project/metric detection
│       └── eval_json.py # Fallback source
│
├── metrics/         # Metrics collection during loop
│   ├── base.py      # MetricsCollector protocol
│   ├── wandb.py     # W&B API polling
│   └── eval_json.py # Read eval.json from training output
│
├── agents/          # Coding agent handoff
│   ├── base.py      # CodingAgent protocol, HandoffContext
│   ├── claude_code.py # Claude Code CLI integration
│   └── detect.py    # Auto-detect available agents
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
├── store/
│   ├── base.py      # RunStore protocol
│   └── sqlite.py    # SQLiteRunStore with migrations
│
└── github/
    └── pr.py        # Git operations & GitHub PR creation
```

### Key Data Flow

- **Config**: `revis.yaml` → `config.py` → Pydantic models
- **Storage**: SQLite in `.revis/revis.db` with runs, suggestions, traces tables
- **Logs**: `.revis/logs/{session_name}.log` for persistent output
- **Branches**: `revis/{session_name}` created from base SHA

### LLM Agent Tools (Safe Changes Only)

The agent in `llm/agent.py` uses these tools defined in `llm/tools.py`:

**Read-only tools:**
- `read_file`: Read files with optional line ranges
- `list_directory`: List files with optional recursion
- `search_codebase`: Regex search across codebase
- `find_definition`: Find function/class definitions
- `get_training_logs`: Read training output logs with filters

**Change tools (safe operations only):**
- `modify_config`: Modify values in YAML/JSON/TOML config files
- `set_next_command`: Override CLI command for next training run
- `request_code_change`: Request handoff to coding agent for code changes

The LLM **cannot** write code directly. Code changes are handed off to external coding agents (Claude Code) or paused for manual intervention.

### Coding Agent Handoff

When the LLM calls `request_code_change`, the loop:
1. Creates a `Suggestion` record in the database
2. Builds a `HandoffContext` with iteration history, metrics, and the suggestion
3. Invokes the configured coding agent (or pauses if `type: none`)
4. Records the result and commits any changes

Coding agents implement the `CodingAgent` protocol in `agents/base.py`.

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
- `executor`: `type: local` or `type: ssh` with host/port/key
- `entry.train`: Training command
- `metrics.source`: `wandb` or `eval_json`
- `metrics.project/entity`: W&B project settings (if using wandb)
- `metrics.primary`: Which metric to optimize
- `metrics.minimize`: Whether lower is better
- `metrics.target`: Optional target value for early stopping
- `guardrails`: plateau window, timeout, retry budget
- `coding_agent.type`: `auto`, `claude-code`, or `none`
- `context.deny`: File patterns agent cannot read

## CLI Commands

| Command | Purpose |
|---------|---------|
| `init` | Interactive setup - creates `revis.yaml` |
| `loop` | Start iteration loop (supports `--background`) |
| `resume` | Resume stopped session |
| `stop` | Graceful stop via signal file |
| `status` | Show progress (supports `--watch`) |
| `list` | List all sessions (supports `--all`) |
| `show` | Show session details and iteration history (`--trace` for tool calls) |
| `compare` | Compare two iterations side-by-side |
| `logs` | View session logs |
| `watch` | Attach to tmux session |
| `export` | Export session data as JSON/CSV |
| `pr` | Push branch & create GitHub PR |
| `delete` | Delete session and optionally branch |

## Database Schema

The SQLite store tracks:
- **sessions**: id, name, branch, status, budget, cost
- **runs**: id, session_id, iteration, change_type, change_description, hypothesis, metrics, outcome
- **suggestions**: id, session_id, run_id, type, content, status, handed_off_to
- **traces**: tool call history for debugging

Run fields track what changed (`change_type`: config, cli_args, code_handoff) and why (`hypothesis`).

## Code Style

- Python 3.11+
- Ruff: line-length=100, rules E,F,I,N,W
- Pydantic for config validation
- Rich for terminal output
- Type hints on public APIs
