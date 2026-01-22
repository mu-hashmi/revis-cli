# Revis

Revis automates the ML iteration loop: run training, analyze results, make changes, repeat. You define a training script and target metric, set a budget, and let it run.

## Installation

```bash
pip install revis
# or
uv add revis
```

**With W&B support** (recommended):
```bash
pip install revis[wandb]
# or
uv add revis --extra wandb
```

Requires an LLM API key (any provider supported by LiteLLM):
```bash
export ANTHROPIC_API_KEY=sk-...
```

## Quick Start

```bash
# Initialize in your ML project (interactive setup)
revis init

# Start the loop
revis loop --name my-experiment --budget 4h
```

The `init` command walks you through setup interactively:

```
$ revis init

Training command: python train.py --config configs/base.yaml

Metrics source:
  ❯ Weights & Biases
    eval.json (manual)

Primary metric to optimize:
  ❯ val_loss
    eval/perplexity

Objective:
  ❯ minimize
    maximize

Execution environment:
  ❯ Local (tmux)
    gpu-server (from ~/.ssh/config)

Coding agent (for code changes):
  ❯ auto (claude-code detected)
    none (pause for manual changes)

The LLM can read your repo. Hide any files?
  ❯ No, allow full access (recommended)
    Yes, select files to hide

Revis initialized!
You can edit revis.yaml to adjust settings at any time.
```

The loop runs until the budget is exhausted, metrics plateau, or a target is reached.

## Main Loop

Each iteration:
1. Runs your training script (local tmux or remote SSH)
2. Collects metrics (from W&B or `eval.json`)
3. Analyzes results and decides what to change
4. Makes safe changes (config files, CLI args) or hands off code changes to a coding agent
5. Commits and runs again

When done, you get a git branch with the full iteration history:

```bash
# View session details and iteration history
revis show my-experiment

# Create a PR from the experiment branch
revis pr my-experiment
```

## Metrics Collection

Revis supports two metrics sources:

### Weights & Biases (Recommended)

If you use W&B, Revis can automatically detect your projects and pull metrics:

```yaml
metrics:
  source: wandb
  project: my-project
  entity: my-team  # optional
  primary: val_loss
  minimize: true
```

### eval.json (Manual)

Add this to your training script to write metrics:

```python
import json
import os
from pathlib import Path

output_dir = Path(os.environ.get("REVIS_OUTPUT_DIR", "."))
output_dir.mkdir(parents=True, exist_ok=True)

metrics = {
    "val_loss": val_loss,
    "accuracy": accuracy,
}
with open(output_dir / "eval.json", "w") as f:
    json.dump({"metrics": metrics}, f)
```

## How Changes Work

Revis uses a **safe-by-default** approach:

1. **Config changes**: The LLM can modify YAML/JSON config files directly
2. **CLI argument changes**: The LLM can change training command arguments
3. **Code changes**: Handed off to a coding agent (Claude Code) or paused for manual intervention

This keeps the LLM focused on experiment iteration while letting specialized tools handle code edits.

Configure coding agent behavior in `revis.yaml`:

```yaml
coding_agent:
  type: auto          # auto, claude-code, or none
  auto_handoff: true  # false = pause and ask before handing off
  verify: true        # run smoke test after code changes
```

## Commands

| Command | Description |
|---------|-------------|
| `revis init` | Interactive setup - creates `revis.yaml` |
| `revis loop --name <name> --budget <budget>` | Start iteration loop |
| `revis status` | Show current session progress |
| `revis list` | List all sessions |
| `revis show <name>` | Show session details and iteration history |
| `revis compare <name> <i1> <i2>` | Compare two iterations |
| `revis logs <name>` | View session logs |
| `revis watch <name>` | Attach to running tmux session |
| `revis stop` | Stop the running session |
| `revis resume <name>` | Resume a stopped session |
| `revis export <name>` | Export data as JSON/CSV |
| `revis pr <name>` | Push branch and create GitHub PR |
| `revis delete <name>` | Delete a session |

## Remote Execution

To run on a remote GPU server, select an SSH host during `revis init` or configure manually:

```yaml
executor:
  type: ssh
  host: gpu-server.example.com
  user: ubuntu
  port: 22
  key_path: ~/.ssh/id_rsa
  work_dir: /home/ubuntu/project
```

Hosts from `~/.ssh/config` are automatically detected during init.

## Requirements

- Python 3.11+
- tmux (for local execution)
- Git repo (changes are committed per iteration)
- Optional: `pip install revis[wandb]` for W&B metrics integration
- Optional: `claude` CLI for code change handoffs

## License

Apache 2.0
