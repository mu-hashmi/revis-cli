# Revis

Revis automates the ML iteration loop: run training, analyze results, make changes, repeat. You define a training script and target metric, set a budget, and let it run.

## Installation

```bash
pip install revis
# or
uv add revis
```

Requires an LLM API key (any provider supported by LiteLLM):
```bash
export ANTHROPIC_API_KEY=sk-...
```

## Quick Start

```bash
# Initialize in your ML project
revis init

# Edit revis.yaml - set your training command and target metric
vim revis.yaml

# Start the loop
revis loop --name my-experiment --budget 4h
```

The loop runs until the budget is exhausted, metrics plateau, or a target is reached.

## What It Does

Each iteration:
1. Runs your training script (local tmux or remote SSH)
2. Reads metrics from `eval.json` written by your script
3. Analyzes results and decides what to change
4. Modifies code/config and commits
5. Runs again

When done, you get a git branch with the full iteration history. Export it as a PR:

```bash
revis export my-experiment
```

## Setup

Revis needs your training script to output metrics to `eval.json`. Add this to the end of your training script:

```python
import json
import os
from pathlib import Path

# Revis sets REVIS_OUTPUT_DIR; fall back to current dir if running manually
output_dir = Path(os.environ.get("REVIS_OUTPUT_DIR", "."))
output_dir.mkdir(parents=True, exist_ok=True)

# Write your metrics - use whatever names make sense for your task
metrics = {
    "val_loss": val_loss,        # your validation loss variable
    "accuracy": accuracy,         # any other metrics you track
}
with open(output_dir / "eval.json", "w") as f:
    json.dump(metrics, f)
```

Then configure `revis.yaml` to match:

```yaml
executor:
  type: local  # or 'ssh' for remote GPU

entry:
  train: "python train.py"  # your training command

metrics:
  primary: val_loss  # must match a key in your eval.json
  minimize: true     # true for loss, false for accuracy
  # target: 0.1      # optional: stop early if reached

guardrails:
  max_run_duration: 2h
  plateau_runs: 3
```

The `metrics.primary` field must match one of the keys you write to `eval.json`.

## Commands

| Command | Description |
|---------|-------------|
| `revis init` | Create `.revis/` and `revis.yaml` |
| `revis loop` | Start iteration loop |
| `revis status` | Show current progress |
| `revis logs <name>` | View session logs |
| `revis stop <name>` | Stop a running session |
| `revis resume <name>` | Resume a stopped session |
| `revis show <name>` | Show session details |
| `revis export <name>` | Push branch and create PR |
| `revis delete <name>` | Delete a session |

## Remote Execution

To run on a remote GPU server:

```yaml
executor:
  type: ssh
  host: gpu-server.example.com
  user: ubuntu
  port: 22
  key_path: ~/.ssh/id_rsa
  work_dir: /home/ubuntu/project
```

Revis syncs your code, runs training over SSH, and syncs results back.

## Requirements

- Python 3.11+
- tmux (for local execution)
- Git repo (changes are committed per iteration)

## License

Apache 2.0
