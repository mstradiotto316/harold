# Harold Isaac Lab Extension

## Overview

Isaac Lab extension for the Harold quadruped robot. Provides direct RL environments
for flat and rough terrain locomotion training using skrl.

For Harold workflows, prefer `AGENTS.md` and `python scripts/harold.py`.

## Installation

### Runtime context

Harold desktop workflows assume the Python environment at `/home/matteo/Desktop/env_isaaclab`:

```bash
source ~/Desktop/env_isaaclab/bin/activate
```

That activation is necessary for repo Python tooling, but it is not sufficient for all Isaac Sim imports. Simulator-backed modules such as `omni.*` and some `isaacsim.*` paths require the Isaac Sim app/runtime context created by Isaac Lab launcher scripts. In practice:

- Use `python scripts/harold.py ...` for normal Harold training and monitoring.
- Use Isaac Lab app entrypoints such as `python harold_isaac_lab/scripts/skrl/train.py ...` when working directly with simulator execution.
- Treat `omni` import failures in a plain shell as a runtime-context problem before assuming a missing library.

### Install the extension

Using a python interpreter that has Isaac Lab installed, install the library in editable mode:

```bash
python -m pip install -e harold_isaac_lab/source/harold_isaac_lab
```

### Verify installation

- List available tasks:

    ```bash
    python harold_isaac_lab/scripts/list_envs.py
    ```

- Run a task:

    ```bash
    python harold_isaac_lab/scripts/skrl/train.py --task=<TASK_NAME>
    ```

- Run with dummy agents (useful for verifying environment configuration):

    ```bash
    # Zero-action agent
    python scripts/zero_agent.py --task=<TASK_NAME>

    # Random-action agent
    python scripts/random_agent.py --task=<TASK_NAME>
    ```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/harold_isaac_lab"
    ]
}
```
