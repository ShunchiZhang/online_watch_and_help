---
description: Code structure of this repository
globs:
alwaysApply: true
---

## Documentation

- [setup.md](mdc:/docs/setup.md): Guide to setup the environment
- [usage.md](mdc:/docs/usage.md): Guide to run the experiments and use other tools

## Dataset

Use [vh_init_structured.py](mdc:/gen_data/vh_init_structured.py) to generate datasets of embodied assistance tasks. It calls other modules in [gen_data/](mdc:/gen_data/).

Generated datasets are stored in [dataset/](mdc:/dataset/) as `*.pik` files.

## Core Components

### Environments

- [arena.py](mdc:/envs/arena.py): Main loop for interactions between RL agents and environment
- [unity_environment.py](mdc:/envs/unity_environment.py): RL environment wrapper for VirtualHome
- [graph_env.py](mdc:/envs/graph_env.py): graph-based environment dynamics for MCTS planner

### Agents

[agents/](mdc:/agents/) folder contains different types of agents:
- [MCTS_agent.py](mdc:/agents/MCTS_agent.py): MCTS-based agent
  - [MCTS_utils.py](mdc:/agents/MCTS_utils.py): heuristic motion planner
  - [MCTS.py](mdc:/agents/MCTS.py): MCTS planner
  - [belief.py](mdc:/agents/belief.py): belief state for MCTS
- [GnP_agent.py](mdc:/agents/GnP_agent.py): NOPA-like agent
  - [AutoToM_prompts.py](mdc:/agents/AutoToM_prompts.py): LLM utils, including prompt templates, Pydantic models for structured output and their methods, and async LLM calls.
  - [AutoToM.py](mdc:/agents/AutoToM.py): process proposal distribution and SMC for goal inference

### Entry Points

Use [main.py](mdc:/main.py) and [main.sh](mdc:/tools/main.sh) to start the experiments. The launch arguments are defined in [arguments.py](mdc:/arguments.py).

### Tools and Utils

Utility functions are defined under [utils/](mdc:/utils/) folder:
- [utils_environment.py](mdc:/utils/utils_environment.py): environment-related utilities
- [utils_exception.py](mdc:/utils/utils_exception.py): exception handling utilities
- [utils_graph.py](mdc:/utils/utils_graph.py): graph-related utilities
- [utils_utils.py](mdc:/utils/utils_utils.py): miscellaneous utilities

## Other Tools

- [plot.ipynb](mdc:/tools/plot.ipynb) plots the results in bar chart using the data in [result.csv](mdc:/tools/result.csv).
