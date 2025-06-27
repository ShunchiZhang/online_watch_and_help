# How to Run

> For VSCode/Cursor users, use [.vscode/launch.json](/.vscode/launch.json) to debug.

create `.env` and set `OPENAI_API_KEY` and `GEMINI_API_KEY`

```sh
cd online_watch_and_help
uv-activate vhome

# single
bash tools/main.sh 8088 1 MCTS "--num_runs=5"

# random_goal
bash tools/main.sh 8589 2 MCTS "--num_runs=3 --helper_goal_type=random"

# oracle_goal
bash tools/main.sh 8290 2 MCTS "--num_runs=3 --helper_goal_type=gt"

# GnP_llm
bash tools/main.sh 8188 2 GnP "--num_runs=3 --autotom_method=llm --autotom_llm_name=gpt-4o"

# GnP_autotom
bash tools/main.sh 8090 2 GnP "--num_runs=3 --autotom_method=autotom --autotom_llm_name=gpt-4o"
bash tools/main.sh 8091 2 GnP "--num_runs=3 --autotom_method=autotom --autotom_llm_name=gpt-4o --process_id=1 --episode_ids  0  1  2  3  4  5  6"
bash tools/main.sh 8092 2 GnP "--num_runs=3 --autotom_method=autotom --autotom_llm_name=gpt-4o --process_id=2 --episode_ids  7  8  9 10 11 12 13"
bash tools/main.sh 8093 2 GnP "--num_runs=3 --autotom_method=autotom --autotom_llm_name=gpt-4o --process_id=3 --episode_ids 14 15 16 17 18 19"
```

To re-eval:
```sh
lla logs/2_per_apt-task_no_tv-apts_0,1,2,4,5-subset_20/*/run_*/results.json
# rm logs/2_per_apt-task_no_tv-apts_0,1,2,4,5-subset_20/*/run_*/results.json
```

# Inspect Logs

```sh
cd online_watch_and_help

bash scripts/open_log.sh 0 01 result
bash scripts/open_log.sh 0 01 eval
```
