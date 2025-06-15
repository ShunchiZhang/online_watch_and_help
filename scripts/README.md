# Run Experiments

> - 06/06 thres: grab 0.80, put 0.60, filter 0.20
> - 06/09 thres: grab 0.70, put 0.40, filter 0.10

```sh
cd online_watch_and_help

# single
bash scripts/main.sh 8088 1 MCTS "--num_runs=5"

# random_goal
bash scripts/main.sh 8089 2 MCTS "--helper_goal_type=random --num_runs=5"

# oracle_goal
bash scripts/main.sh 8090 2 MCTS "--helper_goal_type=gt --num_runs=5"

# llm
bash scripts/main.sh 8091 2 AutoToM "--num_runs=2 --autotom_method=llm --autotom_llm_name=gpt-4o"

# autotom
bash scripts/main.sh 9101 2 AutoToM "--num_runs=2 --autotom_method=autotom --autotom_llm_name=gpt-4o"
bash scripts/main.sh 9101 2 AutoToM "--num_runs=2 --autotom_method=autotom --autotom_llm_name=gpt-4o --process_id=1 --episode_ids  0  1  2  3  4  5  6  7  8  9 10"
bash scripts/main.sh 9201 2 AutoToM "--num_runs=2 --autotom_method=autotom --autotom_llm_name=gpt-4o --process_id=2 --episode_ids 11 12 13 14 15 16 17 18 19 20"
bash scripts/main.sh 9301 2 AutoToM "--num_runs=2 --autotom_method=autotom --autotom_llm_name=gpt-4o --process_id=3 --episode_ids 21 22 23 24 25 26 27 28 29 30 31"
```

To re-eval:
```sh
lla logs/2_per_apt-task_all-apts_0,1,2,4,5/*/run_*/results.json
# rm logs/2_per_apt-task_all-apts_0,1,2,4,5/*/run_*/results.json
```

# Inspect Logs

```sh
cd online_watch_and_help

bash scripts/open_log.sh 0 01 result
bash scripts/open_log.sh 0 01 eval
```
