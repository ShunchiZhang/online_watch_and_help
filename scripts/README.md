# Run Experiments

```sh
cd online_watch_and_help

# single
bash scripts/main.sh 8088 1 MCTS "--num_runs=5"

# random_goal
bash scripts/main.sh 8089 2 MCTS "--helper_goal_type=random --num_runs=5"

# oracle_goal
bash scripts/main.sh 8090 2 MCTS "--helper_goal_type=gt --num_runs=5"

# llm
bash scripts/main.sh 8091 2 AutoToM "--autotom_method=llm --autotom_llm_name=gpt-4o"

# autotom
bash scripts/main.sh 8092 2 AutoToM "--autotom_method=autotom --autotom_llm_name=gpt-4o"
```

# Inspect Logs

```sh
cd online_watch_and_help

bash scripts/open_log.sh 0 01 result
bash scripts/open_log.sh 0 01 eval
```
