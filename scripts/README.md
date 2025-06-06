# Run Experiments

```sh
cd online_watch_and_help

# logs/train_env_task_set_2_full_task.all_apts.0,1,2,4,5/single/main.log
bash scripts/main.sh 8088 1 null

# logs/train_env_task_set_2_full_task.all_apts.0,1,2,4,5/llm/main.log
bash scripts/main.sh 8089 2 llm

# logs/train_env_task_set_2_full_task.all_apts.0,1,2,4,5/autotom/main.log
bash scripts/main.sh 8090 2 autotom
```

# Inspect Logs

```sh
cd online_watch_and_help

bash scripts/open_log.sh 0 01 result
bash scripts/open_log.sh 0 01 eval
```
