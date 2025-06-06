port=$1
num_agents=$2
autotom_method=$3

set -a
source .env
set +a

lsof -i :${port} -t | xargs -r kill -9

executable_file="../../ShunchiZhang/virtualhome/unity/macos_exec.2.2.4.app"
dataset_path="./dataset/structured_agent/train_env_task_set_2_full_task.all_apts.0,1,2,4,5.pik"

python main.py \
  --executable_file="${executable_file}" \
  --dataset_path="${dataset_path}" \
  --num_agents="${num_agents}" \
  --autotom_method="${autotom_method}" \
  --base_port="${port}" \
  --debug
