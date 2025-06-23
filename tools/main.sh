port=$1
num_agents=$2
helper_class=$3
helper_args=$4

set -a
source .env
set +a

lsof -i :${port} -t | xargs -r kill -9

executable_file="../../ShunchiZhang/virtualhome/unity/macos_exec.2.2.4.app"
dataset_path="./dataset/2_per_apt-task_no_tv-apts_0,1,2,4,5-subset_20.pik"

cmd="python main.py \
  --executable_file="${executable_file}" \
  --dataset_path="${dataset_path}" \
  --num_agents="${num_agents}" \
  --base_port="${port}" \
  --helper_class="${helper_class}" \
  --debug \
  ${helper_args}"

echo $cmd
eval $cmd
