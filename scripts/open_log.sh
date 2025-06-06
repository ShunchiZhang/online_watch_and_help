dataset_name="train_env_task_set_2_full_task.all_apts.0,1,2,4,5"

run_id=$1
episode_id=$2
file_name=$3

for method in single gpt-4o autotom; do
  cursor logs/${dataset_name}/${method}/run_${run_id}/episode_${episode_id}/${file_name}.json
done
