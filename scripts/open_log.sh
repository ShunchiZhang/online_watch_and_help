dataset_name="2_per_apt-task_no_tv-apts_0,1,2,4,5-subset_20"

run_id=$1
episode_id=$2
file_name=$3

for method in single gpt-4o autotom; do
  cursor logs/${dataset_name}/${method}/run_${run_id}/episode_${episode_id}/${file_name}.json
done
