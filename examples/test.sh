dataset="rel-f1"
tasks=("driver-position" "driver-top3" "driver-dnf")
outer_aggrs=("sum" "mean" "max" "cat_MLP" "cat_choose" "cat_weightedsum")

for task in "${tasks[@]}"; do
    for outer_aggr in "${outer_aggrs[@]}"; do
        for i in {1..10}; do
            echo "Running gnn_node.py with dataset=${dataset}, task=${task}, outer_aggr=${outer_aggr}"
            CUDA_VISIBLE_DEVICES=0 python gnn_node.py --wandb True --wandb_gradient False --epoch 50 --dataset "$dataset" --task "$task" --outer_aggr "$outer_aggr" --wandb_name "$dataset$task"
        done
    done
done