dataset="rel-f1"
tasks=("driver-position" "driver-top3" "driver-dnf")

for task in "${tasks[@]}"; do
    for i in {1..10}; do
        echo "Running gnn_node.py with dataset=${dataset}, task=${task}, outer_aggr=${outer_aggr}"
        CUDA_VISIBLE_DEVICES=0 python gnn_node.py --wandb True --wandb_gradient False --epoch 50 --dataset "$dataset" --task "$task" --outer_aggr "sum" --wandb_name "${dataset}${task}_changegraph"
    done
done