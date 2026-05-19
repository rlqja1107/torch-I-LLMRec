
#export NCCL_P2P_DISABLE="1"
#export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES="3"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

data=$1
n_gpu=$2
hugging_token_id=$3
data_path="./dataset/Amazon_18/${data}"
output_dir="./output/${data}"
recsys_model_path="./dataset/Amazon_18/${data}/SASRec_${data}.pth"

torchrun --nnodes=$n_node --nproc_per_node=$n_gpu --master_port=13226 \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    main.py --output_dir $output_dir --per_device_train_batch_size 8 --data_path $data_path --data_type $data --recsys_model_path $recsys_model_path --hugging_token_id $hugging_token_id