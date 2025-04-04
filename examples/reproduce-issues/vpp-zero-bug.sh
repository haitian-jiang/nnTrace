export PYTHONWARNINGS=ignore
# export CUDA_VISIBLE_DEVICES=
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export NCCL_ALGO=tree
export NVTE_BATCH_MHA_P2P_COMM=1
export NECK_CONFIG_PATH=/home/ubuntu/efs/zhenyus/NECK/config/gpt-mcore-2308.yaml

if [ "$#" -ne 1 ]; then
    echo "./vpp-zero-bug.sh is_distributed"
    return -1
else
    is_distributed=$1
    export NECK_RUN_NAME=vpp-zero-bug-$is_distributed
fi

if [ $is_distributed -eq 1 ]; then
    DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000 \
    "
    PARALLEL_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --use-distributed-optimizer \
    --num-layers-per-virtual-pipeline-stage 2 \
    --pipeline-model-parallel-size 4 \
    "

elif [ $is_distributed -eq 0 ]; then
    DISTRIBUTED_ARGS=""
    PARALLEL_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    "
else
    echo "Error: is_distributed=$is_distributed"
    return -1
fi

ARCH_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --make-vocab-size-divisible-by 128 \
"

PRECISION_ARGS="
    --attention-softmax-in-fp32 \
    --bf16 \
"

TRAINING_ARGS="
    --clip-grad 1.0 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --lr 1.0 \
    --min-lr 1.0 \
    --train-iters 1 \
    --lr-warmup-fraction 0.0 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --no-gradient-accumulation-fusion \
    --transformer-impl local \
"

IO_ARGS="
    --data-path /workspace/dataset/wikitext/wikitext_text_document \
    --vocab-file /workspace/dataset/gpt2-vocab.json \
    --merge-file /workspace/dataset/gpt2-merges.txt \
    --split 949,50,1 \
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --save /workspace/checkpoints \
    --load /workspace/checkpoints
"

torchrun $DISTRIBUTED_ARGS \
/home/ubuntu/efs/zhenyus/Megatron-LM/pretrain_gpt.py \
$PARALLEL_ARGS $ARCH_ARGS $PRECISION_ARGS $TRAINING_ARGS $IO_ARGS
