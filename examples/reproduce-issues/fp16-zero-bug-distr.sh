# ran with megatron commit 3a46f12e
# oracle script for reproducing https://github.com/NVIDIA/Megatron-LM/issues/663
# need to open main_grad rewrite before in neck.prior_optim
export PYTHONWARNINGS=ignore
# export CUDA_VISIBLE_DEVICES=
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export NCCL_ALGO=tree
export NVTE_BATCH_MHA_P2P_COMM=1
export NECK_CONFIG_PATH=/workspace/repos/NECK/config/gpt.yaml
export NECK_RUN_NAME=gpt-zero-fp16-bug

DISTRIBUTED_ARGS="
    --nproc_per_node 2 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000 \
"

PARALLEL_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --use-distributed-optimizer \
"

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
    --fp16 \
"

TRAINING_ARGS="
    --clip-grad 1.0 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --lr 1.0 \
    --min-lr 1.0 \
    --train-iters 1 \
    --lr-warmup-fraction 0.0 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --use-mcore-models \
    --no-gradient-accumulation-fusion \
    --transformer-impl transformer_engine \
    --no-async-tensor-model-parallel-allreduce \
    --overlap-param-gather \
    --overlap-grad-reduce \
    --no-delay-grad-reduce \
"

IO_ARGS="
    --data-path /workspace/dataset/wikitext_text_document \
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
/workspace/repos/Megatron-LM-reproduce/pretrain_gpt.py \
$PARALLEL_ARGS $ARCH_ARGS $PRECISION_ARGS $TRAINING_ARGS $IO_ARGS
