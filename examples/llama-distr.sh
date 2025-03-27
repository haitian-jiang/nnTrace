set -e
export PYTHONWARNINGS=ignore
# export CUDA_VISIBLE_DEVICES=
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export NCCL_ALGO=tree
export NVTE_BATCH_MHA_P2P_COMM=1
export NVTE_APPLY_QK_LAYER_SCALING=1
export NECK_CONFIG_PATH=/home/ubuntu/efs/zhenyus/NECK/config/gpt.yaml

ARCH_ARGS="
    --num-layers 16 \
    --hidden-size 2048 \
    --ffn-hidden-size 8192 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --rotary-percent 1.0 \
    --use-rope-scaling \
    --rope-scaling-factor 32 \
    --apply-query-key-layer-scaling \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --make-vocab-size-divisible-by 128 \
    --disable-bias-linear \
    --no-bias-swiglu-fusion \
    --no-gradient-accumulation-fusion \
"

PRECISION_ARGS="
    --attention-softmax-in-fp32 \
    --bf16 \
    --fp8-format hybrid \
"

TRAINING_ARGS="
    --clip-grad 1.0 \
    --init-method-std 0.02 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --train-iters 1 \
    --lr-warmup-fraction 0.0 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --use-mcore-models \
    --no-gradient-accumulation-fusion \
    --transformer-impl transformer_engine \
"
# --transformer-impl local \

IO_ARGS="
    --data-path /workspace/dataset/wikitext/wikitext_text_document \
    --vocab-file /workspace/dataset/gpt2-vocab.json \
    --merge-file /workspace/dataset/gpt2-merges.txt \
    --split 949,50,1 \
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --save /workspace/checkpoints/null \
    --load /workspace/checkpoints/null
"

tp_cp_pairs=("1 2" "1 4" "1 8" "2 1" "4 1" "8 1" )

for pair in "${tp_cp_pairs[@]}"; do
    read TP CP <<< "$pair"

    # Modify run name and checkpoint directory based on current configuration.
    export NECK_RUN_NAME="tp${TP}_cp${CP}"
    
    DISTRIBUTED_ARGS="
    --nproc_per_node $((TP * CP)) \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000 \
    "

    # Set parallel arguments.
    PARALLEL_ARGS="
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size ${CP} \
    "

    echo "Running with TP=${TP} and CP=${CP}"
    
    torchrun $DISTRIBUTED_ARGS \
        /home/ubuntu/efs/zhenyus/Megatron-LM/pretrain_gpt.py \
        $PARALLEL_ARGS $ARCH_ARGS $PRECISION_ARGS $TRAINING_ARGS $IO_ARGS
done