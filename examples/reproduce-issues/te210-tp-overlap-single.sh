export PYTHONWARNINGS=ignore
# export CUDA_VISIBLE_DEVICES=
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NECK_CONFIG_PATH=/workspace/repos/NECK/config/gpt.yaml
export NECK_RUN_NAME=tp-overlap-bug-single

PARALLEL_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
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
    --swiglu \
    --norm-epsilon 1e-5 \
    --normalization RMSNorm \
    --disable-bias-linear \
    --use-rope-scaling \
    --untie-embeddings-and-output-weights \
    --no-position-embedding \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --rotary-percent 1.0 \
    --group-query-attention \
    --num-query-groups 8 \
"

PRECISION_ARGS="
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --bf16 \
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
    --no-masked-softmax-fusion \
    --transformer-impl transformer_engine \
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

torchrun \
/workspace/repos/Megatron-LM/pretrain_gpt.py \
$PARALLEL_ARGS $ARCH_ARGS $PRECISION_ARGS $TRAINING_ARGS $IO_ARGS
