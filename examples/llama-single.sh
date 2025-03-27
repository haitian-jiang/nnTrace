export PYTHONWARNINGS=ignore
# export CUDA_VISIBLE_DEVICES=
export NVTE_APPLY_QK_LAYER_SCALING=1
export NECK_CONFIG_PATH=/workspace/repos/NECK/config/gpt.yaml
export NECK_RUN_NAME=

PARALLEL_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
"

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
