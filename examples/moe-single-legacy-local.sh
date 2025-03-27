# run with https://github.com/NVIDIA/Megatron-LM/commit/e122536b7645edcb7ebf099b5c92a443f7dbf8e7
# to reproduce https://github.com/NVIDIA/Megatron-LM/issues/599
export PYTHONWARNINGS=ignore
# export CUDA_VISIBLE_DEVICES=
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NECK_CONFIG_PATH=/workspace/repos/NECK/config/legacy-moe.yaml
export NECK_RUN_NAME=moe-single

PARALLEL_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
"

ARCH_ARGS="
    --num-experts 2 \
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
    #--loss-scale 4096.0 \

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
    --no-gradient-accumulation-fusion \
    --transformer-impl local \
"
    # --use-legacy-models \

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
/workspace/repos/Megatron-LM-reproduce/pretrain_gpt.py \
$PARALLEL_ARGS $ARCH_ARGS $PRECISION_ARGS $TRAINING_ARGS $IO_ARGS
