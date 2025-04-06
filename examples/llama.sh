set -e
export PYTHONWARNINGS=ignore
# export CUDA_VISIBLE_DEVICES=
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export NCCL_ALGO=tree
export NVTE_BATCH_MHA_P2P_COMM=1
export NVTE_APPLY_QK_LAYER_SCALING=1
export NECK_CONFIG_PATH=/home/ubuntu/efs/zhenyus/NECK/config/gpt.yaml

if [ "$#" -ne 6 ]; then
    echo "./llama-distr.sh dp pp vpp tp sp cp"
    return -1
fi

dp=$1
pp=$2
vpp=$3
tp=$4
sp=$5
cp=$6

nproc_per_node=$(($dp * $pp * $tp * $cp))

if [ "$sp" -eq 0 ]; then
    sp_arg=""
elif [ "$sp" -eq 1 ]; then
    sp_arg="--sequence-parallel"
else
    echo "sp=$sp"
    return -2
fi

if [ "$vpp" -eq 1 ]; then
    vpp_arg=""
else 
    vpp_arg="--num-virtual-stages-per-pipeline-rank $vpp"
fi


ARCH_ARGS="
    --num-layers 4 \
    --hidden-size 128 \
    --ffn-hidden-size 128 \
    --num-attention-heads 4 \
    --seq-length 16 \
    --max-position-embeddings 16 \
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
    --micro-batch-size 2 \
    --global-batch-size 8 \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --train-iters 1 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --use-mcore-models \
    --no-gradient-accumulation-fusion \
    --transformer-impl transformer_engine \
    $sp_arg
"
# use local instead of te
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


DISTRIBUTED_ARGS="
--nproc_per_node $nproc_per_node \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000 \
"

# Set parallel arguments.
PARALLEL_ARGS="
    --tensor-model-parallel-size ${tp} \
    --pipeline-model-parallel-size ${pp} \
    --context-parallel-size ${cp} \
    $vpp_arg
"

torchrun $DISTRIBUTED_ARGS \
    /home/ubuntu/efs/zhenyus/Megatron-LM/pretrain_gpt.py \
    $PARALLEL_ARGS $ARCH_ARGS $PRECISION_ARGS $TRAINING_ARGS $IO_ARGS
