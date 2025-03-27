export PYTHONWARNINGS=ignore
# export CUDA_VISIBLE_DEVICES=
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export NCCL_ALGO=tree
export NVTE_BATCH_MHA_P2P_COMM=1
export NECK_CONFIG_PATH=/workspace/repos/NECK/config/gpt.yaml
export NECK_RUN_NAME=

# NOTE: No need to enable NECK in this experiment!  Can recover the patches for neck.
# Just need to patch the VocabEmbedding in layers.py to become buggy.

# This shell script is for doing the experiment of injected bug on VocabEmbedding
# ----- Train buggy model -----
# Step 1. Clone a brand new megatron-lm
# Step 2. Find megatron/core/tensor_parallel/layers.py/@VocabParallelEmbedding.forward()
# Step 3. Replace the original code with the buggy code
#         delete "output_parallel[input_mask, :] = 0.0" and add
#      ```buggy_input_mask = (masked_input < self.vocab_start_index) | (masked_input >= self.vocab_end_index)
#      ```output_parallel[buggy_input_mask, :] = 0.0
#         See details at: https://github.com/haitian-jiang/Megatron-LM/commit/c4b52da36c90d9fd849c59a0f6e839326ab8abd7
# Step 4. Add some print statements in the forward() function and run with --train-iters 1
#         to see if the buggy code is actually executed.
# Step 4. Run this script (--train-iters 10000) and SAVE the checkpoint. 
#         (can replace the --save folder below to something like buggy_checkpoint)

# ----- Validate buggy model -----
# Step 5. Recover the bug: make it back to the original correct implementation
# Step 6. Load the buggy checkpoint and (optionally train for 1 iter) and evaluate it. (set --eval-iters to 1)
#         If Megatron has to train 1 iter before eval, REMEMBER to set the --train-iters to 1 in the TRAINING_ARGS.
#         Maybe you need to delete the latest_checkpoint.txt in the checkpoint folder,
#         I don't know for sure how this validation can be conducted.
#         Maybe you can bypass Megatron's own checkpoint loading and saving, see this for details:
#         https://github.com/haitian-jiang/Megatron-LM/commit/c9cd15f25686c9bfaaff3eabbd8e063fb8c6c73c

# ----- Train the correct model -----
# Step 7. Using correct code (recover the injected bug), 
#         train from scratch for 10000 iters (remember to set --train-iters)
#         Don't need to specifically do validation in this case, as it can be done during the training.
#         So set --eval-iters > 0 when training the correct model.



DISTRIBUTED_ARGS="
    --nproc_per_node 2 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000 \
"

PARALLEL_ARGS="
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
"
    # --use-distributed-optimizer \

ARCH_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --attention-dropout 0.1 \
    --hidden-dropout 0.1 \
    --make-vocab-size-divisible-by 128 \
"

PRECISION_ARGS="
    --attention-softmax-in-fp32 \
    --bf16 \
"
    # --loss-scale 4096.0 \

TRAINING_ARGS="
    --clip-grad 1.0 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --train-iters 10000 \
    --lr-warmup-fraction 0.01 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --use-mcore-models \
    --no-gradient-accumulation-fusion \
    --transformer-impl local \
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
/workspace/repos/Megatron-LM/pretrain_gpt.py \
$PARALLEL_ARGS $ARCH_ARGS $PRECISION_ARGS $TRAINING_ARGS $IO_ARGS
