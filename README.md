# nnTrace: some awesome title

## Preparation

Step 1. Prepare conda environment
```bash
conda create -n neck python=3.10
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install packaging regex psutil pyyaml nltk pybind11 einops
pip install setuptools==68.2.2 cmake==3.29.2 ninja==1.11.1.1
```

Step 2. Install Megatron-LM and its supporting libraries

- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine):
  ```bash
  pip install "transformer_engine[pytorch]"
  
  # pip version is 1.13, need to compile from source for >= 2.0
  # below are the commands for installing TE from source
  sudo apt-get -y install cudnn9-cuda-12
  git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.  git
  cd TransformerEngine
  export NVTE_FRAMEWORK=pytorch
  pip install . 
  ```

- [apex](https://github.com/NVIDIA/apex):
  ```bash
  # below copied from apex
  git clone https://github.com/NVIDIA/apex
  cd apex
  # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports   multiple `--config-settings` with the same key... 
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation   --config-settings "--build-option=--cpp_ext" --config-settings   "--build-option=--cuda_ext" ./
  # otherwise
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation   --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
  ```bash
  git clone https://github.com/NVIDIA/Megatron-LM
  ```


## Install NECK

Current build process requires a conda environment with torch and pybind11.

```bash
git clone git@github.com:haitian-jiang/NECK.git
```


Compile C++ extension: 

```bash
conda activate neck
python setup.py build_ext --inplace
```

Installation:

```bash
pip install -e .
```

Dependency:

```bash
pip install loguru
pip install xxhash
```


## Usage (Run time)

Step 1. Modify Megatron-LM

1. Turn off pipeline parallelism hack.

   In file `Megatron-LM/megatron/training/arguments.py`, find the key word 
   `deallocate_pipeline_outputs`, and set its value to False.
   (I have no idea why megatron hard code it.)
   ```python
   x  kw_args['deallocate_pipeline_outputs'] = True
   √  kw_args['deallocate_pipeline_outputs'] = False
   ```

2. Optionally turn off `save_checkpointing`.
   
   Megatron-LM will save the trained parameters each time after training.
   It can be annoying for experiments and takes long time.
   It can be turned off in the following way:

   In file `Megatron-LM/megatron/training/checkpointing.py`, find the function
   `save_checkpoint`, and make it directly return.


Step 2. Add NECK code to Megatron-LM training code.
File to modify: `Megatron-LM/megatron/training/training.py`

1. Import neck.  You can put `import neck` either in the beginning of the file
   or import it every time you call neck API.

2. Reintialize the model and register hooks. 
   This should be done after the model is prepared but not yet passed in to
   initialize the optimizer.
   Such timing is important because the fp32 main parameters in the optimizer
   is read from the model.
   So the optimizer can correctly read the reinitialized consistent parameters.
   
    ```python
    def setup_model_and_optimizer(model_provider_func,
                                  model_type,
                                  no_wd_decay_cond=None,
                                  scale_lr_cond=None,
                                  lr_mult=1.0,
                                  checkpointing_context=None):
        """Setup model and optimizer."""
        args = get_args()
        timers = get_timers()
        one_logger = get_one_logger()

        model = get_model(model_provider_func, model_type)
        unwrapped_model = unwrap_model(model)

    --> neck.parse_framework_config(args)
    --> for vp_rank, model_chunk in enumerate(model):
    -->     neck.reinit_model(model_chunk, vp_rank)
    -->     neck.register_model_hooks(model_chunk, vp_rank)

        kwargs = {}
        for f in dataclasses.fields(OptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        config = OptimizerConfig(**kwargs)
        config.timers = timers
        optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,
                                           scale_lr_cond, lr_mult,
                                           use_gloo_process_groups=args.   enable_gloo_process_groups)
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
        ...
    ```

3. Add code to record the gradients used for update and the parameters after
   weight update.

    ```python
    def train_step(forward_step_func, data_iterator,
                   model, optimizer, opt_param_scheduler, config):
        ...
    --> for vp_rank, model_chunk in enumerate(model):
    -->     neck.prior_optim_logging(model_chunk, vp_rank)

        # Update parameters.

        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        timers('optimizer').stop()

    --> for vp_rank, model_chunk in enumerate(model):
    -->     neck.post_optim_logging_reset(model_chunk, vp_rank)
        ...
    ```

Step 3. Set up NECK configurations.
NECK reads configuration from environment variable `NECK_CONFIG_PATH`.
Set it to the yaml configuration file in the megatron training command.
The example for NECK configuration can be find in `NECK/config/gpt.yaml`.

The configuration file will read ***parameter annotation*** and 
***module annotation***.  
- The example for parameter annotation is `NECK/config/gpt-param.yaml`.
It indicates how (along which dimension) the parameter is sharded for TP and CP.
- The example for module annotation is `NECK/config/gpt-module-te.yaml`.
It is the module annotation for Transformer Engine modules.
It specifies for each module, how should the activations be sharded, whether and
which input / output activation tensors (and their gradients) need to be logged,
and whether to rewrite them to have a fine-grained localization.



## Notice
1. This option must turn on to ban Megatron-LM from bypassing the torch autograd
   system, so NECK can trace the gradient tensors: 
   `--no-gradient-accumulation-fusion`.
2. All dropout must be set to 0.0 because NECK has not supported consistent
   dropout yet.
   The options are: `--attention-dropout 0.0` and `--hidden-dropout 0.0`.
3. Run the training code for 1 iteration will be enough.
   Otherwise the log size will be too large and everything will be slow.


## Examples

### Train GPT

Here is one example to train GPT with 3D parallelism (TP=2, PP=2, DP=2):

```bash
NECK_CONFIG_PATH=/home/ubuntu/repos/NECK/config/gpt.yaml \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
OMP_NUM_THREADS=1 \
NCCL_ALGO=tree \
PYTHONWARNINGS=ignore \
torchrun
--nproc_per_node 8 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000 \
pretrain_gpt.py \
--tensor-model-parallel-size 2 \
--pipeline-model-parallel-size 2 \
--num-layers 24 \
--hidden-size 1024 \
--num-attention-heads 16 \
--seq-length 1024 \
--max-position-embeddings 1024 \
--micro-batch-size 4 \
--global-batch-size 16 \
--lr 0.00015 \
--train-iters 1 \
--lr-decay-iters 320000 \
--lr-decay-style cosine \
--min-lr 1.0e-5 \
--weight-decay 1e-2 \
--lr-warmup-fraction 0.0 \
--clip-grad 1.0 \
--use-mcore-models \
--attention-softmax-in-fp32 \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--bf16 --data-path /workspace/dataset/wikitext/wikitext_text_document \
--vocab-file /workspace/dataset/gpt2-vocab.json \
--merge-file /workspace/dataset/gpt2-merges.txt \
--split 949,50,1 \
--log-interval 100 \
--save-interval 10000 \
--eval-interval 1000 \
--eval-iters 0 \
--save /workspace/checkpoints \
--load /workspace/checkpoints \
--no-gradient-accumulation-fusion \
--transformer-impl transformer_engine
```

### Training LLaMA

Megatron-LM use `pretrain_gpt.py` with different arguments to make the model a
LLaMA model.

There are one modification in the code needs to be done to cater for LLaMA:
In file `Megatron-LM/pretrain_gpt.py`, inside the function `model_provider`, 
add one line in the `GPTModel` initializer to pass in `rope_scaling_factor`.
```python
def model_provider(pre_process=True, post_process=True):
    ...
    if args.use_legacy_models:
        ...
    else:
        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling,
            --> rope_scaling_factor=args.rope_scaling_factor,
            )
    return model
```

Here is the training command for a LLaMA 3.2 1B model with TP and CP:
```bash
NECK_CONFIG_PATH=/home/ubuntu/repos/NECK/config/gpt.yaml \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
NCCL_ALGO=tree \
NVTE_APPLY_QK_LAYER_SCALING=1 \
OMP_NUM_THREADS=1 \
PYTHONWARNINGS=ignore \
torchrun \
--nproc_per_node 2 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000 \
pretrain_gpt.py \
--tensor-model-parallel-size 2 \
--context-parallel-size 2 \
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
--make-vocab-size-divisible-by 128 \
--normalization RMSNorm \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--clip-grad 1.0 \
--init-method-std 0.02 \
--micro-batch-size 4 \
--global-batch-size 16 \
--lr 0.00015 \
--train-iters 1 \
--lr-decay-iters 320000 \
--lr-decay-style cosine \
--min-lr 1.0e-5 \
--weight-decay 1e-2 \
--lr-warmup-fraction 0.0 \
--use-mcore-models \
--bf16 \
--apply-query-key-layer-scaling \
--attention-softmax-in-fp32 \
--disable-bias-linear \
--no-bias-swiglu-fusion \
--no-gradient-accumulation-fusion \
--data-path /workspace/dataset/wikitext/wikitext_text_document \
--vocab-file /workspace/dataset/gpt2-vocab.json \
--merge-file /workspace/dataset/gpt2-merges.txt \
--split 949,50,1 \
--log-interval 100 \
--save-interval 10000 \
--eval-interval 1000 \
--eval-iters 0 \
--save /workspace/checkpoints \
--load /workspace/checkpoints \
--transformer-impl transformer_engine
```

## Usage (Test time)
After running the training script for 1 iteration, the logs are collected and
stored on disk, say the single device training logs are stored in the folder
`oracle`, and the multi-device training logs are stored in the foler `specimen`.  

The logs collected from different devices are scattered in different files, so
we need to merge them first.
```python
import neck
neck.merge_distr_logs("/path/to/specimen")
```
Run this command in either a python script or a python interactive shell.
It will output some messages to the console (I do not use "log" to eliminate 
ambiguity). And these output will also be stored in `specimen/merge.log`.
**The merger can find and locate some bugs.**
For example, when the gradient for weight update has not been all-reduced.
So if there is an *error* message during the merging process, it means NECK
finds a bug.  And it will output relevant information for locating the bug.

After that, we can conduct the differential testing with for the specimen with
the oracle using the following interface:
```python
import neck
neck.conduct_differential_test("oracle", "specimen", "diff_dir")
```
The differences of the two runs are stored in `diff_dir`, and I haven't write
the tool for analyzing this output.  But user can now directly look at the
`*.log` files to identify if there is bug or not.