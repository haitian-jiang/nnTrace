
## Preparation

Step 1. Prepare conda environment
```bash
conda create -n neck python=3.10
conda activate neck
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# for Megatron-LM
pip install packaging regex psutil pyyaml nltk pybind11 einops six
# for building from source
pip install setuptools>=68.2.2 cmake==3.29.2 ninja==1.11.1.1 wheel>=0.44
```

Step 2. Install Megatron-LM and its supporting libraries

- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine):
  ```bash
  pip install "transformer_engine[pytorch]==1.13.0"
  # current newest version on pypi is 2.1.0, but it has bugs with fp8
  

  # below are the commands for installing TE from source
  # see https://developer.nvidia.com/cudnn-downloads for cudnn
  sudo apt-get -y install cudnn9-cuda-12
  git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git
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
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings   "--build-option=--cuda_ext" ./
  # otherwise
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
  ```bash
  git clone https://github.com/NVIDIA/Megatron-LM
  ```


## Install from source

The C++ extension has dependency on TBB on Linux.  Below is the way to
install TBB locally in python env (if TBB is installed globally and
can be correctly linked, this part can be skipped).

```bash
# apt
sudo apt install libtbb-dev

# pip
pip install tbb
# check if tbb is correctly installed
find $(python -c "import sys; print(sys.prefix)") -name "libtbb.so.12"
# update linker path
export LD_LIBRARY_PATH=/path/to/env/lib:$LD_LIBRARY_PATH
```

Current build process requires a conda environment with torch and pybind11.

```bash
conda activate neck
# clone the repo and enter
# make sure the necessary build tools are installed
pip install setuptools>=68.2.2 cmake==3.29.2 ninja==1.11.1.1 wheel>=0.44
# --no-build-isolation is important because the building procedure
# needs to use the torch in the current environment.
pip install --no-build-isolation -e .
```

If you only want to build the C++ extension, you can run:

```
python setup.py build_ext --inplace
```

Dependency: loguru,  xxhash, omegaconf, numpy, and torch
```bash
# pip install torch==x.y.z --xxx
pip install loguru xxhash omegaconf numpy
```


## Usage (Run time)

Step 1. Modify Megatron-LM

1. Turn off pipeline parallelism hack.

   In file `Megatron-LM/megatron/training/arguments.py`, find the key word 
   `deallocate_pipeline_outputs`, and set its value to False.
   (I have no idea why megatron hard code it.)
   ```python
   ✗  # kw_args['deallocate_pipeline_outputs'] = True
   ✔  kw_args['deallocate_pipeline_outputs'] = False
   ```

2. Turn off `save_checkpoint`.
   
   Megatron-LM will save the trained parameters each time after training.
   It can be annoying for experiments and takes long time.
   It can be turned off in the following way:

   In file `Megatron-LM/megatron/training/checkpointing.py`, find the function
   `save_checkpoint`, and make it directly return.

   **NOTE**: There are other alternative ways to remove the checkpoints, like
   manually remove them after each experiment, or using a monitoring process to
   remove the inodes once any writes happen to the checkpoint saving directory.
   But they are all more complicated than turning off checkpointing.
   Megatron will read these checkpoints before training and skip the
   training if the checkpoints' iteration count is no less than the
   iterations to train.
   So if the checkpoint is not removed, or not skipped saving, later experiments
   will skip the training at all, making the tracing in vain.

3. Optionally close dataset `num_samples` check.
   
   In earlier version of megatron (for example 
   [`6140bf8`](https://github.com/NVIDIA/Megatron-LM/commits/6140bf8deb6f944f3bf1b71e5341b9489b42217a)),
   there is an assertion in `Megatron-LM/megatron/core/datasets/megatron_dataset.py`,
   `__init__` function of `MegatronDataset` class: `assert num_samples > 0`.
   It will prevent training with `--eval-iters 0`.
   Having this option set to 0 disables doing validation and test runs, which
   rules out their inference activation in the forward pass, and it is a must
   for multi-iter training tracing.  Otherwise the iteration counter will be
   wrong.  So comment this assertion out if it exists.


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
    -->     neck.prior_optim(model_chunk, vp_rank)

        # Update parameters.

        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        timers('optimizer').stop()

    --> neck.logger.update_iter_batch_cnt()
    --> for vp_rank, model_chunk in enumerate(model):
    -->     neck.post_optim(model_chunk, vp_rank)
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

Step 4. Set up NECK run name.
NECK uses run name (the name for each experiment) to name the log folders.
The run name can be set in the aforementioned configuration file.  It can also
be set in environment variable `NECK_RUN_NAME`.  The environment variable has a
higher priority to be used over the name from configuration file.  If neither
were set, NECK will use the storing time as the folder name.

**NOTE**: For step 3 and step 4, they can be set in the shell scripts for
running Megatron-LM.  See `examples/` for more details.


## Notice
1. This option must turn on to ban Megatron-LM from bypassing the torch autograd
   system, so NECK can trace the gradient tensors: 
   `--no-gradient-accumulation-fusion`.
2. All dropout must be set to 0.0 because NECK has not supported consistent
   dropout yet.
   The options are: `--attention-dropout 0.0` and `--hidden-dropout 0.0`.
3. Run the training code for 1 iteration will be enough.
   Otherwise the log size will be too large and everything will be slow.


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
neck.conduct_differential_test("oracle_run_name", "specimen_run_name", "diff_dir")
```
The differences of the two runs are stored in `diff_dir`, and I haven't write
the tool for analyzing this output.  But user can now directly look at the
`*.log` files to identify if there is bug or not.



## Examples
Check `examples/` to see the commands for training GPT and LLaMA models with
NECK environment variable setups.

**Training LLaMA**

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
