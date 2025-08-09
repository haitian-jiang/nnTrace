import torch
from neck import CONFIG as NECK_CONFIG
from typing import TYPE_CHECKING
from vllm.config import VllmConfig, ModelConfig, ParallelConfig


def parse_megatron_config(args) -> None:
    if getattr(args, "sequence_parallel", None) is not None:
        # overwrite the value in neck's config yaml if available from megatron
        NECK_CONFIG.parallel.sequence_parallel = args.sequence_parallel
    if getattr(args, "num_layers", None) is not None:
        NECK_CONFIG.architecture.num_layers = args.num_layers
    if getattr(args, "hidden_size", None) is not None:
        NECK_CONFIG.architecture.hidden_size = args.hidden_size
    if getattr(args, "num_attention_heads", None) is not None:
        NECK_CONFIG.architecture.num_attention_heads = args.num_attention_heads
    if getattr(args, "seq_length", None) is not None:
        NECK_CONFIG.architecture.seq_length = args.seq_length
    if getattr(args, "untie_embeddings_and_output_weights", None) is not None:
        share = not args.untie_embeddings_and_output_weights
        NECK_CONFIG.architecture.share_embeddings_and_output_weights = share
    if getattr(args, "bf16", False):
        NECK_CONFIG.architecture.datatype = "bf16"
    if getattr(args, "fp16", False):
        NECK_CONFIG.architecture.datatype = "fp16"
    if getattr(args, "calculate_per_token_loss", False):
        NECK_CONFIG.architecture.calculate_per_token_loss = True
    if getattr(args, "use_distributed_optimizer", False):
        NECK_CONFIG.parallel.use_distributed_optimizer = True
    try:
        from megatron.core.num_microbatches_calculator import get_num_microbatches
        NECK_CONFIG.parallel.num_microbatches = get_num_microbatches()
    except ImportError:
        pass


def parse_vllm_config(vllm_config: VllmConfig) -> None:
    hf_config = vllm_config.model_config.hf_config
    NECK_CONFIG.vocab_size = hf_config.vocab_size
    NECK_CONFIG.num_layers = hf_config.num_hidden_layers
    NECK_CONFIG.hidden_size = hf_config.hidden_size
    NECK_CONFIG.num_attention_heads = hf_config.num_attention_heads
    NECK_CONFIG.num_key_value_heads = hf_config.num_key_value_heads
    vllm_dtype_map = {
        "half": "fp16",
        "float16": "fp16",
        "bfloat16": "bf16",
        "float": "fp32",
        "float32": "fp32",
        torch.half: "fp16",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float: "fp32",
        torch.float32: "fp32",
    }
    if vllm_config.model_config.dtype == "auto":
        pass
    else:
        NECK_CONFIG.architecture.datatype = vllm_dtype_map.get(
            vllm_config.model_config.dtype, "bf16"
        )


def parse_framework_config(args) -> None:
    if NECK_CONFIG.parallel.framework == "Megatron-LM":
        parse_megatron_config(args)
    elif NECK_CONFIG.parallel.framework == "vLLM":
        assert isinstance(args, VllmConfig), \
            "Expected args to be an instance of VllmConfig"
        parse_vllm_config(args)
    else:
        raise NotImplementedError(
            f"Parallel framework {NECK_CONFIG.parallel.framework} is not supported."
        )

__all__ = ["parse_framework_config"]
