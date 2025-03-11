from neck import CONFIG as NECK_CONFIG

if NECK_CONFIG.parallel.framework == "Megatron-LM":
    from megatron.core.model_parallel_config import ModelParallelConfig


def parse_framework_config(config: ModelParallelConfig) -> None:
    if getattr(config, "sequence_parallel", None) is not None:
        # overwrite the value in neck's config yaml if available from megatron
        NECK_CONFIG.parallel.sequence_parallel = config.sequence_parallel
    if getattr(config, "num_layers", None) is not None:
        NECK_CONFIG.architecture.num_layers = config.num_layers
    if getattr(config, "hidden_size", None) is not None:
        NECK_CONFIG.architecture.hidden_size = config.hidden_size
    if getattr(config, "num_attention_heads", None) is not None:
        NECK_CONFIG.architecture.num_attention_heads = config.num_attention_heads
    if getattr(config, "bf16", False):
        NECK_CONFIG.architecture.datatype = "bf16"
    if getattr(config, "fp16", False):
        NECK_CONFIG.architecture.datatype = "fp16"
    try:
        from megatron.core.num_microbatches_calculator import get_num_microbatches
        NECK_CONFIG.parallel.num_microbatches = get_num_microbatches()
    except ImportError:
        pass


__all__ = ["parse_framework_config"]
