from neck import CONFIG as NECK_CONFIG


def parse_framework_config(args) -> None:
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


__all__ = ["parse_framework_config"]
