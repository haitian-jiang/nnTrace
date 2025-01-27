
def get_output_layer_canonical_name() -> str:
    if CONFIG.architecture.share_embeddings_and_output_weights:
        return "word_embeddings.weight"
    else:
        return "output_layer.weight"


TP_SIZE_MAPPING = {
    "word_embeddings.weight": {
        "full": "(vocab_size_with_padding, hidden_size)",
        "slice": "(vocab_size_with_padding // tp_size, hidden_size)",
        "dim": 0,
    },
    "position_embeddings.weight": {
        "full": "(seq_length, hidden_size)",
        "slice": "(seq_length, hidden_size)",
        "dim": None,
    },
    "input_layernorm.weight": {
        "full": "(hidden_size,)",
        "slice": "(hidden_size,)",
        "dim": None,
    },
    "input_layernorm.bias": {
        "full": "(hidden_size,)",
        "slice": "(hidden_size,)",
        "dim": None,
    },
    
    # self_attention.*
    "linear_qkv.layer_norm_weight": {
        "full": "(hidden_size,)",
        "slice": "(hidden_size,)",
        "dim": None,
    },
    "linear_qkv.layer_norm_bias": {
        "full": "(hidden_size,)",
        "slice": "(hidden_size,)",
        "dim": None,
    },
    "linear_proj.weight": {
        "full": "(hidden_size, hidden_size)",
        "slice": "(hidden_size, hidden_size // tp_size)",
        "dim": 1,
    },
    "linear_proj.bias": {
        "full": "(hidden_size,)",
        "slice": "(hidden_size,)",
        "dim": None,
    },
    "linear_qkv.weight": {
        "full": "(hidden_size * 3, hidden_size)",
        "slice": "(hidden_size * 3 // tp_size, hidden_size)",
        "dim": 0,
    },
    "linear_qkv.bias": {
        "full": "(hidden_size * 3,)",
        "slice": "(hidden_size * 3 // tp_size,)",
        "dim": 0,
    },
    
    "pre_mlp_layernorm.weight": {  # name in megatron.core
        "full": "(hidden_size,)",
        "slice": "(hidden_size,)",
        "dim": None,
    },
    "pre_mlp_layernorm.bias": {
        "full": "(hidden_size,)",
        "slice": "(hidden_size,)",
        "dim": None,
    },
    
    # mlp.*
    "linear_fc1.layer_norm_weight": {  # name in transformer_engine
        "full": "(hidden_size,)",
        "slice": "(hidden_size,)",
        "dim": None,
    },
    "linear_fc1.layer_norm_bias": {
        "full": "(hidden_size,)",
        "slice": "(hidden_size,)",
        "dim": None,
    },
    "linear_fc1.weight": {
        "full": "(hidden_size * 4, hidden_size)",
        "slice": "(hidden_size * 4 // tp_size, hidden_size)",
        "dim": 0,
    },
    "linear_fc1.bias": {
        "full": "(hidden_size * 4,)",
        "slice": "(hidden_size * 4 // tp_size,)",
        "dim": 0,
    },
    "linear_fc2.weight": {
        "full": "(hidden_size, hidden_size * 4)",
        "slice": "(hidden_size, hidden_size * 4 // tp_size)",
        "dim": 1,
    },
    "linear_fc2.bias": {
        "full": "(hidden_size,)",
        "slice": "(hidden_size,)",
        "dim": None,
    },
    
    "final_layernorm.weight": {
        "full": "(hidden_size,)",
        "slice": "(hidden_size,)",
        "dim": None,
    },
    "final_layernorm.bias": {
        "full": "(hidden_size,)",
        "slice": "(hidden_size,)",
        "dim": None,
    },

    "output_layer.weight": {
        "full": "(vocab_size_with_padding, hidden_size)",
        "slice": "(vocab_size_with_padding // tp_size, hidden_size)",
        "dim": 0,
    },
}