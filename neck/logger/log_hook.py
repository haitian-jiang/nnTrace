from typing import Callable
from neck.structs import TensorInfo
from neck import CONFIG

"""
Hooks defined in neck.canonical, as it makes the logged tensor canonical. For
example, we undo the effect of divided by num_microbatches in pipeline parallel
for Megatron-LM.
"""

LOG_HOOKS: list[ Callable[[TensorInfo, str],TensorInfo] ] = []


if CONFIG.parallel.framework == "Megatron-LM":
    from neck.canonical.megatron import log_hook_megatron
    LOG_HOOKS.extend(log_hook_megatron)
elif CONFIG.parallel.framework in ["vLLM", "transformers"]:
    pass
else:
    raise NotImplementedError(
        f"Parallel framework {CONFIG.parallel.framework} is not supported."
    )


__all__ = [
    "LOG_HOOKS",
]
