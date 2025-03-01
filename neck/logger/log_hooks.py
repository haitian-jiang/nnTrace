from typing import List, Callable
from neck import CONFIG

"""
Hooks defined in neck.canonical, as it makes the logged tensor canonical. For
example, we undo the effect of divided by num_microbatches in pipeline parallel
for Megatron-LM.
"""

GRAD_LOG_HOOKS: List[Callable] = []


if CONFIG.parallel.framework == "Megatron-LM":
    from neck.canonical.megatron import grad_log_hook_megatron
    GRAD_LOG_HOOKS.extend(grad_log_hook_megatron)
else:
    raise NotImplementedError(
        f"Parallel framework {CONFIG.parallel.framework} is not supported."
    )


__all__ = [
    "GRAD_LOG_HOOKS",
]
