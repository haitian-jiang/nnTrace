import torch
from collections import defaultdict
from typing import Literal, Optional
from abc import ABC, abstractmethod
from neck.structs import (
    TensorAnnotation, 
    ModuleAnnotation, 
    TensorInfo, 
    ShardNDSlices,
    TType,
)
from .logger import log_tensor
from neck import CONFIG
import neck.canonical as canonical
import neck.random


fwd_pass_cnt = defaultdict(lambda: -1)  # Counting from 0
bwd_pass_cnt = defaultdict(lambda: -1)
iter_cnt = 0  # also counting from 0 in consistency with the batch counter


def update_iter_batch_cnt() -> None:
    global iter_cnt
    iter_cnt += 1
    for vp_rank in fwd_pass_cnt.keys():
        fwd_pass_cnt[vp_rank] = -1
    for vp_rank in bwd_pass_cnt.keys():
        bwd_pass_cnt[vp_rank] = -1


def get_param_logging_cname(canon_param_name: str) -> str:
    return f"iter.{iter_cnt}|{canon_param_name}"


__doc__ = """
Log hooks (either forward or backward) should only log unmodified tensors.
Rewrite hooks should always happen after log hooks.
Whenever there is a rewrite hook, there should be a log hook before it.
Sparsify hooks should happen after log hooks and rewrite hooks.
"""


class ModuleHookBase(ABC):
    def __init__(
        self, 
        canon_mod_name: str, 
        t_annotation: TensorAnnotation,
        m_annotation: ModuleAnnotation,
        module: torch.nn.Module,
        vp_rank: int,
    ):
        self.canon_mod_name = canon_mod_name  # canonical module name
        self.t_annotation = t_annotation
        self.m_annotation = m_annotation  # to access output_idx
        self.module = module
        self.dp_rank = CONFIG.parallel.rank.dp_rank
        self.dp_size = CONFIG.parallel.size.dp_size
        self.vp_rank = vp_rank

    def get_canon_act_grad_name(
        self, 
        i_o_: Literal["input", "output"] = "output", 
        idx: int = 0,
        is_grad: bool = False,
    ) -> str:
        # fwd pre hook: input, fwd post hook: output
        assert i_o_ in ("input", "output"), "Invalid input/output type"
        if is_grad:
            t_type = TType.g_activ
            called_times = bwd_pass_cnt[self.vp_rank]
        else:
            t_type = TType.activ
            called_times = fwd_pass_cnt[self.vp_rank]

        g_batch_id = canonical.calc_global_batch_id(
            called_times,
            self.dp_size,
            self.dp_rank,
        )

        # TODO: now is string replacement, could be more general like regex
        module_name = self.canon_mod_name
        if self.m_annotation.fusion is not None:
            if i_o_ == "input":
                module_name = module_name.replace(
                    self.m_annotation.suffix, 
                    self.m_annotation.fusion[0]
                )
            elif i_o_ == "output":
                module_name = module_name.replace(
                    self.m_annotation.suffix, 
                    self.m_annotation.fusion[-1]
                )

        return f"iter.{iter_cnt}|batch.{g_batch_id}|" \
               f"{module_name}|{t_type}.{i_o_}.{idx}"

    def get_tensor_info(
        self, 
        activ: torch.Tensor,
        canonical_name: str,
        is_grad: bool = False,
    ) -> TensorInfo:
        shape_slice = canonical.infer_slice(activ.shape, self.t_annotation)
        # NOTE: no double checking of shape and slice here, 
        # could add if exists in annotation.
        return TensorInfo(
            tensor=activ.detach().cpu(),
            canonical_name=canonical_name,
            shard_slice=shape_slice,
            module_class=self.module.__class__.__name__,
            is_grad=is_grad,
        )  # TODO: add info of whether from module with input rewritten.


class FwdPreLogHook(ModuleHookBase):
    """
    Must be prepend to a module at last, because it need to log out the output
    of the last module, which must NOT been already rewritten.
    NOTE: only support tensors in `args`, no `kwargs`.  Could add if needed.
    """

    def __call__(
        self, 
        module: torch.nn.Module, 
        args: tuple[torch.Tensor],
    ) -> None:
        """
        Log the input (actually output of last module) activation.
        The input (`args`) should be a tuple of one single tensor.
        Dynamically construct the canonical name from module name,
        input / output, and global batch index.
        Get shape and slice from tensor annotation stored during construction.
        """
        assert module is self.module, \
            f"module mismatch in {self.canon_mod_name} pre log"
        if len(args) == 1:
            self.m_annotation.input_idx = [0]
        if self.m_annotation.input_idx is not None:
            idx: list[int] = self.m_annotation.input_idx
            for i in idx:
                activation = args[i]
                assert isinstance(activation, torch.Tensor), \
                    "fwd_pre_log expects a tensor"
                canon_act_name = self.get_canon_act_grad_name("input", i)
                tensor_info = self.get_tensor_info(activation, canon_act_name)
                log_tensor(canon_act_name, TType.activ, tensor_info)
        else:
            for i, activation in enumerate(args):
                if not isinstance(activation, torch.Tensor):
                    continue
                self.action(activation, i)


class FwdPreLogHook(FwdPreHook):
    """
    Must be prepend to a module at last, because it need to log out the output
    of the last module, which must NOT been already rewritten.
    NOTE: only support tensors in `args`, no `kwargs`.  Could add if needed.
    """
    def action(self, tensor: torch.Tensor, i: int) -> None:
        """
        Dynamically construct the canonical name from module name,
        input / output, and global batch index.
        Get shape and slice from tensor annotation stored during construction.
        """
        assert isinstance(tensor, torch.Tensor), "fwd_pre_log expects a tensor"
        canon_act_name = self.get_canon_act_grad_name("input", i)
        tensor_info = self.get_tensor_info(tensor, canon_act_name)
        log_tensor(canon_act_name, TType.activ, tensor_info)


class FwdPreRewriteHook(FwdPreHook):
    """
    Rewrite all tensors in the input of the module.
    NOTE: Must directly rewrite the tensor.data.  Return the new tensor will 
    break the computation graph by losing the `grad_fn` attribute.
    """
    def action(self, tensor: torch.Tensor, i: int) -> None:
        assert isinstance(tensor, torch.Tensor), "fwd_pre_rewrite expects a tensor"
        canon_act_name = self.get_canon_act_grad_name("input", i)
        shard_slice = canonical.infer_slice(tensor.shape, self.t_annotation)
        new_activation = neck.random.rand(
            canon_act_name, 
            shard_slice,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        new_activation = 0.02 * (new_activation-0.5)  # rescale to [-0.01, 0.01]
        tensor.data = new_activation.data

    def __call__(
        self, 
        module: torch.nn.Module, 
        args: tuple,
    ) -> None:
        assert module is self.module, \
            f"module mismatch in {self.canon_mod_name} pre rewrite"
        if len(args) == 1:
            self.m_annotation.input_idx = 0
        if self.m_annotation.input_idx is not None:
            idx: list[int] = self.m_annotation.input_idx
            for i in idx:
                self._rewrite_input(args[i], i)
        else:  # TODO: can force use must provide input_idx
            for i, arg in enumerate(args):
                if not isinstance(arg, torch.Tensor):
                    continue
                # TODO: inputs use the same t_annotation, but they may not be sharded in the same way.
                self._rewrite_input(arg, i)
class FwdPostHook(ModuleHookBase, ABC):

    @abstractmethod
    def action(self, activation: torch.Tensor, i: int) -> None:
        raise NotImplementedError
    
    def __call__(
        self,
        module: torch.nn.Module,
        args: tuple,
        output: torch.Tensor | tuple,  # python >= 3.10
    ) -> None:
        # checkings
        if self.module is not None:
            assert module == self.module, f"module mismatch in {self.canon_mod_name} post hook"
        assert isinstance(output, torch.Tensor) or isinstance(output, tuple), \
            "fwd_post_hook expects a tensor or a tuple of tensors"
        
        if isinstance(output, torch.Tensor):
            self.action(output, 0)
        elif self.m_annotation.output_idx is not None:
            i = self.m_annotation.output_idx  # TODO: can consider make it list
            activation = output[i]
            assert isinstance(activation, torch.Tensor), "fwd_post_hook expects a tensor"
            self.action(activation, i)
        else:
            for i, activation in enumerate(output):
                if isinstance(activation, torch.Tensor):
                    self.action(activation, i)

class FwdPostLogHook(FwdPostHook):
    def action(self, activation: torch.Tensor, idx: int = 0) -> None:
        canon_act_name = self.get_canon_act_grad_name("output", idx)
        tensor_info = self.get_tensor_info(activation, canon_act_name)
        log_tensor(canon_act_name, TType.activ, tensor_info)


class FwdPostRewriteHook(FwdPostHook):
    def action(self, tensor: torch.Tensor, idx: int = 0) -> None:
        assert isinstance(tensor, torch.Tensor), "fwd_post_rewrite expects a tensor"
        canon_act_name = self.get_canon_act_grad_name("output", idx)
        shard_slice = canonical.infer_slice(tensor.shape, self.t_annotation)
        new_activation = neck.random.rand(
            canon_act_name, 
            shard_slice,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        new_activation = 0.02 * (new_activation-0.5)  # rescale to [-0.01, 0.01]
        tensor.data = new_activation.data

    def __call__(
        self,
        module: torch.nn.Module,
        args: tuple,
        output: torch.Tensor | tuple,
    ):
        # checkings
        if self.module is not None:
            assert module == self.module, f"module mismatch in {self.canon_mod_name} post log"
        assert isinstance(output, torch.Tensor) or isinstance(output, tuple), \
            "fwd_post_log expects a tensor or a tuple of tensors" 
        
        if isinstance(output, torch.Tensor):
            self._rewrite_output(output, 0)
        elif self.m_annotation.output_idx is not None:
            idx = self.m_annotation.output_idx
            activation = output[idx]
            assert isinstance(activation, torch.Tensor), "fwd_post_log expects a tensor"
            self._rewrite_output(activation, idx)
        else:
            for idx, activation in enumerate(output):
                if isinstance(activation, torch.Tensor):
                    self._rewrite_output(activation, idx)

class BwdPreHook(ModuleHookBase, ABC):

    @abstractmethod
    def action(self, grad: torch.Tensor, i: int) -> None:
        raise NotImplementedError
    
    def __call__(
        self,
        module: torch.nn.Module,
        grad_output: tuple[torch.Tensor],
    ) -> None:
        if self.m_annotation.output_idx is not None:
            i = self.m_annotation.output_idx  # TODO: consider using list
            grad = grad_output[i]
            assert isinstance(grad, torch.Tensor), \
                f"Got {type(grad)} grad_input in {self.canon_mod_name}"
            self.action(grad, i)
            return
        for i, grad in enumerate(grad_output):
            if grad is None:
                continue
            assert isinstance(grad, torch.Tensor), \
                f"bwd_pre_hook get non tensor: {type(grad)}"
            self.action(grad, i)

class BwdPreLogHook(BwdPreHook):
    """
    Use it only because need to log the `grad_input` (result) of the module 
    closer to loss, so must be executed before the rewrite hook.
    Use it only when necessary.
    """
    def action(self, grad: torch.Tensor, idx: int) -> None:
        canon_grad_name = self.get_canon_act_grad_name("output", idx, True)
        tensor_info = self.get_tensor_info(grad, canon_grad_name, True)
        log_tensor(canon_grad_name, TType.g_activ, tensor_info)


class BwdPreRewriteHook(BwdPreHook):
    """
    There should not be any weight gradient, so can safely rewrite the 
    grad_output.  Should always use this to rewrite.
    """
    def action(self, grad: torch.Tensor, idx: int) -> None:
        assert isinstance(grad, torch.Tensor), \
            f"bwd pre hook get non tensor: {type(grad)}"
        canon_grad_name = self.get_canon_act_grad_name("output", idx, True)
        shard_slice = canonical.infer_slice(grad.shape, self.t_annotation)
        new_grad = neck.random.rand(
            canon_grad_name, 
            shard_slice,
            device=grad.device,
            dtype=grad.dtype,
        )
        new_grad = 0.02 * (new_grad - 0.5)  # rescale to [-0.01, 0.01]
        grad.data = new_grad.data


    def __call__(
        self,
        module: torch.nn.Module,
        grad_output: tuple[torch.Tensor],
    ) -> None:
        if self.m_annotation.output_idx is not None:
            i = self.m_annotation.output_idx
            grad = grad_output[i]
            assert isinstance(grad, torch.Tensor), \
                f"Got {type(grad)} grad_output in {self.canon_mod_name}"
            self._rewrite_grad_output(grad, i)
            return
        for i, grad in enumerate(grad_output):
            if grad is None:
                continue
            assert isinstance(grad, torch.Tensor), \
                f"bwd pre hook get non tensor: {type(grad)}"
            self._rewrite_grad_output(grad, i)


class BwdPostLogHook(ModuleHookBase):
    def __call__(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor],
        grad_output: tuple[torch.Tensor],
    ) -> None:
        for idx, grad in enumerate(grad_input):
            if grad is None or not torch.is_floating_point(grad):
                # either None or torch.Tensor in C++ backend,
                # but could be attention mask (BoolTensor)
                # TODO: or check input_idx in m_annotation
                continue
            assert isinstance(grad, torch.Tensor), \
                f"bwd post hook get non tensor {type(grad)}"
            canon_grad_name = self.get_canon_act_grad_name("input", idx, True)
            tensor_info = self.get_tensor_info(grad, canon_grad_name, True)
            log_tensor(canon_grad_name, TType.g_activ, tensor_info)


def register_hook(
    module: torch.nn.Module,
    canon_mod_name: str,
    annotation: "ModuleAnnotation",
    vp_rank: int,
) -> None:
    assert isinstance(module, torch.nn.Module), \
        "module must be an instance of torch.nn.Module"
    assert hasattr(module, "forward"), \
        "module must have a forward method"
    
    """
    forward pre hooks: 
      log output from last module 
      rewrite input for this module
      other user-defined pre hooks
    """
    if annotation.rewrite_input:
        assert annotation.input is not None, \
            f"Cannot rewrite input without tensor annotation: {canon_mod_name}"
        module.register_forward_pre_hook(
            FwdPreRewriteHook(
                canon_mod_name,
                annotation.input,
                annotation,
                module,
                vp_rank,
            ),
            prepend=True,
        )

    if annotation.input is not None:
        module.register_forward_pre_hook(
            FwdPreLogHook(
                canon_mod_name,
                annotation.input,
                annotation,
                module,
                vp_rank,
            ),
            prepend=True,
        )

    """
    forward post hooks:
      other user-defined post hooks
      log output of this module
      rewrite output for next module
    """
    if annotation.output is not None:
        module.register_forward_hook(
            FwdPostLogHook(
                canon_mod_name,
                annotation.output,
                annotation,
                module,
                vp_rank,
            )
        )
    
    if annotation.rewrite_output:
        module.register_forward_hook(
            FwdPostRewriteHook(
                canon_mod_name,
                annotation.output,
                annotation,
                module,
                vp_rank,
            )
        )

    if annotation.grad is None: 
        return
    
    # backward hooks
    if annotation.grad.get("input", None):
        assert annotation.input is not None, \
            f"Cannot log input without tensor annotation: {canon_mod_name}"
        module.register_full_backward_hook(
            BwdPostLogHook(
                canon_mod_name,
                annotation.input,
                annotation,
                module,
                vp_rank,
            )
        )

    if annotation.rewrite_grad_output:
        assert annotation.output is not None, \
            f"Cannot rewrite output without tensor annotation: {canon_mod_name}"
        module.register_full_backward_pre_hook(
            BwdPreRewriteHook(
                canon_mod_name,
                annotation.output,
                annotation,
                module,
                vp_rank,
            ),
            prepend=True
        )

    if annotation.grad.get("output", None):
        assert annotation.output is not None, \
            f"Cannot log output without tensor annotation: {canon_mod_name}"
        module.register_full_backward_pre_hook(
            BwdPreLogHook(
                canon_mod_name,
                annotation.output,
                annotation,
                module,
                vp_rank,
            ),
            prepend=True
        )


class BatchCounterHook:
    """
    NOTE: Only the "model" with a vp_rank can have this hook, not the submodules
    """
    def __init__(self, vp_rank: int, is_fwd: bool):
        self.vp_rank = vp_rank
        self.is_fwd = is_fwd

    def __call__(self, module: torch.nn.Module, args: tuple):
        if self.is_fwd:
            fwd_pass_cnt[self.vp_rank] += 1
        else:
            bwd_pass_cnt[self.vp_rank] += 1


def regist_batch_cnt_hook(
        model: torch.nn.Module, 
        vp_rank: int, 
        is_fwd: bool,
    ) -> None:
    """
    Assumption: 
    Each time the model is being called, it consumes a new micro batch.
    """
    assert isinstance(model, torch.nn.Module), \
        "model must be an instance of torch.nn.Module"
    assert hasattr(model, "forward"), \
        "model must have a forward method"
    if is_fwd:
        model.register_forward_pre_hook(
            BatchCounterHook(vp_rank, is_fwd), 
            prepend=True
        )
    else:
        model.register_full_backward_pre_hook(
            BatchCounterHook(vp_rank, is_fwd), 
            prepend=True
        )


class ParamLogHook:
    def __init__(
        self,
        canon_param_name: str,
        vp_rank: int,
        shard_slice: ShardNDSlices,
        fw_canon_name: Optional[str] = None,
    ) -> None:
        self.canon_param_name = canon_param_name
        self.dp_rank = CONFIG.parallel.rank.dp_rank
        self.dp_size = CONFIG.parallel.size.dp_size
        self.vp_rank = vp_rank
        self.shard_slice = shard_slice
        self.fw_canon_name = fw_canon_name

    def _get_canon_param_grad_name(self) -> str:
        g_batch_id = canonical.calc_global_batch_id(
            bwd_pass_cnt[self.vp_rank],
            self.dp_size,
            self.dp_rank,
        )
        return f"iter.{iter_cnt}|batch.{g_batch_id}|{self.canon_param_name}"

    def __call__(
        self,
        grad: torch.Tensor,
    ) -> None:
        canonical_grad_name = self._get_canon_param_grad_name()
        tensor_info = TensorInfo(
            tensor=grad.detach().cpu(),
            canonical_name=canonical_grad_name,
            shard_slice=self.shard_slice,
            is_grad=True,
            extra = {"fw_canon_name": self.fw_canon_name} 
                    if self.fw_canon_name else {},
        )
        log_tensor(canonical_grad_name, TType.g_param, tensor_info)


__all__ = [
    #--- hook classes ---#
    "FwdPreLogHook",
    "FwdPreRewriteHook",
    "FwdPostLogHook",
    "FwdPostRewriteHook",
    "BwdPreLogHook",
    "BwdPreRewriteHook",
    "BwdPostLogHook",
    "BatchCounterHook",
    "ParamLogHook",
    #--- APIs ---#
    "update_iter_batch_cnt",
    "get_param_logging_cname",
    "register_hook",
    "regist_batch_cnt_hook",
]
