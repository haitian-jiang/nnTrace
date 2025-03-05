import torch
from collections import defaultdict
from typing import Literal
from neck.structs import (
    TensorAnnotation, 
    ModuleAnnotation, 
    TensorInfo, 
    ShardSlice,
    TType,
)
from .logger import log_tensor
from neck import CONFIG
import neck.canonical as canonical
import neck.random


_fwd_pass_cnt = defaultdict(lambda: -1)  # Counting from 0
_bwd_pass_cnt = defaultdict(lambda: -1)


class ModuleHookBase:
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
            called_times = _bwd_pass_cnt[self.vp_rank]
        else:
            t_type = TType.activ
            called_times = _fwd_pass_cnt[self.vp_rank]

        g_batch_id = canonical.calc_global_batch_id(
            called_times,
            self.dp_size,
            self.dp_rank,
        )

        return f"batch.{g_batch_id}|{self.canon_mod_name}|{t_type}.{i_o_}.{idx}"

    def get_tensor_info(
            self, 
            activation: torch.Tensor,
            canonical_name: str,
            is_grad: bool = False,
        ) -> TensorInfo:
        full_shape, slices = canonical.infer_tp_slice(activation.shape, self.t_annotation)
        # NOTE: no double checking of shape and slice here, could add if exists in annotation.
        return TensorInfo(
            tensor=activation.detach().cpu(),
            canonical_name=canonical_name,
            shape=ShardSlice(slices, full_shape),
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
            self.m_annotation.input_idx = 0
        if self.m_annotation.input_idx is not None:  # only one input
            idx = self.m_annotation.input_idx
            activation = args[idx]
            assert isinstance(activation, torch.Tensor), \
                "fwd_pre_log expects a tensor"
            canon_act_name = self.get_canon_act_grad_name("input", idx)
            tensor_info = self.get_tensor_info(activation, canon_act_name)
            log_tensor(canon_act_name, TType.activ, tensor_info)
        else:
            for idx, activation in enumerate(args):
                if not isinstance(activation, torch.Tensor):
                    continue
                canon_act_name = self.get_canon_act_grad_name("input", idx)
                tensor_info = self.get_tensor_info(activation, canon_act_name)
                log_tensor(canon_act_name, TType.activ, tensor_info)


class FwdPreRewriteHook(ModuleHookBase):
    """Rewrite all tensors in the input of the module.
    """
    def _rewrite_input(self, tensor: torch.Tensor, idx: int) -> None:
        assert isinstance(tensor, torch.Tensor), "fwd_pre_rewrite expects a tensor"
        canon_act_name = self.get_canon_act_grad_name("input", idx)
        full_shape, slices = canonical.infer_tp_slice(tensor.shape, self.t_annotation)
        new_activation = neck.random.rand(canon_act_name, full_shape, slices)
        new_activation = new_activation -0.5  # rescale to [-1/2, 1/2]
        # NOTE: 
        # Must directly rewrite the tensor.data.  Return the new tensor will 
        # break the computation graph by losing the `grad_fn` attribute.
        tensor.data = new_activation.data.to(tensor.dtype).to(tensor.device)

    def __call__(
            self, 
            module: torch.nn.Module, 
            args: tuple,
        ) -> None:
        assert module is self.module, \
            f"module mismatch in {self.canon_mod_name} pre rewrite"
        if len(args) == 1:
            self.m_annotation.input_idx = 0
        if self.m_annotation.input_idx is not None:  # only one input
            idx = self.m_annotation.input_idx
            self._rewrite_input(args[idx], idx)

        for idx, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                continue
            # TODO: inputs use the same t_annotation, but they may not be sharded in the same way.
            self._rewrite_input(arg, idx)
    

class FwdPostLogHook(ModuleHookBase):
    def _log_output(self, activation: torch.Tensor, idx: int = 0) -> None:
        canon_act_name = self.get_canon_act_grad_name("output", idx)
        tensor_info = self.get_tensor_info(activation, canon_act_name)
        log_tensor(canon_act_name, TType.activ, tensor_info)

    def __call__(
            self, 
            module: torch.nn.Module, 
            args: tuple, 
            output: torch.Tensor | tuple,  # python >= 3.10
        ) -> None:

        # checkings
        if self.module is not None:
            assert module == self.module, f"module mismatch in {self.canon_mod_name} post log"
        assert isinstance(output, torch.Tensor) or isinstance(output, tuple), \
            "fwd_post_log expects a tensor or a tuple of tensors"
        
        if isinstance(output, torch.Tensor):
            assert self.m_annotation.output_idx is None
            self._log_output(output)
        elif self.m_annotation.output_idx is not None:
            idx = self.m_annotation.output_idx
            activation = output[idx]
            assert isinstance(activation, torch.Tensor), "fwd_post_log expects a tensor"
            self._log_output(activation, idx)
        else:
            for idx, activation in enumerate(output):
                if isinstance(activation, torch.Tensor):
                    self._log_output(activation, idx)


class BwdPreLogHook(ModuleHookBase):
    """
    Use it only because need to log the `grad_input` (result) of the module 
    closer to loss, so must be executed before the rewrite hook.
    Use it only when necessary.
    """
    def __call__(
        self, 
        module: torch.nn.Module,
        grad_output: tuple[torch.Tensor],
    ) -> None:
        for idx, grad in enumerate(grad_output):
            if grad is None:
                continue
            assert isinstance(grad, torch.Tensor), f"bwd pre hook get non tensor: {type(grad)}"
            canon_grad_name = self.get_canon_act_grad_name("output", idx, 
                                                           is_grad=True)
            tensor_info = self.get_tensor_info(grad, canon_grad_name, True)
            log_tensor(canon_grad_name, TType.g_activ, tensor_info)


class BwdPreRewriteHook(ModuleHookBase):
    """
    There should not be any weight gradient, so can safely rewrite the 
    grad_output.  Should always use this to rewrite.
    """
    def __call__(
        self,
        module: torch.nn.Module,
        grad_output: tuple[torch.Tensor],
    ) -> None:
        for idx, grad in enumerate(grad_output):
            if grad is None:
                continue
            assert isinstance(grad, torch.Tensor), f"bwd pre hook get non tensor: {type(grad)}"
            canon_grad_name = self.get_canon_act_grad_name("output", idx, 
                                                           is_grad=True)
            full_shape, slices = canonical.infer_tp_slice(grad.shape, 
                                                          self.t_annotation)
            new_grad = neck.random.rand(canon_grad_name, full_shape, slices)
            new_grad = new_grad - 0.5  # rescale to [-1/2, 1/2]
            grad.data = new_grad.data.to(grad.dtype).to(grad.device)


class BwdPostLogHook(ModuleHookBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_fwd = False

    def __call__(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor],
        grad_output: tuple[torch.Tensor],
    ) -> None:
        for idx, grad in enumerate(grad_input):
            if grad is None:
                continue
            assert isinstance(grad, torch.Tensor), f"bwd post hook get non tensor {type(grad)}"
            canon_grad_name = self.get_canon_act_grad_name("input", idx, 
                                                           is_grad=True)
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
        module.register_full_pre_backward_hook(
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
            _fwd_pass_cnt[self.vp_rank] += 1
        else:
            _bwd_pass_cnt[self.vp_rank] += 1


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
        full_shape: tuple[int, ...],
        tp_slice: tuple[slice, ...],
    ) -> None:
        self.canon_param_name = canon_param_name
        self.dp_rank = CONFIG.parallel.rank.dp_rank
        self.dp_size = CONFIG.parallel.size.dp_size
        self.vp_rank = vp_rank
        self.full_shape = full_shape
        self.tp_slice = tp_slice

    def _get_canon_param_grad_name(self) -> str:
        g_batch_id = canonical.calc_global_batch_id(
            _bwd_pass_cnt[self.vp_rank],
            self.dp_size,
            self.dp_rank,
        )
        return f"batch.{g_batch_id}|{self.canon_param_name}"

    def __call__(
        self,
        grad: torch.Tensor,
    ) -> None:
        canonical_grad_name = self._get_canon_param_grad_name()
        tensor_info = TensorInfo(
            tensor=grad.detach().cpu(),
            canonical_name=canonical_grad_name,
            shape=ShardSlice(self.tp_slice, self.full_shape),
            is_grad=True,
        )
        log_tensor(canonical_grad_name, TType.g_param, tensor_info)
