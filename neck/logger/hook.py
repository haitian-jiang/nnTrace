import torch
from typing import Tuple, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from neck.interface import TensorAnnotation, ModuleAnnotation
from .logger import log_tensor, TensorInfo
import neck.canonical as canonical
import neck.random


class HookBase:
    def __init__(
            self, 
            canon_mod_name: str, 
            t_annotation: "TensorAnnotation",
            m_annotation: "ModuleAnnotation",
            module: Optional[torch.nn.Module] = None,
        ):
        self.canon_mod_name = canon_mod_name  # canonical module name
        self.t_annotation = t_annotation
        self.m_annotation = m_annotation  # to access output_idx
        self.module = module


class FwdHookBase(HookBase):
    def get_canon_act_name(self, is_module_output=True) -> str:
        # TODO: add global batch index
        # pre hook: input, post hook: output
        activation_suffix = "output" if is_module_output else "input"
        return f"batch|{self.canon_mod_name}|{activation_suffix}"

    def get_tensor_info(self, activation: torch.Tensor) -> TensorInfo:
        full_shape, slices = canonical.infer_tp_slice(activation.shape, self.t_annotation)
        # NOTE: no double checking of shape and slice here, could add if exists in annotation.
        return TensorInfo(
            tensor=activation.cpu(),
            canonical_name=self.get_canon_act_name(),
            full_shape=full_shape,
            shard_slice=slices,
        )


class FwdPreLogHook(FwdHookBase):
    def __call__(
            self, 
            module: torch.nn.Module, 
            args: Tuple[torch.Tensor],
        ) -> None:
        """
        Log the input (actually output of last module) activation.
        The input (`args`) should be a tuple of one single tensor.
        Dynamically construct the canonical name from module name,
        input / output, and global batch index.
        Get shape and slice from tensor annotation stored during construction.
        """
        assert len(args) == 1, "fwd_pre_log expects one argument"
        assert isinstance(args[0], torch.Tensor), "fwd_pre_log expects a tensor"
        if self.module is not None:
            assert module is self.module, f"module mismatch in {self.canon_mod_name} pre log"

        canon_act_name = self.get_canon_act_name(is_module_output=False)
        tensor_info = self.get_tensor_info(args[0])
        log_tensor(canon_act_name, "activ", tensor_info)


class FwdPreRewriteHook(HookBase):
    def __call__(
            self, 
            module: torch.nn.Module, 
            args: Tuple,
        ) -> None:
        assert len(args) == 1, "fwd_pre_rewrite expects one argument"
        assert isinstance(args[0], torch.Tensor), "fwd_pre_rewrite expects a tensor"
        if self.module is not None:
            assert module is self.module, f"module mismatch in {self.canon_mod_name} pre rewrite"
        
        canon_act_name = self.get_canon_act_name(is_module_output=False)
        full_shape, slices = canonical.infer_tp_slice(args[0].shape, self.t_annotation)
        new_activation = neck.random.rand(canon_act_name, full_shape, slices)
        
        # NOTE: 
        # Must directly rewrite the tensor.data.  Return the new tensor will 
        # break the computation graph by losing the `grad_fn` attribute.
        # We have to hack/keep track of the computation graph ourselves:
        # Whenever the tensor is rewritten in the forward pass, we must also
        # rewrite the backward one.  # TODO: verify this.
        args[0].data = new_activation.data.to(args[0].dtype).to(args[0].device)
    

class FwdPostLogHook(FwdHookBase):
    def __call__(
            self, 
            module: torch.nn.Module, 
            args: Tuple, 
            output: torch.Tensor | Tuple,  # python >= 3.10
        ) -> None:

        # checkings
        if self.module is not None:
            assert module == self.module, f"module mismatch in {self.canon_mod_name} post log"
        assert isinstance(output, torch.Tensor) or isinstance(output, tuple), \
            "fwd_post_log expects a tensor or a tuple of tensors"
        if isinstance(output, torch.Tensor):
            assert self.m_annotation.output_idx is None
            activation = output
        else:
            assert self.m_annotation.output_idx is not None
            activation = output[self.m_annotation.output_idx]
            assert isinstance(activation, torch.Tensor), "fwd_post_log expects a tensor"
        
        # log the output activation
        canon_act_name = self.get_canon_act_name(is_module_output=True)
        tensor_info = self.get_tensor_info(activation)
        log_tensor(canon_act_name, "activ", tensor_info)


class FwdPostRewriteHook(HookBase):
    def __call__(
            self, 
            module: torch.nn.Module, 
            args: Tuple, 
            output: torch.Tensor | Tuple,  # python >= 3.10
        ) -> None:
        # checkings
        if self.module is not None:
            assert module == self.module, f"module mismatch in {self.canon_mod_name} post rewrite"
        assert isinstance(output, torch.Tensor) or isinstance(output, tuple), \
            "fwd_post_rewrite expects a tensor or a tuple of tensors"
        if isinstance(output, torch.Tensor):
            assert self.m_annotation.output_idx is None
            activation = output
        else:
            assert self.m_annotation.output_idx is not None
            activation = output[self.m_annotation.output_idx]
            assert isinstance(activation, torch.Tensor), "fwd_post_rewrite expects a tensor"
        
        # rewrite the output activation
        canon_act_name = self.get_canon_act_name(is_module_output=True)
        full_shape, slices = canonical.infer_tp_slice(activation.shape, self.t_annotation)
        new_activation = neck.random.rand(canon_act_name, full_shape, slices)
        activation.data = new_activation.data.to(activation.dtype).to(activation.device)


def bwd_pre_log_hook(
        module: torch.nn.Module, 
        grad_output: Tuple[torch.Tensor],
    ) -> None:
    pass


def bwd_pre_rewrite_hook(
        module: torch.nn.Module, 
        grad_output: Tuple[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
    pass


def bwd_post_log_hook(
        module: torch.nn.Module, 
        grad_input: Tuple[torch.Tensor],
        grad_output: Tuple[torch.Tensor],
    ) -> None:
    pass


def bwd_post_rewrite_hook(
        module: torch.nn.Module, 
        grad_input: Tuple[torch.Tensor],
        grad_output: Tuple[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
    pass


def register_hook(
        module: torch.nn.Module,
        canon_mod_name: str,
        annotation: "ModuleAnnotation",
    ) -> None:
    # TODO: only fwd hook now, later add bwd hook.

    assert isinstance(module, torch.nn.Module), \
        "module must be an instance of torch.nn.Module"
    assert hasattr(module, "forward"), \
        "module must have a forward method"
    
    # forward pre hooks: 
    #   log output from last module 
    #   rewrite input for this module
    #   other user-defined pre hooks
    if annotation.rewrite_input:
        assert annotation.input is not None, \
            f"Cannot rewrite input without tensor annotation: {canon_mod_name}"
        module.register_forward_pre_hook(
            FwdPreRewriteHook(
                canon_mod_name,
                annotation.input,
                annotation,
                module,
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
            ),
            prepend=True,
        )

    # forward post hooks:
    #  other user-defined post hooks
    #  log output of this module
    #  rewrite output for next module
    if annotation.output is not None:
        module.register_forward_hook(
            FwdPostLogHook(
                canon_mod_name,
                annotation.output,
                annotation,
                module,
            ),
        )
    
    if annotation.rewrite_output:
        assert annotation.output is not None, \
            f"Cannot rewrite output without tensor annotation: {canon_mod_name}"
        module.register_forward_hook(
            FwdPostRewriteHook(
                canon_mod_name,
                annotation.output,
                annotation,
                module,
            ),
        )