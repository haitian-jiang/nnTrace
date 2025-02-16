import torch
from collections import defaultdict
from typing import Tuple, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from neck.interface import TensorAnnotation, ModuleAnnotation
    from omegaconf import OmegaConf
from .logger import log_tensor, TensorInfo
from neck import CONFIG
import neck.canonical as canonical
import neck.random


_fwd_pass_cnt = defaultdict(lambda: -1)  # Counting from 0
_bwd_pass_cnt = defaultdict(lambda: -1)


class HookBase:
    def __init__(
            self, 
            canon_mod_name: str, 
            t_annotation: "TensorAnnotation",
            m_annotation: "ModuleAnnotation",
            module: torch.nn.Module,
            neck_config: "OmegaConf",
            vp_rank: int,
            is_fwd: bool = True,
        ):
        self.canon_mod_name = canon_mod_name  # canonical module name
        self.t_annotation = t_annotation
        self.m_annotation = m_annotation  # to access output_idx
        self.module = module
        self.neck_config = neck_config
        self.dp_rank = CONFIG.parallel.rank.dp_rank
        self.dp_size = CONFIG.parallel.size.dp_size
        self.vp_rank = vp_rank

    def get_canon_act_grad_name(
            self, 
            is_output: bool = True, 
            idx: int | str = 0,
            is_grad: bool = False,
        ) -> str:
        # fwd pre hook: input, fwd post hook: output
        i_o_ = "output" if is_output else "input"
        if is_grad:
            i_o_ = f"g_{i_o_}"
            called_times = _bwd_pass_cnt[self.vp_rank]
        else:
            called_times = _fwd_pass_cnt[self.vp_rank]

        # NOTE: hardcoded for Megatron-LM, can read the relation from config
        local_batch_id = called_times  # PP schedule
        global_batch_id = local_batch_id * self.dp_size + self.dp_rank  # DP

        return f"batch.{global_batch_id}|{self.canon_mod_name}|{i_o_}.{idx}"

    def get_tensor_info(
            self, 
            activation: torch.Tensor,
            canonical_name: str,
        ) -> TensorInfo:
        full_shape, slices = canonical.infer_tp_slice(activation.shape, self.t_annotation)
        # NOTE: no double checking of shape and slice here, could add if exists in annotation.
        return TensorInfo(
            tensor=activation.detach().cpu(),
            canonical_name=canonical_name,
            full_shape=full_shape,
            shard_slice=slices,
            module_class=self.module.__class__.__name__,
        )


class FwdPreLogHook(HookBase):
    # Must be prepend to a module at last, because it need to log out the output
    # of the last module, which must NOT been already rewritten.
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
        assert module is self.module, \
            f"module mismatch in {self.canon_mod_name} pre log"
        if len(args) == 1:
            self.m_annotation.input_idx = 0
        if self.m_annotation.input_idx is not None:  # only one input
            activation = args[self.m_annotation.input_idx]
            assert isinstance(activation, torch.Tensor), \
                "fwd_pre_log expects a tensor"
            canon_act_name = self.get_canon_act_grad_name(False)
            tensor_info = self.get_tensor_info(activation, canon_act_name)
            log_tensor(canon_act_name, "activ", tensor_info)
        else:
            for idx, activation in enumerate(args):
                if not isinstance(activation, torch.Tensor):
                    continue
                canon_act_name = self.get_canon_act_grad_name(False, idx)
                tensor_info = self.get_tensor_info(activation, canon_act_name)
                log_tensor(canon_act_name, "activ", tensor_info)


class FwdPreRewriteHook(HookBase):
    """Rewrite all tensors in the input of the module.
    """
    def __call__(
            self, 
            module: torch.nn.Module, 
            args: Tuple,
        ) -> None:
        assert module is self.module, \
            f"module mismatch in {self.canon_mod_name} pre rewrite"
        for idx, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                continue
            canon_act_name = self.get_canon_act_grad_name(False, idx)
            # TODO: weight and bias use the same t_annotation, but they may not be sharded in the same way.
            full_shape, slices = canonical.infer_tp_slice(arg.shape, self.t_annotation)
            new_activation = neck.random.rand(canon_act_name, full_shape, slices)
            # NOTE: 
            # Must directly rewrite the tensor.data.  Return the new tensor will 
            # break the computation graph by losing the `grad_fn` attribute.
            arg.data = new_activation.data.to(arg.dtype).to(arg.device)
    

class FwdPostLogHook(HookBase):
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
            canon_act_name = self.get_canon_act_grad_name(True)
            tensor_info = self.get_tensor_info(activation, canon_act_name)
            log_tensor(canon_act_name, "activ", tensor_info)
        elif self.m_annotation.output_idx is not None:
            activation = output[self.m_annotation.output_idx]
            assert isinstance(activation, torch.Tensor), "fwd_post_log expects a tensor"
            canon_act_name = self.get_canon_act_grad_name(True, self.m_annotation.output_idx)
            tensor_info = self.get_tensor_info(activation, canon_act_name)
            log_tensor(canon_act_name, "activ", tensor_info)
        else:
            for idx, activation in enumerate(output):
                if not isinstance(activation, torch.Tensor):
                    continue
                canon_act_name = self.get_canon_act_grad_name(True, idx)
                tensor_info = self.get_tensor_info(activation, canon_act_name)
                log_tensor(canon_act_name, "activ", tensor_info)


# class FwdPostRewriteHook(HookBase):
#     # NOTE: I don't think this function should ever be used.
#     def __call__(
#             self, 
#             module: torch.nn.Module, 
#             args: Tuple, 
#             output: torch.Tensor | Tuple,  # python >= 3.10
#         ) -> None:
#         # checkings
#         if self.module is not None:
#             assert module == self.module, f"module mismatch in {self.canon_mod_name} post rewrite"
#         assert isinstance(output, torch.Tensor) or isinstance(output, tuple), \
#             "fwd_post_rewrite expects a tensor or a tuple of tensors"
#         if isinstance(output, torch.Tensor):
#             assert self.m_annotation.output_idx is None
#             activation = output
#         else:
#             assert self.m_annotation.output_idx is not None
#             activation = output[self.m_annotation.output_idx]
#             assert isinstance(activation, torch.Tensor), "fwd_post_rewrite expects a tensor"
        
#         # rewrite the output activation
#         canon_act_name = self.get_canon_act_grad_name(True)
#         full_shape, slices = canonical.infer_tp_slice(activation.shape, self.t_annotation)
#         new_activation = neck.random.rand(canon_act_name, full_shape, slices)
#         activation.data = new_activation.data.to(activation.dtype).to(activation.device)


class BwdPreLogHook(HookBase):
    """use it only because need to log the `grad_input` of the module near loss,
    so must be executed before the rewrite hook.  Use it only when necessary.
    """
    def __call__(
        self, 
        module: torch.nn.Module,
        grad_output: Tuple[torch.Tensor],
    ) -> None:
        for idx, grad in enumerate(grad_output):
            if not isinstance(grad, torch.Tensor):
                continue
            canon_grad_name = self.get_canon_act_grad_name(True, idx)
            tensor_info = self.get_tensor_info(grad, canon_grad_name)
            log_tensor(canon_grad_name, "g_activ", tensor_info)


class BwdPreRewriteHook(HookBase):
    """
    There should not be any weight gradient, so can safely rewrite the 
    grad_output.  Should always use this to rewrite.
    """
    def __call__(
        self,
        module: torch.nn.Module,
        grad_output: Tuple[torch.Tensor],
    ) -> None:
        for idx, grad in enumerate(grad_output):
            assert isinstance(grad, torch.Tensor), "bwd pre hook get non tensor"
            canon_grad_name = self.get_canon_act_grad_name(True, idx, is_grad=True)
            full_shape, slices = canonical.infer_tp_slice(grad.shape, self.t_annotation)
            new_grad = neck.random.rand(canon_grad_name, full_shape, slices)
            grad.data = new_grad.data.to(grad.dtype).to(grad.device)


class BwdPostLogHook(HookBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_fwd = False

    def __call__(
        self,
        module: torch.nn.Module,
        grad_input: Tuple[torch.Tensor],
        grad_output: Tuple[torch.Tensor],
    ) -> None:
        for idx, grad in enumerate(grad_input):
            if not isinstance(grad, torch.Tensor):
                continue
            canon_grad_name = self.get_canon_act_grad_name(False, idx, is_grad=True)
            tensor_info = self.get_tensor_info(grad, canon_grad_name)
            log_tensor(canon_grad_name, "g_activ", tensor_info)


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
                CONFIG,
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
                CONFIG,
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
                CONFIG,
                vp_rank,
            )
        )

    if annotation.grad is None: 
        return
    
    # backward hooks
    if annotation.grad.get("input", None):
        module.register_full_backward_hook(
            BwdPostLogHook(
                canon_mod_name,
                annotation.input,
                annotation,
                module,
                CONFIG,
                vp_rank,
            )
        )

    if annotation.rewrite_input:
        module.register_full_pre_backward_hook(
            BwdPreRewriteHook(
                canon_mod_name,
                annotation.input,
                annotation,
                module,
                CONFIG,
                vp_rank,
            ),
            prepend=True
        )

    if annotation.grad.get("output", None):
        module.register_full_backward_pre_hook(
            BwdPreLogHook(
                canon_mod_name,
                annotation.output,
                annotation,
                module,
                CONFIG,
                vp_rank,
            ),
            prepend=True
        )
    



class BatchCounterHook:
    # NOTE: only the "model" with a vp_rank will have this hook, not the submodules

    def __init__(self, vp_rank: int, is_fwd: bool):
        self.vp_rank = vp_rank
        self.is_fwd = is_fwd

    def __call__(self, module: torch.nn.Module, args: Tuple):
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
