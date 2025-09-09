from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
from omegaconf import OmegaConf
from neck import CONFIG


# ../../../config/
_config_folder = Path(__file__).resolve().parent / "../../config"


@dataclass
class TensorAnnotation:
    """Annotation for tensors, can be used for parameters and activations.

    Attributes:
        suffix: The trailing part of the tensor name.  For example, the suffix
                of "module.module.decoder.layers.0.mlp.linear_fc1.weight" is
                "linear_fc1.weight".  Cannot be `None` for parameters, but can 
                be `None` for activations.
        tp_dim: The dimension of the tensor that is sharded. If `None`, the
                tensor is not sharded.
        full: The full shape of the tensor before sharding.
        slice: The slice shape of the tensor after sharding.

    Notes:
        - The full and slice fields are used for double checking. If it does not 
          match the actual tensor, it will raise an error.  
        - The full and slice shapes are essentially inferred from the runtime 
          shape and the tp_dim.
    """
    suffix: Optional[str] = None
    canonical_suffix: Optional[str] = None
    tp_dim: Optional[int] = None
    sp_dim: Optional[int] = None
    cp_dim: Optional[int] = None  # only for activations
    full: Optional[str] = None
    slice: Optional[str] = None

    @classmethod
    def from_dict(cls, raw_annotation: OmegaConf) -> "TensorAnnotation":
        annotation = cls()
        for field, value in raw_annotation.items():
            setattr(annotation, field, value)
        return annotation


@dataclass
class ModuleAnnotation:
    r"""Annotation for modules, containing tensor annotation for input / output.

    Attributes:
        suffix: The trailing part of the module name.  For example, the suffix
                of "module.module.decoder.layers.0.mlp" is "mlp".  For
                "module.module.decoder.layers.0", the suffix is r"layers.\d".
        input: The tensor annotation for the input tensor.
        output: The tensor annotation for the output tensor.
        input_idx: The index of the input tensor in the module's input list.
        output_idx: The index of the output tensor in the module's output list.
        rewrite_input: Rewrite the input tensor with consistent value or not.
        rewrite_output: Rewrite the output tensor with consistent value or not.
    """
    turnoff: bool = False
    suffix: Optional[str] = None
    input: Optional[TensorAnnotation] = None
    output: Optional[TensorAnnotation] = None
    input_idx: Optional[int] = None
    output_idx: Optional[int] = None
    rewrite_input: bool = False
    rewrite_output: bool = False
    rewrite_grad_output: bool = False
    grad: Optional[OmegaConf] = None
    fusion: Optional[list[str]] = None

    @classmethod
    def from_dict(cls, raw_annotation: OmegaConf) -> "ModuleAnnotation":
        annotation = cls()
        for field, value in raw_annotation.items():
            if field in ("input", "output"):
                value = TensorAnnotation.from_dict(value)
            setattr(annotation, field, value)
        if CONFIG.neck.get("rewrite", False) is False:
            annotation.rewrite_input = False
            annotation.rewrite_output = False
            annotation.rewrite_grad_output = False
        return annotation


def load_annotations(a_type: str) -> OmegaConf | None:
    assert CONFIG is not None, "Please set NECK_CONFIG_PATH"
    assert a_type in CONFIG.annotations, f"Annotation type {a_type} is not supported."
    annotation_file = CONFIG.annotations[a_type]
    if annotation_file is None:
        return None
    if not Path(annotation_file).is_absolute():
        annotation_file = _config_folder / annotation_file
    with open(annotation_file) as f:
        annotations = OmegaConf.load(f)
    return annotations


def _parse_param_annotation() -> dict[str, TensorAnnotation] | None:
    """
    None means no annotation file is provided. 
    Param annotations are not needed for inference, so it can be None.
    """
    param_annotations = {}
    raw_annotations = load_annotations("param")
    if raw_annotations is None:
        return None
    for suffix, raw_annotation in raw_annotations.items():
        annotation = TensorAnnotation.from_dict(raw_annotation)
        annotation.suffix = suffix
        param_annotations[suffix] = annotation
    return param_annotations
    

def _parse_module_annotation() -> dict[str, ModuleAnnotation]:
    module_annotations = {}
    raw_annotations = load_annotations("module")
    for suffix, raw_annotation in raw_annotations.items():
        annotation = ModuleAnnotation.from_dict(raw_annotation)
        annotation.suffix = suffix
        module_annotations[suffix] = annotation
    return module_annotations


param_annotations = _parse_param_annotation()
module_annotations = _parse_module_annotation()


def to_general_name(local_name: str) -> str:
    r"""
    Convert a local name to a general name by replacing digits with \d.
    e.g., module.module.decoder.layers.\d.self_attention
    """
    name_decomp = local_name.split(".")
    for i, part in enumerate(name_decomp):
        if part.isdigit():
            name_decomp[i] = r"\d"
            break
    return ".".join(name_decomp)


def get_annotation(
    local_name: str,
    type: Literal["param", "module"],
) -> Optional[TensorAnnotation | ModuleAnnotation]:
    """Get the annotation given a module or parameter's local name.

    Args:
        local_name: The name of the module or parameter from named_x().
        type: The type of the annotation, either "param" or "module".

    Returns:
        The annotation for the module or parameter, or `None` if not found.
    """

    if type == "param":
        annotations = param_annotations
    elif type == "module":
        annotations = module_annotations
    else:
        raise ValueError(f"Annotation type {type} is not supported.")

    general_name = to_general_name(local_name)
    for suffix in annotations:
        if general_name.endswith(suffix):
            return annotations[suffix]
    return None


__all__ = [
    "TensorAnnotation",
    "ModuleAnnotation",
    "param_annotations",
    "module_annotations",
    "to_general_name",
    "get_annotation",
]
