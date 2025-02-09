from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, Union, Dict
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
        tp_dim: The dimension of the tensor that is sharded.
        full: The full shape of the tensor before sharding.
        slice: The slice shape of the tensor after sharding.

    Notes:
        - The full and slice fields are used for double checking. If it does not 
          match the actual tensor, it will raise an error.  
        - The full and slice shapes are essentially inferred from the runtime 
          shape and the tp_dim.
    """
    suffix: Optional[str] = None
    tp_dim: Optional[int] = None
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
    """Annotation for modules, containing tensor annotation for input / output.

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
    suffix: Optional[str] = None
    input: Optional[TensorAnnotation] = None
    output: Optional[TensorAnnotation] = None
    input_idx: Optional[int] = None
    output_idx: Optional[int] = None
    rewrite_input: Optional[bool] = False
    rewrite_output: Optional[bool] = False

    @classmethod
    def from_dict(cls, raw_annotation: OmegaConf) -> "ModuleAnnotation":
        annotation = cls()
        for field, value in raw_annotation.items():
            if field in ("input", "output"):
                value = TensorAnnotation.from_dict(value)
            setattr(annotation, field, value)
        return annotation


def load_annotations(a_type: str) -> OmegaConf:
    assert a_type in CONFIG.annotations, f"Annotation type {a_type} is not supported."
    annotation_file = CONFIG.annotations[a_type]
    if not Path(annotation_file).is_absolute():
        annotation_file = _config_folder / annotation_file
    with open(annotation_file) as f:
        annotations = OmegaConf.load(f)
    return annotations



def _parse_param_annotation() -> Dict[str, TensorAnnotation]:
    param_annotations = {}
    raw_annotations = load_annotations("param")
    for suffix, raw_annotation in raw_annotations.items():
        annotation = TensorAnnotation.from_dict(raw_annotation)
        annotation.suffix = suffix
        param_annotations[suffix] = annotation
    return param_annotations
    

def _parse_module_annotation() -> Dict[str, ModuleAnnotation]:
    module_annotations = {}
    raw_annotations = load_annotations("module")
    for suffix, raw_annotation in raw_annotations.items():
        annotation = ModuleAnnotation.from_dict(raw_annotation)
        annotation.suffix = suffix
        module_annotations[suffix] = annotation
    return module_annotations


param_annotations = _parse_param_annotation()
module_annotations = _parse_module_annotation()


__all__ = [
    "TensorAnnotation",
    "ModuleAnnotation",
    "param_annotations",
    "module_annotations"
]
