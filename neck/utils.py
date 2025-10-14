import torch
import torch.distributed.tensor as dtensor


TENSOR_TYPE = torch.Tensor | dtensor.DTensor


def get_local_shape(tensor: TENSOR_TYPE) -> torch.Size:
    if isinstance(tensor, dtensor.DTensor):
        return tensor._local_tensor.shape
    elif isinstance(tensor, torch.Tensor):
        return tensor.shape
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")
    

def rewrite_tensor_data(dst: TENSOR_TYPE, src: torch.Tensor) -> None:
    assert isinstance(src, torch.Tensor), "New value must be a torch.Tensor"
    
    if not dst.device == src.device:
        src = src.to(dst.device)

    if isinstance(dst, torch.Tensor):
        assert dst.shape == src.shape, f"Shape mismatch: {dst.shape} vs {src.shape}"
        dst.data.copy_(src.data)
    elif isinstance(dst, dtensor.DTensor):
        local_dst = dst._local_tensor
        assert local_dst.shape == src.shape, \
            f"Shape mismatch: {local_dst.shape} vs {src.shape}"
        local_dst.data.copy_(src.data)
    else:
        raise ValueError(f"Unsupported tensor type: {type(dst)}")