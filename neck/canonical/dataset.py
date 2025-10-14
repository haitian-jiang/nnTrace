import torch
from .data_parallel import calc_global_batch_id
from neck import CONFIG
from neck.utils import get_local_shape, rewrite_tensor_data
from neck.logger.torch_hook import iter_cnt
from neck.structs import ShardNDSlices
import neck.random


class LMLocalMiniBatchRewriter:
    """
    Provide the concatenated minibatch (composed of pipeline microbatches) 
    local to each dp rank.
    """
    def __init__(self):
        self.num_microbatches = CONFIG.parallel.num_microbatches
        self.dp_rank = CONFIG.parallel.rank.dp_rank
        self.dp_size = CONFIG.parallel.size.dp_size
        self.vocab_size = CONFIG.architecture.vocab_size
        self.micro_batch_size = None

    def generate_consistent_data(self, seq_len: int):
        shape = (self.micro_batch_size, seq_len)
        shard_slice = ShardNDSlices(shape)
        x_micro_batches = []
        y_micro_batches = []
        for mbid in range(self.num_microbatches):
            gbid = calc_global_batch_id(mbid, self.dp_size, self.dp_rank)
            canonical_prefix = f"iter.{iter_cnt}|batch.{gbid}|data"
            x_micro_batches.append(
                neck.random.randint(
                    0, self.vocab_size,
                    f"{canonical_prefix}.x",
                    shard_slice,
                )
            )
            y_micro_batches.append(
                neck.random.randint(
                    0, self.vocab_size,
                    f"{canonical_prefix}.y",
                    shard_slice,
                )
            )
        new_x = torch.cat(x_micro_batches, dim=0).contiguous()
        new_y = torch.cat(y_micro_batches, dim=0).contiguous()
        return new_x, new_y

    def rewrite_data(self, x: torch.Tensor, y: torch.Tensor) -> None:
        assert x.shape == y.shape, "x and y must have the same shape"
        batch_size, seq_len = get_local_shape(x)
        micro_batch_size = batch_size // self.num_microbatches
        assert micro_batch_size * self.num_microbatches == batch_size
        self.micro_batch_size = micro_batch_size
        new_x, new_y = self.generate_consistent_data(seq_len)
        rewrite_tensor_data(x, new_x)
        rewrite_tensor_data(y, new_y)

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        self.rewrite_data(x, y)
