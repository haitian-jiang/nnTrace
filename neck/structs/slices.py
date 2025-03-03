from typing import Sequence, Optional, Union
import torch


class ShardSlice:
    """
    A class to represent a the slice of a sharded tensor in its full tensor.

    Attributes:
        shard_slice: Tuple[slice, ...]: The slice of the sharded tensor.
        full_shape: Tuple[int, ...]: Full shape of the tensor before sharding.
                    Used for adding two slices together.  Also don't need to 
                    store the full shape again in TensorInfo.
    """
    def __init__(
            self, 
            shard_slice: Sequence[slice] | None, 
            full_shape: Sequence[int]
        ):
        """
        Args:
            shard_slice: The slice of the sharded tensor.  If None, return the
                         full tensor slice.
            full_shape: Full shape of the tensor before sharding.
        """
        if shard_slice is None:
            shard_slice = [slice(0, dim) for dim in full_shape]
        assert len(shard_slice) == len(full_shape)
        for s, dim in zip(shard_slice, full_shape):
            assert 0 <= s.start < s.stop <= dim
        self.shard_slice = tuple(shard_slice)
        self.full_shape = tuple(full_shape)
    
    def __repr__(self):
        # [0:4096, 2048:4096]
        slc = f"[{', '.join(f'{s.start}:{s.stop}' for s in self.shard_slice)}]"
        shp = str(self.full_shape).replace(" ", "")
        return f"{slc}@{shp}"
    
    def __eq__(self, other):
        return (
            self.shard_slice == other.shard_slice and 
            self.full_shape == other.full_shape
        )
    
    def __len__(self):
        return len(self.shard_slice)
    
    def __getitem__(self, idx):
        return self.shard_slice[idx]
    
    def __iter__(self):
        return iter(self.shard_slice)
    
    def __hash__(self):
        return hash(
            (tuple((s.start, s.stop) for s in self.shard_slice), 
             self.full_shape)
        )
    
    def __lt__(self, other: "ShardSlice") -> bool:
        assert not self.has_overlap(other), f"{self} {other}"
        for s1, s2 in zip(self.shard_slice, other.shard_slice):
            if s1.start < s2.start:
                return True
            elif s1.start > s2.start:
                return False
            else:
                assert s1.stop == s2.stop, "Wrongly sharded."
        return False
    
    def __gt__(self, other: "ShardSlice") -> bool:
        return other < self
    
    def __add__(self, other: "ShardSlice") -> "ShardSlice":
        """
        Add two ShardSlices together. 
        """
        mergeable, dim = self.mergeable(other)
        assert mergeable, f"The two ShardSlices can't be merged {self} {other}"
        new_ = list(self.shard_slice)
        if self < other:
            new_[dim] = slice(self[dim].start, other[dim].stop)
        else:
            new_[dim] = slice(other[dim].start, self[dim].stop)
        return ShardSlice(new_, self.full_shape)

    def __radd__(self, other: Union["ShardSlice", int]) -> "ShardSlice":
        """
        For sum() support, as the default initial value in sum() is 0.
        """
        if isinstance(other, int):
            assert other == 0, "Can only add 0 to a ShardSlice in sum()."
            return self
        return self.__add__(other)

    def mergeable(self, other: "ShardSlice") -> tuple[bool, Optional[int]]:
        """
        Check if the two ShardSlices can be merged / added into one.  The two 
        slices must be next to each other in the adding dimension.  The slices 
        ahead of the adding dimension must be the same.  The slices after the 
        adding dimension must be the full  slice, which can be verified using 
        the full_shape.
        
        Returns:
            bool: Whether the two ShardSlices can be merged.
            int: The adding dimension.  If the two ShardSlices can't be merged,
                the adding dimension is None.
        """
        if self.full_shape != other.full_shape:
            return False, None
        
        adding_dim = None
        
        # find the adding dimension and do checking
        for dim in range(len(self)):
            if self[dim] == other[dim]:  # leading dims
                continue
            # first different dimension
            if adding_dim is None:
                if self[dim].stop == other[dim].start or \
                    self[dim].start == other[dim].stop:
                    adding_dim = dim
                else:
                    return False, None
            else:  # trailing dims
                full_slice = slice(0, self.full_shape[dim])
                if self[dim] != full_slice or other[dim] != full_slice:
                    return False, None
        return True, adding_dim

    def has_overlap(self, other: "ShardSlice") -> bool:
        """
        Has overlap means has overlap in all dimensions.
        """
        for s1, s2 in zip(self.shard_slice, other.shard_slice):
            if not(
                s1.start < s2.stop and s1.stop > s2.start or
                s2.start < s1.stop and s2.stop > s1.start
            ):
                return False
        return True

    def is_full(self) -> bool:
        """
        Check if the ShardSlice is the full tensor.
        """
        for dim, size in enumerate(self.full_shape):
            if self.shard_slice[dim] != slice(0, size):
                return False
        return True

    @staticmethod
    def get_completeness(shard_slices: Sequence["ShardSlice"]) -> torch.Tensor:
        """
        Check if the list of ShardSlices is complete, i.e., covers the full
        tensor.
        """
        full_shape = shard_slices[0].full_shape
        for shard_slice in shard_slices:
            assert shard_slice.full_shape == full_shape

        mask = torch.zeros(full_shape, dtype=torch.int8)
        for shard_slice in shard_slices:
            mask[shard_slice.shard_slice] += 1
        return mask


__all__ = ["ShardSlice"]
