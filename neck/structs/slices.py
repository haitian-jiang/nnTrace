from typing import Sequence, Optional, Union, Iterable
import torch


class NeckSlice:  # immutable slice, replace `slice`, not support step yet.
    def __init__(self, start: int, stop: int):
        assert 0 <= start < stop
        self.start = start
        self.stop = stop

    def __eq__(self, other):
        return self.start == other.start and self.stop == other.stop
    
    def __lt__(self, other):
        return (self.start, self.stop) < (other.start, other.stop)
    
    def __gt__(self, other):
        return (self.start, self.stop) > (other.start, other.stop)
    
    def __le__(self, other):
        return (self.start, self.stop) <= (other.start, other.stop)
    
    def __ge__(self, other):
        return (self.start, self.stop) >= (other.start, other.stop)
    
    def __hash__(self):
        return hash((self.start, self.stop))

    def __repr__(self):
        return f"{self.start}:{self.stop}"
    
    def __str__(self):
        return self.__repr__()
    
    @property
    def pyslice(self) -> slice:
        return slice(self.start, self.stop)


class NDSlice(tuple):
    def __new__(
            cls,
            *args: NeckSlice | torch.Size | Iterable[NeckSlice]
        ):
        if len(args) == 1:
            if isinstance(args[0], NeckSlice):
                nd_slice = (args[0], )
            elif isinstance(args[0], torch.Size):  # order of elif matters!
                nd_slice = tuple(NeckSlice(0, dim) for dim in args[0])
            elif isinstance(args[0], Iterable):  # Iterable[NeckSlice]
                nd_slice = tuple(args[0])
            else:
                raise ValueError(f"Wrong input {args}")
        else:  # args is tuple[NeckSlice] already
            nd_slice = args
        return super().__new__(cls, nd_slice)
    
    def __repr__(self):
        return f"[{', '.join(repr(s) for s in self)}]"

    def __str__(self):
        return self.__repr__()
    
    @property
    def shape(self) -> torch.Size:
        return torch.Size(s.stop - s.start for s in self)
    
    @property
    def pyslice(self) -> tuple[slice, ...]:
        return tuple(s.pyslice for s in self)


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
            full_shape: Sequence[int],
            shard_slice: Optional[Sequence[slice]]=None, 
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
        return ShardSlice(self.full_shape, new_)

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


class ShardNDSlices:
    def __init__(
            self, 
            full_shape: Sequence[int],
            local_shape: Optional[Sequence[int]] = None,
            slice_map: Optional[dict[NDSlice, NDSlice]] = None,
        ):

        # deal with full shape initialization
        if slice_map is None:  # a full slice
            local_shape = full_shape  # overwrite the local shape
            full_slice = NDSlice(NeckSlice(0, dim) for dim in full_shape)
            slice_map = {full_slice: full_slice}

        self._check_init_input(full_shape, local_shape, slice_map)
        self.full_shape = torch.Size(full_shape)
        self.local_shape = torch.Size(local_shape)
        self.slice_map = slice_map

    @staticmethod
    def _check_init_input(
            full_shape: Sequence[int], 
            local_shape: Sequence[int], 
            slice_map: dict[NDSlice, NDSlice],
        ):
        assert len(full_shape) == len(local_shape), "Expect same dim."
        for local_, global_ in slice_map.items():
            # same dim
            assert len(local_) == len(global_) == len(full_shape)
            # no out of bound
            for s, dim in zip(local_, local_shape):
                assert 0 <= s.start < s.stop <= dim, "Local slice out of bound."
            for s, dim in zip(global_, full_shape):
                assert 0 <= s.start < s.stop <= dim, "Global slice out of bound."
        
        # check the local slices covers cover the whole local_shape
        mask = torch.zeros(local_shape, dtype=torch.int8)
        for local_ in slice_map.keys():
            mask[local_.pyslice] += 1
        assert torch.all(mask == 1), "Local slices wrong."

        # check the global slices has no overlap
        mask = torch.zeros(full_shape, dtype=torch.int8)
        for global_ in slice_map.values():
            mask[global_.pyslice] += 1
        assert torch.all(mask <= 1), "Global slices overlap."

    def __eq__(self, other: "ShardNDSlices") -> bool:
        return (
            self.full_shape == other.full_shape and 
            self.local_shape == other.local_shape and 
            self.slice_map == other.slice_map
        )
    
    def __len__(self):
        return len(self.full_shape)


    def __hash__(self):
        slices = tuple(sorted(self.slice_map.items()))
        return hash(
            (self.full_shape, self.local_shape, slices)
        )

    def __repr__(self):
        if len(self.slice_map) == 1:
            # [0:4096, 2048:4096]@(4096, 4096)
            shard_slice: NDSlice = list(self.slice_map.values())[0]
            shp = f"({', '.join(str(dim) for dim in self.full_shape)})"
            return f"{shard_slice}@{shp}"
        # check if the pieces scattered in one dimension
        # [0:256⨁768:1024, 0:1024]@(1024, 1024)
        # True most of the time
        slices: list[NDSlice] = sorted(self.slice_map.values())
        if (d := self._all_same_but_one_dim(list(self.slice_map.values()))) is not None:
            inner = [str(_1d) for _1d in slices[0]]  # ["0:256", "0:1024"]
            inner[d] = "⨁".join(str(s[d]) for s in slices)  # "0:256⨁768:1024"
            slc = f"[{', '.join(inner)}]"
            shp = f"({', '.join(str(dim) for dim in self.full_shape)})"
            return f"{slc}@{shp}"
        else: # join each of them one by one, for 2D+ TP or partitions
            # [0:4, 0:4]⨁[4:8, 4:8]@(8, 8)
            slc = "⨁".join(str(subslice) for subslice in slices)
            shp = f"({', '.join(str(dim) for dim in self.full_shape)})"
            return f"{slc}@{shp}"

    def __str__(self):
        return self.__repr__()
    
    @staticmethod
    def _all_same_but_one_dim(slices: list["NDSlice"]) -> Optional[int]:
        """
        Check if the subslices are scattered in only one dimension.
        If true, return the dimension. Otherwise, return None.
        """
        ndim = len(slices[0])
        assert all(len(s) == ndim for s in slices), "Different dims."
        sub_slices = sorted(slices)
        are_same = [False] * ndim
        for dim in range(ndim):
            are_same[dim] = all(
                s[dim] == sub_slices[0][dim] for s in sub_slices
            )
        if sum(are_same) == ndim - 1:
            return are_same.index(False)
        return None

    def is_full(self) -> bool:
        # local slice coverage is checked in the initialization
        return self.local_shape == self.full_shape
    
    def get_eigen_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.full_shape, dtype=torch.int8)
        for global_ in self.slice_map.values():
            mask[global_.pyslice] += 1
        return mask
    
    def has_overlap(self, other: "ShardNDSlices") -> bool:
        assert self.full_shape == other.full_shape
        mask = self.get_eigen_mask() + other.get_eigen_mask()
        return torch.any(mask > 1)

    @staticmethod
    def get_completeness(shards: Sequence["ShardNDSlices"]) -> torch.Tensor:
        """
        Check if the list of ShardNDSlices is complete, i.e., covers the full
        tensor.
        """
        full_shape = shards[0].full_shape
        for shard in shards:
            assert shard.full_shape == full_shape

        mask = torch.zeros(full_shape, dtype=torch.int8)
        for shard in shards:
            mask += shard.get_eigen_mask()
        return mask



__all__ = ["ShardSlice", "ShardNDSlices", "NeckSlice", "NDSlice"]
