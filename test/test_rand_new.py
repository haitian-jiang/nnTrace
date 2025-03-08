import unittest
import neck
from neck.structs import ShardNDSlices, NeckSlice, NDSlice


class TestRand(unittest.TestCase):
    def test_rand(self):
        full_shape = (10, 23, 30)
        name = "test_rand"
        full_tensor = neck.random.rand(name, ShardNDSlices(full_shape))

        local_1 = NDSlice(
            NeckSlice(0, 5), NeckSlice(0, 10), NeckSlice(0, 15)
        )
        local_2 = NDSlice(
            NeckSlice(0, 5), NeckSlice(0, 10), NeckSlice(15, 30)
        )
        global_1 = NDSlice(
            NeckSlice(0, 5), NeckSlice(0, 10), NeckSlice(0, 15)
        )
        global_2 = NDSlice(
            NeckSlice(5, 10), NeckSlice(10, 20), NeckSlice(15, 30)
        )
        local_shape = (5, 10, 30)

        shard_slice = ShardNDSlices(
            full_shape, local_shape, {local_1: global_1, local_2: global_2}
        )
        local_tensor = neck.random.rand(name, shard_slice)

        for local_, global_ in shard_slice.slice_map.items():
            self.assertTrue(
                (full_tensor[global_.pyslice] == local_tensor[local_.pyslice]).all(),
                f"Local tensor {local_} does not match global tensor {global_}"
            )

if __name__ == "__main__":
    unittest.main()
