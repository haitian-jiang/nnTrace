import unittest
import torch
from neck.structs import ShardNDSlices
from neck.structs.slices import NDSlice, NeckSlice


class TestNeckSlice(unittest.TestCase):
    def test_init(self):
        slice1 = NeckSlice(0, 10)
        self.assertEqual(slice1.start, 0)
        self.assertEqual(slice1.stop, 10)
        with self.assertRaises(AssertionError):
            NeckSlice(10, 0)

    def test_order(self):
        slice1 = NeckSlice(0, 10)
        slice2 = NeckSlice(10, 20)
        slice3 = NeckSlice(0, 20)
        self.assertLess(slice1, slice2)
        self.assertLess(slice1, slice3)
        self.assertLessEqual(slice3, slice3)
        self.assertGreater(slice2, slice1)
        self.assertGreater(slice3, slice1)
        self.assertGreaterEqual(slice3, slice3)

    def test_repr(self):
        slice1 = NeckSlice(0, 10)
        self.assertEqual(repr(slice1), "0:10")

    def test_pyslice(self):
        slice1 = NeckSlice(0, 10)
        self.assertEqual(slice1.pyslice, slice(0, 10))

class TestNDSlice(unittest.TestCase):
    def test_init(self):
        slice1 = NDSlice(NeckSlice(0, 10), NeckSlice(0, 10))
        slice2 = NDSlice((NeckSlice(0, 10), NeckSlice(0, 10)))
        slice3 = NDSlice([NeckSlice(0, 10), NeckSlice(0, 10)])
        slice4 = NDSlice(s for s in [NeckSlice(0, 10), NeckSlice(0, 10)])
        self.assertEqual(slice1, slice2)
        self.assertEqual(slice1, slice3)
        self.assertEqual(slice1, slice4)
        slice5 = NDSlice(NeckSlice(0, 10))
        self.assertEqual(slice5, slice1[:1])
        shape = torch.Size((10, 10))
        slice6 = NDSlice(shape)
        self.assertEqual(slice6, slice1)


    def test_repr(self):
        slice1 = NDSlice(NeckSlice(0, 10), NeckSlice(0, 10))
        self.assertEqual(repr(slice1), "[0:10, 0:10]")
    
    def test_shape(self):
        slice1 = NDSlice(NeckSlice(0, 10), NeckSlice(0, 10))
        self.assertEqual(slice1.shape, torch.Size((10,10)))
        slice2 = NDSlice(NeckSlice(5, 10), NeckSlice(13, 20))
        self.assertEqual(slice2.shape, torch.Size((5, 7)))
    
    def test_pyslice(self):
        slice1 = NDSlice(NeckSlice(0, 10), NeckSlice(0, 10))
        self.assertEqual(slice1.pyslice, (slice(0, 10), slice(0, 10)))


class TestShardNDSlices(unittest.TestCase):
    def test_init_full_shape(self):
        shard = ShardNDSlices((8, 8))
        self.assertEqual(shard.full_shape, (8, 8))
        self.assertEqual(shard.local_shape, (8, 8))
        self.assertEqual(shard.slice_map, {NDSlice(NeckSlice(0, 8), NeckSlice(0, 8)): NDSlice(NeckSlice(0, 8), NeckSlice(0, 8))})

    def test_initialization(self):
        full_shape = (1024, 1024)
        local_shape = (512, 1024)
        slice_map = {
            NDSlice(NeckSlice(0, 256), NeckSlice(0, 1024)): NDSlice(NeckSlice(0, 256), NeckSlice(0, 1024)),
            NDSlice(NeckSlice(256, 512), NeckSlice(0, 1024)): NDSlice(NeckSlice(768, 1024), NeckSlice(0, 1024)),
        }
        shard = ShardNDSlices(full_shape, local_shape, slice_map)
        self.assertEqual(shard.full_shape, full_shape)
        self.assertEqual(shard.local_shape, local_shape)
        self.assertEqual(shard.slice_map, slice_map)

    def test_hashing(self):
        full_shape = (1024, 1024)
        local_shape = (512, 1024)
        slice_map1 = {
            NDSlice(NeckSlice(0, 256), NeckSlice(0, 1024)): NDSlice(NeckSlice(0, 256), NeckSlice(0, 1024)),
            NDSlice(NeckSlice(256, 512), NeckSlice(0, 1024)): NDSlice(NeckSlice(768, 1024), NeckSlice(0, 1024)),
        }
        slice_map2 = {
            NDSlice(NeckSlice(256, 512), NeckSlice(0, 1024)): NDSlice(NeckSlice(768, 1024), NeckSlice(0, 1024)),
            NDSlice(NeckSlice(0, 256), NeckSlice(0, 1024)): NDSlice(NeckSlice(0, 256), NeckSlice(0, 1024)),
        }
        shard1 = ShardNDSlices(full_shape, local_shape, slice_map1)
        shard2 = ShardNDSlices(full_shape, local_shape, slice_map2)
        shard_set = {shard1, shard2}
        self.assertEqual(len(shard_set), 1)

    def test_length(self):
        full_shape = (1024, 1024)
        local_shape = (512, 1024)
        slice_map = {
            NDSlice(NeckSlice(0, 256), NeckSlice(0, 1024)): NDSlice(NeckSlice(0, 256), NeckSlice(0, 1024)),
            NDSlice(NeckSlice(256, 512), NeckSlice(0, 1024)): NDSlice(NeckSlice(768, 1024), NeckSlice(0, 1024)),
        }
        shard = ShardNDSlices(full_shape, local_shape, slice_map)
        self.assertEqual(len(shard), 2)

    def test_repr(self):
        full_shape = (1024, 1024)
        local_shape = (512, 1024)
        slice_map = {
            NDSlice(NeckSlice(0, 256), NeckSlice(0, 1024)): NDSlice(NeckSlice(0, 256), NeckSlice(0, 1024)),
            NDSlice(NeckSlice(256, 512), NeckSlice(0, 1024)): NDSlice(NeckSlice(768, 1024), NeckSlice(0, 1024)),
        }
        shard = ShardNDSlices(full_shape, local_shape, slice_map)
        self.assertEqual(repr(shard), "[0:256⨁768:1024, 0:1024]@(1024, 1024)")

    def test_repr_2d(self):
        full_shape = (8, 8)
        local_shape = (4, 8)
        slice_map = {
            NDSlice(NeckSlice(0, 4), NeckSlice(0, 4)): NDSlice(NeckSlice(0, 4), NeckSlice(0, 4)),
            NDSlice(NeckSlice(0, 4), NeckSlice(4, 8)): NDSlice(NeckSlice(4, 8), NeckSlice(4, 8)),
        }
        shard = ShardNDSlices(full_shape, local_shape, slice_map)
        self.assertEqual(repr(shard), "[0:4, 0:4]⨁[4:8, 4:8]@(8, 8)")

    def test_overlap(self):
        full_shape = (1024, 1024)
        local_shape1 = (512, 1024)
        slice_map1 = {
            NDSlice(NeckSlice(0, 256), NeckSlice(0, 1024)): NDSlice(NeckSlice(0, 256), NeckSlice(0, 1024)),
            NDSlice(NeckSlice(256, 512), NeckSlice(0, 1024)): NDSlice(NeckSlice(768, 1024), NeckSlice(0, 1024)),
        }
        local_shape2 = (1024, 1024)
        slice_map2 = {
            NDSlice(NeckSlice(0, 1024), NeckSlice(0, 1024)): NDSlice(NeckSlice(0, 1024), NeckSlice(0, 1024)),
        }
        shard1 = ShardNDSlices(full_shape, local_shape1, slice_map1)
        shard2 = ShardNDSlices(full_shape, local_shape2, slice_map2)
        self.assertTrue(shard1.has_overlap(shard2))


if __name__ == "__main__":
    unittest.main()
