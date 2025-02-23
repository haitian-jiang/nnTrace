import unittest
from neck.structs import ShardSlice

class TestShardSlice(unittest.TestCase):
    def test_initialization(self):
        shard = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        self.assertEqual(shard.shard_slice, (slice(0, 4), slice(4, 8)))
        self.assertEqual(shard.full_shape, (8, 8))

    def test_repr(self):
        shard = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        self.assertEqual(repr(shard), "[0:4, 4:8]")

    def test_equality(self):
        shard1 = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        shard2 = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        shard3 = ShardSlice([slice(0, 4), slice(2, 6)], [8, 8])
        self.assertEqual(shard1, shard2)
        self.assertNotEqual(shard1, shard3)
    
    def test_length(self):
        shard = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        self.assertEqual(len(shard), 2)

    def test_getitem(self):
        shard = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        self.assertEqual(shard[0], slice(0, 4))
        self.assertEqual(shard[1], slice(4, 8))

    def test_iteration(self):
        shard = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        self.assertEqual(list(iter(shard)), [slice(0, 4), slice(4, 8)])

    def test_hashing(self):
        shard1 = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        shard2 = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        shard_set = {shard1, shard2}
        self.assertEqual(len(shard_set), 1)
    
    def test_comparison(self):
        shard1 = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        shard2 = ShardSlice([slice(4, 8), slice(4, 8)], [8, 8])
        self.assertLess(shard1, shard2)
        self.assertGreater(shard2, shard1)
    
    def test_addition(self):
        shard1 = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        shard2 = ShardSlice([slice(4, 8), slice(4, 8)], [8, 8])
        shard_sum = shard1 + shard2
        self.assertEqual(shard_sum, ShardSlice([slice(0, 8), slice(4, 8)], [8, 8]))
    
    def test_mergeable(self):
        shard1 = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        shard2 = ShardSlice([slice(4, 8), slice(4, 8)], [8, 8])
        mergeable, dim = shard1.mergeable(shard2)
        self.assertTrue(mergeable)
        self.assertEqual(dim, 0)
    
    def test_has_overlap(self):
        shard1 = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        shard2 = ShardSlice([slice(2, 6), slice(4, 8)], [8, 8])
        shard3 = ShardSlice([slice(4, 8), slice(4, 8)], [8, 8])
        self.assertTrue(shard1.has_overlap(shard2))
        self.assertFalse(shard1.has_overlap(shard3))
    
    def test_sum(self):
        shard1 = ShardSlice([slice(0, 4), slice(4, 8)], [8, 8])
        shard2 = ShardSlice([slice(4, 8), slice(4, 8)], [8, 8])
        shard_sum = sum([shard1, shard2])
        self.assertEqual(shard_sum, ShardSlice([slice(0, 8), slice(4, 8)], [8, 8]))

    def test_complete(self):
        shard1 = ShardSlice([slice(0, 4), slice(0, 4)], (8, 8))
        shard2 = ShardSlice([slice(0, 4), slice(4, 8)], (8, 8))
        shard3 = ShardSlice([slice(4, 8), slice(0, 4)], (8, 8))
        shard4 = ShardSlice([slice(4, 8), slice(4, 8)], (8, 8))
        self.assertTrue(ShardSlice.is_complete([shard1, shard2, shard3, shard4]))
        shard5 = ShardSlice([slice(2, 6), slice(2, 6)], (8, 8))
        self.assertFalse(ShardSlice.is_complete([shard1, shard2, shard3, shard4, shard5]))
        shard6 = ShardSlice([slice(0, 3), slice(0, 4)], (8, 8))
        self.assertFalse(ShardSlice.is_complete([shard6, shard2, shard3, shard4]))

if __name__ == "__main__":
    unittest.main()
