import torch
import random
import unittest
import argparse
from tqdm import trange
from neck.structs import ShardSlice, TensorInfo


class TestShardSlice(unittest.TestCase):
    reps = 100

    def setUp(self):
        torch.manual_seed(0)
        random.seed(0)

    def test_add_tensor_info(self):
        for i in trange(self.reps, desc="Testing Add Tensor Info"):
            with self.subTest(i=i):
                num_tensors = random.randint(10, 20)
                dim = random.randint(2, 4)
                full_shape = tuple(random.randint(10, 50) for _ in range(dim))
                c_name = f"test_{i}"
                tensors = [torch.rand(full_shape) for _ in range(num_tensors)]
                tensor_infos = []
                for t in tensors:
                    t_info = TensorInfo(t, c_name, ShardSlice(full_shape))
                    tensor_infos.append(t_info)
                sum_tensor = sum(tensors).to(torch.float64)
                sum_tensor_info = TensorInfo.add_tensor_info(tensor_infos)
                self.assertTrue(torch.allclose(sum_tensor, sum_tensor_info.tensor))



if __name__ == "__main__":
    unittest.main()
