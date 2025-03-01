import unittest
import random
import argparse
import neck
from tqdm import trange


class TestRandomTensor(unittest.TestCase):
    reps = 1000
    trials = 10

    @classmethod
    def setUpClass(cls):
        """Parse command-line arguments to control reps and trials."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--reps", type=int, default=1000, help="Number of tensor tests")
        parser.add_argument("--trials", type=int, default=10, help="Number of slice tests per tensor")
        parser.add_argument("--backend", type=str, default="pytorch", help="Backend for tensor operations")
        args, _ = parser.parse_known_args()

        cls.reps = args.reps
        cls.trials = args.trials
        cls.backend = args.backend

    def setUp(self):
        """Set up the test environment by seeding the random number generator."""
        random.seed(42)

    def test_correctness(self):
        """Test if independently generated slices of a tensor are consistent."""
        
        for i in trange(self.reps, desc="Testing Random Tensor Consistency"):
            with self.subTest(i=i):
                dim = random.randint(2, 4)
                shape = tuple(random.randint(10, 50) for _ in range(dim))
                name = f"test_{i}"

                full_tensor = neck.random.rand(name, shape)

                for j in range(self.trials):
                    with self.subTest(j=j):
                        get_slice = lambda dim: slice(*sorted(random.sample(range(dim), 2)))
                        slices = tuple(get_slice(d) for d in shape)

                        sliced_from_full = full_tensor[slices]
                        sliced_from_rng = neck.random.rand(name, shape, slices, self.backend)

                        self.assertTrue(
                            (sliced_from_full == sliced_from_rng).all(),
                            f"Mismatch in trial {j} for tensor {name} with shape {shape}"
                        )


if __name__ == "__main__":
    unittest.main()
