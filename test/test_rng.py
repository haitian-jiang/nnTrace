import neck
import random
from tqdm import trange


def test_correctness(reps=1000, trials=10):
    for i in trange(reps):
        # print(f"Testing tensor {i + 1}/{reps}")
        dim = random.randint(2, 4)
        shape = tuple(random.randint(10, 50) for _ in range(dim))  # Random shape
        name = "test" + str(i)
        # print(f"Testing tensor {name} with shape {shape}")
        full_tensor = neck.random.rand(name, shape)
        for j in range(trials):
            get_slice = lambda dim: slice(*sorted(random.sample(range(dim), 2)))
            slices = tuple(get_slice(d) for d in shape)
            sliced_from_full = full_tensor[slices]
            sliced_from_rng = neck.random.rand(name, shape, slices)
            assert (sliced_from_full == sliced_from_rng).all(), f"Mismatch in trial {j} for tensor {name} with shape {shape}"
    print("All tests passed!")


if __name__ == "__main__":
    random.seed(42)
    test_correctness()
