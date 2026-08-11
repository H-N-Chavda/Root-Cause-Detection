import numpy as np

from utils import pc_algorithm


def test_pc_on_synthetic_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y = x + rng.normal(scale=0.3, size=300)
    z = y + rng.normal(scale=0.3, size=300)
    data = np.column_stack([x, y, z])

    result = pc_algorithm(data, alpha=0.05, max_cond_set_size=1)
    assert result.adjacency[0] == {1, 2}
