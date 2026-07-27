from tools.generate_heldout_mixed_workloads import evenly_spaced_indices
from tools.shard_mps_workload import worker_capacities


def test_evenly_spaced_indices_include_density_endpoints():
    indices = evenly_spaced_indices(256, 64)

    assert len(indices) == 64
    assert len(set(indices)) == 64
    assert indices[0] == 0
    assert indices[-1] == 255


def test_mps_capacities_cover_arbitrary_pool_exactly():
    capacities = worker_capacities(258)

    assert len(capacities) == 16
    assert sum(capacities) == 258
    assert max(capacities) - min(capacities) == 1
