from __future__ import annotations

import pytest

from benchmarks.analyze_omc_csp_outer_policy import (
    _distribution,
    _maximum_nested_delta,
)


def test_outer_policy_distribution_reports_static_dispersion():
    summary = _distribution([2.0, 4.0, 6.0])

    assert summary["minimum"] == 2.0
    assert summary["mean"] == 4.0
    assert summary["maximum"] == 6.0
    assert summary["coefficient_of_variation"] == pytest.approx(
        (2.0 / 3.0) ** 0.5 / 2.0
    )


def test_outer_policy_endpoint_delta_supports_nested_arrays():
    assert _maximum_nested_delta(
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.5, 1.5], [2.0, 5.0]],
    ) == 1.0

    with pytest.raises(ValueError, match="different lengths"):
        _maximum_nested_delta([1.0], [1.0, 2.0])
