from __future__ import annotations

from benchmarks.analyze_omc_csp_source_backed_ablation import _same_contract


def _payload(**overrides):
    contract = {
        "optimizer": "bfgs",
        "optimizer_dtype": "torch.float64",
        "cell_filter": "frechet",
        "cutoff_A": 6.0,
        "skin_A": 0.5,
        "force_mode": "autograd",
        "fmax_eV_per_A": 0.05,
        "max_steps": 500,
        "scheduling": "auto",
    }
    contract.update(overrides)
    return {"contract": contract}


def test_contract_comparison_normalizes_legacy_defaults():
    legacy = _payload()
    explicit = _payload(
        linear_algebra_backend="auto",
        tail_recovery="none",
        tail_recovery_optimizer=None,
    )

    assert _same_contract(legacy, explicit)


def test_contract_comparison_keeps_enabled_recovery_significant():
    disabled = _payload(tail_recovery="none")
    enabled = _payload(
        tail_recovery="ase_bfgs",
        tail_recovery_optimizer="bfgs",
    )

    assert not _same_contract(disabled, enabled)
