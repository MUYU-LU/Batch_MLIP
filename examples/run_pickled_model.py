"""Run relaxation or MD with a checkpoint containing a complete nn.Module."""

from __future__ import annotations

import argparse

from ase.io import read, write

from atombit_batch import (
    AseGraphBatch,
    BatchedPotential,
    batched_fire_relax,
    batched_langevin_baoab,
    batched_velocity_verlet,
    initialize_maxwell_boltzmann,
)
from atombit_batch.loaders import infer_cutoff, load_full_torch_model, parse_dtype


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["relax", "nve", "nvt"])
    parser.add_argument("checkpoint")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--checkpoint-key")
    parser.add_argument("--cutoff", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--force-mode", choices=["autograd", "direct", "auto"], default="autograd")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fmax", type=float, default=0.03)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--friction-per-fs", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    model = load_full_torch_model(args.checkpoint, key=args.checkpoint_key)
    cutoff = infer_cutoff(model, args.cutoff)
    state = AseGraphBatch.from_ase(
        read(args.input, index=":"),
        cutoff=cutoff,
        skin=args.skin,
        device=args.device,
        dtype=dtype,
    )
    potential = BatchedPotential(
        model,
        device=args.device,
        dtype=dtype,
        force_mode=args.force_mode,
    )

    if args.mode == "relax":
        result = batched_fire_relax(
            state, potential, fmax=args.fmax, max_steps=args.steps
        )
    else:
        initialize_maxwell_boltzmann(
            state,
            args.temperature_K,
            seed=args.seed,
            remove_com=True,
            force_exact_temperature=True,
        )
        if args.mode == "nve":
            result = batched_velocity_verlet(
                state,
                potential,
                timestep_fs=args.timestep_fs,
                n_steps=args.steps,
            )
        else:
            result = batched_langevin_baoab(
                state,
                potential,
                timestep_fs=args.timestep_fs,
                n_steps=args.steps,
                temperature_K=args.temperature_K,
                friction_per_fs=args.friction_per_fs,
                seed=args.seed + 1,
            )

    write(args.output, result.state.to_ase(result.evaluation, wrap=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
