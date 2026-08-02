from __future__ import annotations

from batch_mlip.workloads import WorkloadJob, WorkloadManifest
from benchmarks.build_mlip_planning_profile import enrich_manifest


def _job(order: int) -> WorkloadJob:
    digest = f"{order + 1:064x}"
    return WorkloadJob(
        system_id=f"job-{order}",
        group_id="family",
        duplicate_group=f"duplicate-{order}",
        order=order,
        dataset_id="dataset",
        source_path=f"job-{order}.cif",
        source_sha256=digest,
        normalized_structure_sha256=digest,
        frame_index=0,
        atom_count=1,
        species=("H",),
        chemical_formula="H",
        pbc=(True, True, True),
        cell_A=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        volume_A3=1.0,
        constraints=(),
        topology_edge_counts={"cutoff=6.000_skin=0.000": 12},
    )


def test_enrichment_preserves_jobs_and_adds_model_topology_contract():
    source = WorkloadManifest(
        workload_id="source",
        version=1,
        family="family",
        operation="optimization",
        cell_mode="variable",
        arrival_mode="closed",
        jobs=(_job(0), _job(1)),
        metadata={"purpose": "test"},
    ).seal()
    enriched = enrich_manifest(
        source,
        {
            ("job-0.cif", 0): (8, 10),
            ("job-1.cif", 0): (14, 18),
        },
        workload_id="source-mace",
        cutoff=4.5,
        skin=0.5,
        model_id="MACE-OFF23-Small",
    )

    enriched.verify()
    assert [job.system_id for job in enriched.jobs] == ["job-0", "job-1"]
    assert [job.order for job in enriched.jobs] == [0, 1]
    assert enriched.jobs[0].topology_edge_counts == {
        "cutoff=4.500_skin=0.000": 8,
        "cutoff=4.500_skin=0.500": 10,
        "cutoff=6.000_skin=0.000": 12,
    }
    assert enriched.metadata["source_workload_manifest_sha256"] == (
        source.manifest_sha256
    )
