"""Run OpenFold3 / RF3 / ESMFold directly on this machine's GPU, no Modal.

For anyone without a Modal account: this calls the same core logic
(openfold3_core.py, rf3_core.py, esmfold_core.py) the Modal apps use, but
runs it in-process here instead of dispatching to a remote container.
Requires a CUDA GPU and the relevant tool already installed -- see
notebooks/fold_colab.ipynb, which installs everything and calls these
commands on a Colab GPU runtime.

    python -m fold.local_run openfold3 --input-file examples/hmgcr_rosuvastatin.txt
    python -m fold.local_run rf3 --sequence ... --smiles ...
    python -m fold.local_run esmfold --sequence ...
"""

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _resolve_sequence_smiles(sequence: str, smiles: str, input_file: str) -> tuple[str, str]:
    if input_file:
        from fold.inputs import parse_sequence_smiles_file

        return parse_sequence_smiles_file(input_file)

    if not sequence or not smiles:
        from fold.targets import HMGCR_1HWL_SEQUENCE, ROSUVASTATIN_SMILES

        sequence = sequence or HMGCR_1HWL_SEQUENCE
        smiles = smiles or ROSUVASTATIN_SMILES
        print("no sequence/smiles given, using HMGCR + rosuvastatin as a smoke test")

    return sequence, smiles


def _resolve_job_name(job_name: str, input_file: str) -> str:
    return job_name or (Path(input_file).stem if input_file else "") or "prediction"


@app.command()
def openfold3(
    sequence: str = "",
    smiles: str = "",
    input_file: str = "",
    job_name: str = "",
    cache_dir: str = "./cache/openfold3",
    output_dir: str = "./outputs",
):
    """Run OpenFold3 cofolding on this machine's GPU."""
    from fold.openfold3_core import run_predict

    sequence, smiles = _resolve_sequence_smiles(sequence, smiles, input_file)
    job_name = _resolve_job_name(job_name, input_file)

    job_out = run_predict(sequence, smiles, job_name, Path(cache_dir), Path(output_dir))
    print(f"done -- outputs in {job_out}")


@app.command()
def rf3(
    sequence: str = "",
    smiles: str = "",
    input_file: str = "",
    job_name: str = "",
    use_msa: bool = True,
    cache_dir: str = "./cache/rf3",
    output_dir: str = "./outputs",
):
    """Run RosettaFold3 cofolding on this machine's GPU."""
    from fold.rf3_core import run_predict

    sequence, smiles = _resolve_sequence_smiles(sequence, smiles, input_file)
    job_name = _resolve_job_name(job_name, input_file)

    job_out = run_predict(
        sequence, smiles, job_name, Path(cache_dir), Path(output_dir), use_msa=use_msa
    )
    print(f"done -- outputs in {job_out}")


@app.command()
def esmfold(
    sequence: str = "",
    input_file: str = "",
    out_path: str = "outputs/esmfold_prediction.pdb",
):
    """Run ESMFold single-sequence folding on this machine's GPU."""
    from fold.esmfold_core import fold_sequence

    if input_file:
        from fold.inputs import parse_sequence_smiles_file

        sequence, _ = parse_sequence_smiles_file(input_file)
    if not sequence:
        from fold.targets import HMGCR_1HWL_SEQUENCE

        sequence = HMGCR_1HWL_SEQUENCE
        print("no sequence given, using HMGCR 1HWL construct as a smoke test")

    pdb_bytes = fold_sequence(sequence)
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(pdb_bytes)
    print(f"done -- wrote {dest}")


if __name__ == "__main__":
    app()
