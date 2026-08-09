# fold

Protein structure prediction and protein/ligand cofolding, with no local GPU
required.

OpenFold3 and ESMFold need more VRAM than most machines have locally, so
those run on [Modal](https://modal.com/) instead — all you need locally is
CPU and a Modal account. ESM2 embeddings are small enough to run on CPU
directly. The split:

| Tool | What it does | Where it runs | Why |
|---|---|---|---|
| [OpenFold3](https://github.com/aqlaboratory/openfold-3) | Protein + ligand cofolding | Modal GPU (A10G) | needs 32GB+ VRAM |
| [RosettaFold3 (RF3)](https://github.com/RosettaCommons/foundry) | Protein + ligand cofolding (independent method, for cross-checking) | Modal GPU (A10G) | same VRAM class as OpenFold3 |
| [ESMFold](https://huggingface.co/facebook/esmfold_v1) | Single-sequence structure prediction | Modal GPU (A10G) | ~3B-param backbone, ~11GB of weights |
| ESM2 | Protein embeddings + cosine similarity | Local CPU | small enough (650M params, ~2.6GB) to run comfortably on CPU |

Both GPU workloads cache their model weights in a persistent [Modal
Volume](https://modal.com/docs/guide/volumes) on first run, so subsequent
runs skip the multi-GB download. The local ESM2 path uses the standard
HuggingFace Hub cache (`~/.cache/huggingface`), which works the same way.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) if you don't have it.
2. Clone this repo and install dependencies:

   ```bash
   git clone https://github.com/MauricioCafiero/fold.git
   cd fold
   uv sync
   ```

3. Authenticate with [Modal](https://modal.com/) (needed for cofolding and
   ESMFold, not for the local ESM2 embeddings):

   ```bash
   uv run modal setup
   ```

   This opens a browser flow and writes a token to `~/.modal.toml`. Verify it
   worked with `uv run modal profile current`.

## Usage

### Protein/ligand cofolding (OpenFold3)

Runs on a Modal A10G GPU. `--detach` keeps the job running on Modal even if
your machine sleeps or disconnects.

```bash
uv run modal run --detach src/fold/app.py::main \
  --sequence "YOUR_PROTEIN_SEQUENCE" \
  --smiles "YOUR_LIGAND_SMILES" \
  --job-name my_target
```

Omit `--sequence`/`--smiles` to run the built-in smoke test instead: human
HMG-CoA reductase (catalytic domain, the exact construct from [PDB
1HWL](https://www.rcsb.org/structure/1HWL)) cofolded with rosuvastatin.

MSAs are computed remotely via the ColabFold MSA server (no local sequence
databases needed). Output structures (`.cif`), per-sample confidence scores,
and run metadata land in `outputs/<job_name>/`.

A single run currently takes ~7 minutes on A10G (mostly diffusion sampling;
the remote MSA search itself is seconds). First run is much slower — it has
to build the container image and download the ~2.1GB checkpoint.

### Protein/ligand cofolding, second opinion (RosettaFold3)

Same idea as OpenFold3 above, but an architecturally independent model —
useful for cross-checking a prediction rather than trusting one method's
confidence score alone.

```bash
uv run modal run --detach src/fold/rf3_app.py::main \
  --sequence "YOUR_PROTEIN_SEQUENCE" \
  --smiles "YOUR_LIGAND_SMILES" \
  --job-name my_target
```

Omit `--sequence`/`--smiles` for the same HMGCR + rosuvastatin smoke test.
Unlike the OpenFold3 path, this one does **not** compute an MSA automatically
— the RF3 CLI accepts a precomputed `msa_path` (`.a3m`/`.fasta`) per protein
component, but has no equivalent of `--use_msa_server`. Without one it runs
effectively single-sequence, which noticeably lowers confidence scores (see
Smoke tests below) — for a fair comparison against OpenFold3, generate an
MSA yourself and wire it into `query_components` in `rf3_app.py`.

Output lands in `outputs/rf3_<job_name>/`, structured as `.cif` models plus
`_confidences.json`/`_summary_confidences.json` per sample.

### Single-sequence folding (ESMFold)

No MSA step needed — much simpler and faster than cofolding.

```bash
uv run modal run --detach src/fold/esmfold_app.py::main \
  --sequence "YOUR_PROTEIN_SEQUENCE"
```

Writes a `.pdb` file to `outputs/esmfold_prediction.pdb` (path configurable
via `--out-path`). Omit `--sequence` to run the same HMGCR construct as a
smoke test.

### Protein embeddings + similarity (ESM2, local)

No Modal, no GPU — runs directly on CPU.

```python
from fold.embeddings import embed_sequence, cosine_similarity

vec = embed_sequence("MLSRLFRMHGLFVASHPWEVIVG...")
sim = cosine_similarity(seq_a, seq_b)
```

Uses `facebook/esm2_t33_650M_UR50D` by default; pass `model_name=` to use a
different ESM2 checkpoint size.

## Smoke tests

All four paths have been run end-to-end and verified, using human HMG-CoA
reductase (HMGCR) as the test protein throughout:

- **OpenFold3 cofolding** — the catalytic-domain construct from [PDB
  1HWL](https://www.rcsb.org/structure/1HWL) cofolded with rosuvastatin
  (SMILES from PubChem CID 446157, cross-checked against the PDB ligand
  FBI), MSA computed via the ColabFold server. Result: `avg_plddt` 91.3,
  `ptm` 0.90, `iptm` 0.87, `has_clash` 0.0 — a confident, clash-free
  prediction, consistent with the real crystal structure it's based on.
- **RosettaFold3 cofolding** — same protein + ligand, but no MSA supplied
  (see caveat above). Result: `overall_plddt` 0.67, `ptm` 0.36, `iptm` 0.39,
  `has_clash` false — ran cleanly with no errors, but noticeably
  lower-confidence than the MSA-backed OpenFold3 run. Not a fair
  method-vs-method comparison as configured; mainly confirms the pipeline
  itself works.
- **ESMFold** — same HMGCR sequence, single-chain (no ligand). Produced a
  valid 3,483-atom PDB structure with per-residue confidence in the B-factor
  column.
- **ESM2 embeddings** — cosine similarity of the HMGCR sequence against
  itself (1.0000, as expected) and against a partially-scrambled decoy
  (0.9158, correctly lower).

## Cost notes

- Compute only bills while a Modal function is actually running — ephemeral
  `modal run` jobs (even detached ones) stop billing once they finish. Check
  `modal app list` if you want to confirm nothing's still active.
- The cached weights in Modal Volumes cost storage money indefinitely
  (roughly $0.10/GB-month) until deleted with `modal volume delete <name>`.
  This is deliberate — it's what avoids re-downloading multi-GB checkpoints
  on every run — but worth knowing about if the project goes idle for a long
  time.
