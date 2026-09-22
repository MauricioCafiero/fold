# Handoff — fold: ipSAE + OpenFold3 verification, dimer support

Updated 2026-09-22, same day as the ipSAE session: the follow-up session
folded the open decisions into the main app (all committed and pushed).
Everything in the original list below was verified live; the deltas since
are in "What the follow-up session changed".

## What the follow-up session changed (2026-09-22, same day)

1. **Dimer support folded into the main app** (open decision 1):
   `--sequence-b` on both the Modal app (`src/fold/app.py`) and the
   no-Modal CLI (`fold.local_run openfold3`), backed by
   `openfold3_core.build_query_chains(sequence, smiles, second_sequence)`
   — protein+ligand, protein+protein, and protein+protein+ligand queries
   are all valid. Input files take an optional `SEQUENCE_B:` line via the
   new `fold.inputs.parse_input_file` (`CofoldInput` dataclass);
   `parse_sequence_smiles_file` is kept for the RF3/ESMFold paths and still
   rejects files without a SMILES. **RF3 dimer support is NOT included**
   (per-component MSA handling makes it a bigger change).
   `verify_of3_dimer.py` was deleted — its dimer query construction is in
   the main app now, and its barnase/barstar sequences live on as
   `examples/barnase_barstar.txt`.
2. **Interface confidence auto-printed on every cofold run** (open
   decision 3): `fold.analyze.report_interface_scores` (best-effort —
   prints why it can't score instead of failing the run) is called after
   every cofold in `app.py`, `rf3_app.py`, and both `local_run` commands.
   The `interface-scores` CLI shares the same pair-printing code. Verified
   live against the existing outputs: barnase-barstar 0.858 / 0.909 /
   0.467 (ipSAE/pDockQ2/pDockQ on the best sample) and HMGCR 0.150 / 0.150
   / 0.649 — matching the numbers recorded below.
3. 33 tests pass (`uv run pytest -q`), including new ones for
   `parse_input_file`, `build_query_chains`, and
   `report_interface_scores`.

## What the ipSAE session added (2026-09-22)

1. **`src/fold/ipsae.py`** — per-interface confidence scores (ipSAE /
   pDockQ / pDockQ2) for cofolding outputs, adapted from ColabFold's
   MIT-licensed vendored implementation (PR #846). Entry point:
   `interface_scores(conf_json_path, cif_path)` → dict of
   `{ipsae, pdockq, pdockq2, summary}`, keyed by unordered chain pair.
   Also exposed as `python -m fold.analyze interface-scores <job_dir>`.
2. **Cross-validation against the official DunbrackLab `ipsae.py`**
   (the "frun" task): both tools run on a real OpenFold3
   barnase–barstar dimer (5 seeds, outputs in
   `outputs/barnase_barstar_verify/`). Result:
   - **ipSAE: exact agreement to all 6 decimals on every seed.** e.g.
     sample 2 B→A 0.870239 on both tools. Physically sane too (~0.86
     for a femtomolar-affinity heterodimer).
   - **Official tool's pDockQ/pDockQ2 are broken on OpenFold3
     output**: its AF3 path keys on `atom_plddts`, OpenFold3 writes
     `plddt` (per-atom), so the script silently falls into
     `plddt = np.zeros(numres)` and pins at the formula floor
     (0.0183 / 0.0073). It also undercounts chain tokens via its
     CA/`C1`/`C3` CIF atom-name scan (115 vs our exact 121 for chain
     A). Its PAE-only ipSAE is unaffected by both bugs — which is why
     only that column matches.
   - Ours on the same 5 seeds: ipSAE 0.838–0.870 (direction-dependent,
     max over directions is the summary), pDockQ 0.47–0.49, pDockQ2
     0.90–0.91.
3. **AF3/OpenFold3 fallback in `fold.ipsae`** — `_derive_af3_tokens`:
   OpenFold3 full confidences carry **no per-token arrays** (only
   `plddt`/`pde`/`pae`; `plddt` is per-ATOM, in CIF file order), so
   tokens are derived from the CIF (1 per protein residue, 1 per
   ligand atom) when `token_chain_ids` is absent. Covered by
   `test_interface_scores_af3_fallback`.
4. **Modal image + weights bump (the old "step 2")** — `app.py` now
   pins `openfoldconsortium/openfold3:0.5-conda`, and
   `openfold3_core.py` downloads the OpenBind checkpoint
   `of3-ob-2025-06-30-174k.pt` (2.29 GB, plain HTTPS from
   `s3://openfold3-data/openfold3-parameters/`). Verified live.
5. **HMGCR smoke test re-run on the new image** (README numbers were
   from `stable` + the old 155k checkpoint): avg_plddt 88.5, ptm 0.88,
   iptm 0.86, has_clash 0.0 (old: 91.3 / 0.90 / 0.87 / 0.0 — no
   regression). `interface-scores` runs on the fresh output: ipSAE
   0.150, pDockQ2 0.150, pDockQ 0.649. Outputs in
   `outputs/hmgcr_of305/`.
6. README "Interface confidence" section and smoke-test bullet updated;
   project memory updated. All 23 tests pass (`uv run pytest -q`).

## Open decisions

(All three from the original list are now resolved: dimer support is in the
main app, the work is committed and pushed, and interface scores print
automatically on every cofold run.)

## File inventory

(As of the follow-up session everything below is committed. `verify_of3_dimer.py`
from the scratch list is deleted; new files since the ipSAE session:
`examples/barnase_barstar.txt`, `tests/test_query_build.py`.)

Modified:
- `src/fold/app.py` — image pin `0.5-conda` (comment explains PR #142
  and the pixi/Modal-runner breakage)
- `src/fold/openfold3_core.py` — OpenBind checkpoint URL/filename
- `src/fold/analyze.py` — `interface-scores` CLI command
- `README.md` — ipSAE section, caveats, smoke numbers

New (ipSAE session):
- `src/fold/ipsae.py` — the module
- `tests/test_ipsae.py` — 8 tests (+ 2 added by the follow-up session)
- `verify_of3_dimer.py` — scratch dimer job (deleted by the follow-up
  session; sequences preserved in `examples/barnase_barstar.txt`)

Data:
- `outputs/barnase_barstar_verify/…` — the cross-validation dimer
  (note: nested `barnase_barstar_verify/barnase_barstar_verify/seed_42/`)
- `outputs/hmgcr_of305/…` — the re-verified smoke test

## Gotchas that will bite again if upstream changes

- **RF3** writes `atom_plddts` on a 0–1 scale (AF2/AF3 use 0–100);
  `fold.ipsae` detects and rescales. It also writes ligands as ~1
  token/atom, whose tiny pseudo-chain deflates ipSAE's d0 —
  protein–ligand ipSAE is marked experimental in the README.
- **OpenFold3** full confidences: no `token_chain_ids` /
  `token_res_ids` / `atom_plddts`; `plddt` is per-atom in CIF file
  order. `interface_scores` raises a targeted "PR #142" ValueError if
  `pae` is missing (pre-PR-#142 images/PyPI packages).
- **Docker tags**: `stable` is frozen at 2026-03-30 (no PAE);
  `0.5-pixi` breaks the Modal runner (`ModuleNotFoundError: grpclib` —
  pixi's env shadows modal's injected deps); `0.5-conda` works.
- **Checkpoints**: `of3-p2-155k.pt` pairs only with the 0.4.x
  architecture; 0.5.0 needs `of3-ob-2025-06-30-174k.pt` (state_dict
  layout changed — per-block `attention_pair_bias.layer_norm_z`).
- **Official DunbrackLab `ipsae.py`** (a copy is at `/tmp/
  ipsae_official.py`, transient): supports AF2/AF3/Boltz only; crashes
  on RF3 output (KeyError `chain_pair_iptm`; ligand-token shape
  mismatch) and, on OpenFold3 output, silently zeroes pLDDT for
  pDockQ/pDockQ2 (key-name mismatch, see above).
- Weights are cached in the Modal volume `openfold3-weights` (both
  checkpoints are there); the local `~/.cache` equivalent is populated
  by `ensure_weights` on first use.

## Reproduce

```bash
# tests (local, no GPU)
uv run pytest -q

# interface scores on the verified dimer
uv run python -m fold.analyze interface-scores \
  outputs/barnase_barstar_verify/barnase_barstar_verify

# interface scores on the fresh smoke test
uv run python -m fold.analyze interface-scores outputs/hmgcr_of305

# re-run the OpenFold3 smoke test on Modal (~$0.10–0.40 on A10G)
uv run modal run src/fold/app.py --job-name <name>
```