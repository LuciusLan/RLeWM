# Isolated reproduction diagnostics

These tools investigate data, numerical and environment contracts. They are
separate from the P1/P2/P3 runners and are not paper benchmark methods. Use one
entry point rather than adding another per-probe shell script:

```bash
bash codes/diagnostics/run.sh COMMAND --help
```

| Command | Purpose | Implementation / guide |
|---|---|---|
| `cube-replay` | Restore an episode start, execute its recorded raw actions, compare states and RGB. No expert or learned model. | `cube_replay/`; [guide](../../docs/diagnostics/cube-replay.md) |
| `cube-policy collect\|evaluate\|all` | Paired new Markov/plan-oracle calibration episodes and Frozen H1/H5 prediction; no TTT/CEM. | `cube_policy/`; [guide](../../docs/diagnostics/cube-policy.md) |
| `retrieval-stats measure\|analyze\|all` | Fresh validation F/R overlap and post-U1 reverse/FR-support turnover, with the three-LR grid on four tasks. | `retrieval_stats/`; [guide](../../docs/diagnostics/retrieval-stats.md) |
| `frozen` | Compare native LeWM, live project and cached Frozen latent rollouts. Loads pretrained weights. | `frozen.py`; [guide](../../docs/stages/simulation.md#frozen-rollout-and-environment-audit) |
| `environment` | Package/dependency audit; optional simulator smoke in subprocesses. | `environment.py`; same guide |
| `targets inspect\|replay` | Inspect AdaJEPA's numeric targets or replay them explicitly through SWM PushT. | `targets.py`; [guide](../../docs/stages/simulation.md#adajepa-plan_targetspkl) |

`common.py` contains model-free path/checksum/output helpers; `cli.py` registers
commands and imports their execution code lazily. `run.sh` supplies the headless
launch environment and honors `PYTHON`. `run-frozen.sh` is the existing paired
PushT/Cube convenience wrapper.

The `setup-physics-*` / `with-physics-overlay.sh` utilities manage optional
physics overrides. Prefer the documented
[child venv](../../docs/stages/simulation.md#child-venv-sharing-the-current-packages)
when comparing MuJoCo versions; the main environment is preserved.

## Cube replay module boundary

| File in `cube_replay/` | Responsibility |
|---|---|
| `__init__.py` | Lightweight argument registration; no numerical/simulator imports |
| `data.py` | Read-only HDF5 field aliases, episode selection, action alignment and source receipts |
| `cube.py` | The only replay module that knows SWM/OGBench/MuJoCo internals |
| `report.py` | Numerical comparison, CSV/NPZ and image artifacts; no simulator imports |
| `run.py` | Orchestration and result/exit-status contract |
| `test_contracts.py` | Synthetic data/metric checks; no simulator, weights or project datasets |

Dependency direction is `CLI → replay package → diagnostics.common / external
packages`. Cube replay does **not** import `lewm_tta`, `experiments` or P1/P2/P3
configs. Core experiment modules must not import it. Its raw-control replay is
deliberately different from model-block execution and should not be folded into
the main planner as a convenience helper.

Keep outputs under `artifacts/diagnostics/` (new directory per run); numerical
results belong beside their original artifacts. No replay caches, indexes or
results live in the source package. There is no JSONL dump. For a later
refactor, relocate the package and update its CLI registration and `common`
imports; no main experiment implementation or saved benchmark lock needs to
change. Generalize a backend only after another task actually needs this probe.

The older `codes/lewm_reproduce.py` is still the independent native/project MPC
reference. It has not been moved or rewritten as part of this diagnostic.

## Cube policy-pilot boundary

`cube_policy/protocol.py` owns the profile, independent case seeds, immutable
generation identity and receipts. `collection.py` owns native resets, both
oracles and trajectory storage; it reuses audited stepping/rendering from
`cube_replay/cube.py`. `reporting.py` owns small statistics, previews and tables.
`evaluation.py` selects the archived numerical core and calls the existing
`diagnostics.frozen.one_case`; no second rollout/preprocessing implementation
is introduced. `run.py` coordinates the phases through the same CLI.

The only main-project numerical dependency is one-way, from the Frozen
evaluation phase into the **verified archived** WM adapter/reader. Collection
and behavior summaries do not import the model or main experiment engine.
P1 caches/indexes and benchmark query manifests are unnecessary. No core module
imports this diagnostic package. The profile remains under
`codes/configs/diagnostics/cube-policy.json`; all generated data/results belong
under the pilot artifact root.

Generation and evaluation have separate code identities: an evaluator/report
fix can reuse recorded data with a new evaluation name. The validated plan
recipe is now available in the main `FreshCube` adapter via the `cube-plan`
collection profile; see the [collection guide](../../docs/stages/collection.md#cube-plan-oracle-validation-and-test-collection).
Both implementations call the installed native OGBench oracle. The completed
pilot's generation code stays unchanged to preserve its recorded identity;
the main collector does not import diagnostic orchestration.

## Retrieval-statistics boundary

`retrieval_stats/run.py` pins the completed fresh P2 reports and selects their
verified numerical cores in isolated dataset subprocesses. `measure.py` calls
the existing adaptation session, retriever and support selector; its extra
post-update retrieval does not take another gradient step. Source/query caches
are opened read-only and must already exist. `metrics.py` and `analysis.py`
own numeric set/weight comparisons and episode-clustered CSV summaries.
Updates use reverse-only support by default; the unchanged within-position
forward ranking and hypothetical FR-support changes are supplementary probes.
`--scope predictor` uses the existing full-predictor scope with the action
encoder frozen. `comparison.py` reuses completed last-block records read-only,
checks paired protocol/cache identities and reports episode-clustered scope
differences. No new final-block measurement is needed for this comparison.
Compact compressed per-query receipts permit resume. No main experiment
module depends on this package; it can be removed or relocated independently
of benchmark execution. See the [complete protocol](../../docs/diagnostics/retrieval-stats.md).
