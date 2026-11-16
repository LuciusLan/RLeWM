# Experiment code

Use `bash codes/run.sh STAGE COMMAND`. The [stage guides](../docs/README.md)
document the active commands; `bash codes/run.sh list` shows available profiles.

```text
codes/
├── run.sh                  # One CUDA/environment launcher
├── experiment.py           # One Python entrypoint
├── experiments/
│   ├── cli.py              # Stage/variant dispatch, dry runs, report replay
│   ├── catalog.py          # Active stages and backend selection
│   ├── config.py           # Inheritance and immutable resolved JSON inputs
│   ├── provenance.py       # Runtime identities and verified source archives
│   ├── workflows.py        # Complete/resume the fixed six-cell P3 comparison
│   ├── progress.py         # Current and archived terminal progress
│   ├── backends/           # P0 and P1 command frontends
│   └── tools/              # Extraction, episode splits, saved-result figures
├── configs/
│   ├── p1/                 # Shared dataset/checkpoint/cache/retrieval source
│   ├── p2/                 # Base + temporal, initial, follow-up profiles
│   ├── p3/                 # Base + factorial, shared, proposal profiles
│   ├── sim/                # Small paired simulator replay / MPC profile
│   ├── collect/            # Fresh evaluation episodes and optional shift regimes
│   └── coverage/           # Frozen rank-coverage diagnostic
├── lewm_tta/
│   ├── *.py                # Model, caches, exact retrieval, adaptation, P1
│   ├── collection/         # Reservation, atomic storage, auditing, external query banks
│   ├── rollout/            # P2 scopes, schedules, cohorts, metrics
│   ├── planning/           # Shared TTT, actions, CEM
│   ├── proposals/          # Action-fragment retrieval, injection, factorials
│   ├── simulation/         # Shared real execution; task-specific environments/
│   ├── rank_coverage/      # Frozen coverage, compact records, statistics
│   └── splits/             # Episode-partition and exposure metadata
├── requirements/p0.txt     # Audited dependencies
├── tests/                  # Model-free interface/statistics checks
└── legacy/                 # Former scripts/configs; reference and compatibility
```

P2's former P2/P2.5/follow-up executables select the same `rollout/` engine.
P3a/b/c select `planning/` and `proposals/`. There is no duplicate numerical
implementation. The old `lewm_tta.p2` and `lewm_scope` imports remain compatibility
aliases.

`sim` reuses those planning components for saved-plan simulator replay and
closed-loop MPC. Physics, state reconstruction and task scores are isolated
under `simulation/environments/`; see the [pilot guide](../docs/stages/simulation.md).

```bash
bash codes/run.sh prepare-cohorts --dry-run
bash codes/run.sh p2 evaluate --variant initial --dry-run
bash codes/run.sh p3 evaluate --variant proposals --dry-run
bash codes/run.sh p3 compare --dry-run
```

The main profiles use the new split and preserve prepared cache identities.
Older configs are in [legacy/configs](legacy/configs/); historical reports and
locks use verified source replay. See [legacy migration](legacy/README.md) and
[the shared protocol](../docs/protocol.md).
