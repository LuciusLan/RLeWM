# Active configurations

| Directory | Contents |
|---|---|
| `p1/` | PushT source split and memory-only Two-Room/Reacher/Cube profiles; checkpoint, data, cache and retrieval |
| `p2/` | Shared base, schedule/scope variants and the fresh-query comparison |
| `p3/` | Factorial, shared-TTT, proposal and temporal variants, including fresh-query profiles |
| `coverage/` | Independent frozen rank-coverage pilot |
| `collect/` | Fresh evaluation trajectory collection and optional shifts |
| `sim/` | Physical simulator replay/MPC profiles |

Profiles inherit through `extends`, relative to their own file. Dictionaries
merge recursively; arrays and scalars replace their parent values. The unified
launcher resolves inheritance before calling the existing engine validators.

```bash
bash codes/run.sh p2 config
bash codes/run.sh p3 config --variant shared
bash codes/run.sh p3 evaluate --variant proposals --dry-run
bash codes/run.sh fresh prepare --dry-run
```

`p2/fresh.json`, `p3/fresh.json` and `p3/fresh-temporal.json` bind
`query_dataset.collection_config` to `collect/default.json`. The resolver derives
the regime's query manifest and shared cache from that collection's current
`output_dir`, then freezes explicit paths in the engine input. A custom
`query_dataset.cache_dir` can override only the cache location. Explicit
`manifest`/`cache_dir` query settings remain supported for standalone profiles.
The [fresh workflow](../../docs/stages/collection.md#consolidated-p2p3-workflow-for-the-existing-collection)
uses these profiles for preparation, checks, validation and locked test.

Editing shared P1 data/preprocessing/retrieval settings can invalidate prepared
assets; this migration preserves those settings. Query-selection/horizon
changes require compatible cohort manifests and separate experiment directories.
The launcher stores full content-addressed inputs under
`artifacts/experiment-configs/`; reports retain effective settings.
Former flat profiles are preserved in [`../legacy/configs/`](../legacy/configs/).

The new `p1/{tworooms,reacher,cube}.json` profiles inherit `source-base.json` and
prepare only retrieval memory. Use `bash codes/run.sh preprocess inspect` then
`bash codes/run.sh preprocess prepare` for archive extraction, combined storage
checks and sequential P1 encoding/indexing. See the [preprocessing guide](../../docs/stages/preprocessing.md).
