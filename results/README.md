# Results Index

Start here if you are sharing or reviewing the subliminal-learning experiments.

| File | Purpose |
|---|---|
| [`shareable-summary.md`](shareable-summary.md) | Short summary suitable for sharing with collaborators. |
| [`subliminal-learning-consolidated-results.md`](subliminal-learning-consolidated-results.md) | Full validated consolidation across factual-transfer, owl preference, in-context, batch-invariant, partial/debug, and cluster/HPC artifacts. |
| [`initial-results.md`](initial-results.md) | Original factual-transfer experiment write-up. |

To regenerate the consolidated reports from local experiment artifacts:

```bash
python scripts/consolidate_subliminal_results.py
```

Large local sweep outputs, adapters, and trainer checkpoints are intentionally ignored by Git. Commit concise Markdown summaries and scripts; archive large artifacts separately if needed.
