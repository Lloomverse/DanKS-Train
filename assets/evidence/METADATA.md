# Open-source evidence pack

Numbers used by the README charts under `assets/online-training-*.png`.

## Throughput comparison

| Label | env step / s | Notes |
| --- | ---: | --- |
| Naive multi-table online | 225.19 | single-update wall-clock baseline |
| Current colocated online | ~4343 | warm-run median |

Hardware specs and calendar dates of internal training runs are omitted on purpose.

## Strength check (200-game self-play)

File: `eval_init_vs_latest_200.json` (checkpoint paths redacted)

- Result: continued online weight win rate **57.5%** vs init **42.5%**
- Chart: `assets/online-training-strength.png`

## v0.1.1 vs v0.1.0 (200-game self-play)

File: `eval_v011_vs_v010_200.json`

- Opponent: released `v0.1.0`
- Seed 2026: `v0.1.1` win rate **55.0%**, **+0.32** levels per game
- Seed 2027: `v0.1.1` win rate **56.5%**, **+0.40** levels per game

## Public weight naming

Current release tag: `v0.1.1`. The asset name stays date-free. `v0.1.0` remains published.

Release assets use a date-free name, for example:

`top10_selector_v3_online.pt`

Do not publish internal filenames that embed experiment stamps or host configs.

## Scripts

```bash
python scripts/export_online_metrics.py \
  --log /path/to/online.log \
  --out-csv assets/evidence/online_training_metrics.csv \
  --stride 20
python scripts/plot_open_source_evidence.py \
  --metrics-csv assets/evidence/online_training_metrics.csv \
  --out-dir assets \
  --baseline-steps-per-sec 225.19 \
  --baseline-label $'Naive online\n(tables)' \
  --current-label $'Current online\n(colocated+PPO)'
```
