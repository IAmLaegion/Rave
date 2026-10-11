# Rave benchmarks

The bar to beat: any engine change has to match or beat these before it ships.

| # | File | What | Settings |
|---|---|---|---|
| 1 | `benchmark1-fire-x-runaway-earworm.*` | Fire x Runaway, DJ Earworm engine (user: "almost perfect") | Runaway first and base, 126 BPM, Fire sped up x1.26, -3 st; pre-lyrics build (job 114b0f244449) |
| 2 | `benchmark2-fire-x-runaway-stems-earworm.*` | Fire x Runaway from stem packs, DJ Earworm engine (user: "near perfect") | Fire base 100 BPM, Runaway slowed x0.794, least-damage key Bbm (Fire -3 st, Runaway 0), 125 s (job 03b5c8d4f2cd) |

**Retired (2026-10-11, user):** the two Fire x Runaway benchmarks above are no longer the bar. The benchmark is
now the reference corpus: the 40+ surveyed mashups/nightcore/8D tracks (`dev/survey/survey.json`) plus the four
decoded references (`dev/refs/`). Renders are scored against the reference distributions for their style.
