# Elo-Poisson vs Dixon-Coles: a fair dynamic comparison

Date: 2026-07-13 · Command:

```bash
worldcup-predictor compare-dixon-coles \
  --matches data/raw/international_results.csv \
  --cutoff 2024-01-01 \
  --refit-days 30 \
  --output data/processed/model_comparison_dc.csv
```

## Protocol

A static Dixon-Coles fit frozen at the cutoff would go stale over a
2.5-year test window while Elo keeps updating after every match, so the
comparison keeps both models equally dynamic. Neither model ever sees a
match before predicting it:

- **Elo-Poisson**: fitted on all matches before the cutoff; Elo ratings then
  update after each test match (production protocol; Poisson mapping
  parameters stay at their cutoff fit).
- **Dixon-Coles**: refit every 30 days on a rolling 10-year window
  (26 refits across the test period), predictions frozen between refits.

Test set: 2,644 international matches, 2024-01-01 → 2026-07-11 (includes
the 2026 World Cup group stage and knockouts through the quarter-finals).
Training before cutoff: 46,861 matches.

## Results

| Metric (lower is better unless noted) | Elo-Poisson | Dixon-Coles | Winner |
|---|---|---|---|
| RPS (primary) | 0.16693 | **0.16229** | Dixon-Coles (−2.8%) |
| Log Loss | 0.86652 | **0.84817** | Dixon-Coles |
| Brier Score | 0.50917 | **0.49680** | Dixon-Coles |
| Outcome accuracy (higher) | 60.67% | 60.70% | tie |
| Mean predicted draw probability | 21.65% | **23.04%** | Dixon-Coles |
| Actual draw rate | 23.68% | 23.68% | — |

## Conclusions

1. **Dixon-Coles wins on every proper scoring rule.** The gap is consistent
   across RPS, log loss and Brier — this is the roadmap's Upgrade 1
   delivering exactly what it was designed for.
2. **The draw under-estimation is essentially fixed.** Independent Poisson
   systematically under-prices draws (21.7% predicted vs 23.7% actual);
   the ρ low-score correction closes most of that gap (23.0%).
3. Accuracy is unchanged, as expected — argmax rarely differs; the gains
   are in probability calibration, which is what the simulator consumes.

## Using Dixon-Coles for the tournament simulation

```bash
worldcup-predictor train-dixon-coles \
  --matches data/raw/international_results.csv \
  --since 2016-07-01 \
  --output models/dixon_coles_current.json

worldcup-predictor simulate \
  --model models/dixon_coles_current.json \
  --simulations 10000 \
  --output data/processed/simulation_2026_dc.csv
```

`simulate` dispatches on the `model_version` stored in the file, and
`catch-up` refits a Dixon-Coles model file in place (rolling 10-year
window) just like it does for Elo-Poisson, so both stay fresh.

### Final-four forecasts side by side (2026-07-13, semifinal eve)

Both conditioned on all 28 real knockout results; 10k/5k simulations.

| Team | Elo-Poisson champion % | Dixon-Coles champion % |
|---|---|---|
| Spain | 31.5% | 32.3% |
| Argentina | 31.4%* | 28.5% |
| France | 24.2% | 19.4% |
| England | 16.3% | 19.8% |

*Elo run used seed 42 with 10k simulations; DC used 5k. The models agree
on Spain/Argentina as favourites and differ mainly on France vs England.

**Recommendation**: Dixon-Coles is now the better-calibrated match model
and is fully wired into the simulator; Elo-Poisson remains the default for
continuity of the published forecast series during this tournament.
