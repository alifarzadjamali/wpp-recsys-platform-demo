# Retailrocket run

This run used all 2,756,101 events from the [Retailrocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset). The time split put 1,929,270 events in training, 413,415 in validation, and 413,416 in test. The catalog contained 450,268 candidates: 200,975 appeared in training and the rest had metadata available before the cutoff.

V2 and V3 each trained for three epochs on the same fixed sample of up to 300,000 next-item pairs. They ran on the local GPU. Evaluation ranked the entire frozen catalog for up to 300 users in each cohort. The exact settings are in `artifacts/run.json` when you run the project locally.

## Held-out test results at 10

| Group | Model | Users | Recall | NDCG |
| --- | --- | ---: | ---: | ---: |
| All sampled users | V1 baseline | 298 | 0.01091 | 0.00475 |
| All sampled users | V2 neural | 298 | 0.01007 | 0.00339 |
| All sampled users | V3 HSTU-style | 298 | 0.01007 | 0.00347 |
| Warm users | V1 baseline | 264 | 0.02155 | 0.01355 |
| Warm users | V2 neural | 264 | 0.00343 | 0.00320 |
| Warm users | V3 HSTU-style | 264 | 0.00126 | 0.00051 |
| New users after one event | V1 baseline | 191 | 0.26091 | 0.15766 |
| New users after one event | V2 neural | 191 | 0.03578 | 0.01878 |
| New users after one event | V3 HSTU-style | 191 | 0.03334 | 0.01666 |
| Cold items | All three | 300 | 0 | 0 |

V1 is the winner **under this training budget and split** for users with history. The all-user row is too close to make a strong claim: V1 found four hits among 298 sampled users, and each neural model found three. Each model found one hit among 300 zero-history new users because all three use the same popularity fallback. None retrieved a cold item in the sampled cold-item cohort.

The frozen catalog could not serve 25,196 of the 413,416 test events (about 6.1%). Those events are counted in `unavailable_events` in the full CSV but are not ranking targets. The test population is very sparse, and these results are offline ranking scores. They do not measure clicks caused by recommendations or long-term reward. The neural models also saw only a sampled set of training pairs; more tuning or training may change their ranking.

The complete local report is `artifacts/comparison.csv`. It includes precision, MAP, and catalog coverage for every group, plus validation rows. The repository ignores `artifacts/` so the dataset and generated files stay local.
