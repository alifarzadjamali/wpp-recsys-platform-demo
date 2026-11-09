# Recommendation lab

A small place to explore event data and compare three recommenders. The WPP files are private and are not included. You can run the made-up demo data straight away, or use the public Retailrocket event log.

## Layout

| Folder | What lives there |
| --- | --- |
| `data/` | Readers, schema checks, time split, and a tiny demo dataset |
| `analytics/` | EDA tables and ranking metrics |
| `dashboard/` | Streamlit app |
| `models/` | V1 hybrid baseline, V2 neural two-tower, V3 sequence attention |
| `tests/` | Small tests for splits, tuning, ranking, and model output |

The models all score the same catalog at the same cutoff. The dashboard reads saved comparison results, so opening a page does not train three models.

## Try it

Use Python 3.11 or 3.12. From the repo root:

```bash
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
python run.py --source demo --epochs 3
streamlit run dashboard/app.py
```

The comparison lands in `artifacts/comparison.csv`. The demo data is invented, so its scores say nothing about real users.

## Use public data

[Retailrocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) is the closest public match we found. It has visitor views, add-to-cart events, transactions, item properties, and a category tree. Download its original files from Kaggle and put `events.csv`, `item_properties_part1.csv`, and `item_properties_part2.csv` in `data/retailrocket/`. Then run:

```bash
python run.py --source retailrocket --path data/retailrocket --max-events 100000 --epochs 3
```

Remove `--max-events` for the full event log. A small head sample is useful for checking the pipeline, but it is not a benchmark. Retailrocket is published under [CC BY-NC-SA 4.0](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset); check the license before commercial use. We do not ship a copy of it.

The WPP reader still works with a local parquet file:

```bash
python run.py --source wpp --path data/wpp.parquet --epochs 3
```

WPP needs `member_id_str`, `merchant`, and `createdAt`. Its optional `name`, `categories`, `is_anonymous`, and `session_id` columns are used when present. The WPP catalog is built only from events before the training cutoff. There is no public WPP data or WPP category map in this repo.

## What the three versions do

V1 is the baseline hybrid: latent factors from a sparse user-item matrix, co-visits, optional sessions, categories, popularity, and recency. V2 learns user and item towers from interaction histories and category features. V3 is a compact causal attention sequence model inspired by Meta's [HSTU research](https://arxiv.org/abs/2402.17152). It is a local comparison model, not Meta's production HSTU implementation. [Meta's code](https://github.com/meta-recsys/generative-recommenders) has a much larger training and serving stack.

The report includes precision, recall, MAP, NDCG, and catalog coverage at 10. It also has rows for new users, users with short histories, and items with no training events. The `new_user` row has zero history, so all three models fall back to popularity. `new_user_after_one` gives the model that user's first held-out event and tests it on later events. Cold items can use category metadata if that metadata existed before the cutoff. Items absent from the frozen catalog cannot be ranked; their held-out event count appears as `unavailable_events`.

We use a global time split: 70% train, 15% validation, 15% test. Training uses only the first slice. The dashboard's weight sliders evaluate V1 on validation and do not alter the saved comparison. Full-catalog ranking excludes items already seen in training. Results depend on the dataset, cutoff, and training budget; the newer model is not guaranteed to win.

## Data and license notes

This code is MIT licensed. That license does not cover WPP data or external datasets. Older Git history may contain data-derived notebook outputs, so review the history before sharing a fork. See [the data notes](docs/data.md) for the schema mapping and limits of the public replacement.
