import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from data.load import normalize, split_time, load_retailrocket
from models.common import ModelData, rank
from models.v1_baseline import Baseline
from models.v2_neural import NeuralTwoTower
from models.v3_sequence import SequenceAttention
from analytics.evaluate import evaluate


class PipelineTest(unittest.TestCase):
    def setUp(self):
        events = pd.DataFrame({
            "user_id": ["a", "b"] * 8 + ["new", "a", "b", "new"],
            "item_id": ["1", "2", "2", "3", "3", "1", "1", "2", "2", "3", "3", "1", "1", "3", "2", "1", "4", "4", "2", "3"],
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC"),
        })
        catalog = pd.DataFrame({"item_id": ["1", "2", "3", "4"], "category": ["x", "y", "x", "x"], "name": ["One", "Two", "Three", "Four"]})
        self.events, self.catalog = normalize(events, catalog)
        self.train, self.valid, self.test = split_time(self.events)
        self.data = ModelData(self.train, self.catalog)

    def test_split_and_catalog(self):
        self.assertLess(self.train.timestamp.max(), self.valid.timestamp.min())
        self.assertEqual(self.data.popularity[self.data.item_index["4"]], 0)
        self.assertNotIn("new", self.data.user_index)

    def test_baseline_weights_and_seen_filter(self):
        model = Baseline().fit(self.data)
        original = model.score("a")
        model.weights = {"latent": 0, "co_visit": 0, "session": 0, "category": 0, "popular": 1, "recency": 0}
        self.assertFalse(np.allclose(original, model.score("a")))
        self.assertTrue(set(rank(original, self.data.seen["a"], 2)).isdisjoint(self.data.seen["a"]))
        report = evaluate(model, self.data, self.test, k=2)
        self.assertIn("new_user", set(report.group))
        self.assertIn("cold_item", set(report.group))
        held = self.test.iloc[:2].copy()
        held["user_id"] = "brand_new"
        held["item_id"] = ["1", "4"]
        one_shot = evaluate(model, self.data, held, k=2)
        self.assertIn("new_user_after_one", set(one_shot.group))

    def test_neural_models_score_full_catalog(self):
        for model in (NeuralTwoTower(dim=8, epochs=1), SequenceAttention(dim=8, epochs=1)):
            model.fit(self.data)
            self.assertEqual(len(model.score("a")), len(self.catalog))
            self.assertTrue(np.isfinite(model.score("new")).all())

    def test_retailrocket_catalog_stops_at_cutoff(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            pd.DataFrame({"timestamp": np.arange(10) * 86400000, "visitorid": [1] * 10,
                          "event": ["view"] * 10, "itemid": [1] * 7 + [2] * 3}).to_csv(base / "events.csv", index=False)
            pd.DataFrame({"timestamp": [0, 9 * 86400000], "itemid": [1, 2],
                          "property": ["categoryid", "categoryid"], "value": ["a", "b"]}).to_csv(base / "item_properties_part1.csv", index=False)
            events, catalog = load_retailrocket(base)
            self.assertEqual(len(events), 10)
            self.assertEqual(catalog.item_id.tolist(), ["1"])
            self.assertIn("session_id", events)


if __name__ == "__main__":
    unittest.main()
