from unittest import TestCase
import numpy as np
import pandas as pd

from kangaroo.feature_selection.feature_selection import independent_categorical_features


class FeatureSelectionTest(TestCase):
    def setUp(self) -> None:
        possible_values_1 = ["A", "B", "C", "D"]
        probs_1 = [0.15, 0.25, 0.4, 0.2]
        values_1 = [np.random.choice(possible_values_1, p=probs_1) for _ in range(500)]
        possible_values_2 = ["0", "1", "2", "3", "4"]
        probs_2 = [0.10, 0.20, 0.35, 0.2, 0.15]
        values_2 = [np.random.choice(possible_values_2, p=probs_2) for _ in range(500)]
        self.data = pd.DataFrame({"c1": values_1, "c2": values_2})

    def test_duplicated_column(self):
        """When there is a duplicated column, the second one should not be returned."""
        pd_t = self.data
        pd_t["c3"] = pd_t["c2"]
        selected_features = independent_categorical_features(pd_t, 2)
        self.assertEqual(selected_features, ["c1", "c2"])

    def test_column_is_function_of_another(self):
        """When there is a column that is a function of another, the second one should not be returned."""
        pd_t = self.data
        pd_t["c3"] = pd_t["c1"].apply(lambda x: x + np.random.choice(["M", "N"], p=[0.5, 0.5]))
        selected_features = independent_categorical_features(pd_t, 2)
        self.assertEqual(selected_features, ["c1", "c2"])

