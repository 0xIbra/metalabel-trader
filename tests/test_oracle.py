import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.oracle.main import Oracle
from src.oracle.mock_model import MockModel

class TestOracle(unittest.TestCase):
    def test_mock_model_fallback(self):
        # Initialize without model path should use MockModel
        oracle = Oracle(model_path="non_existent.json")
        self.assertIsInstance(oracle.model, MockModel)

    def test_predict_execute(self):
        oracle = Oracle()
        # Mock the model to return high probability
        oracle.model.predict_proba = MagicMock(return_value=np.array([[0.1, 0.9]]))

        features = {"z_score": 2.5, "rsi": 70}
        action = oracle.predict(features)

        self.assertEqual(action, "EXECUTE")

    def test_predict_no_action(self):
        oracle = Oracle()
        # Mock the model to return low probability
        oracle.model.predict_proba = MagicMock(return_value=np.array([[0.8, 0.2]]))

        features = {"z_score": 0.5, "rsi": 50}
        action = oracle.predict(features)

        self.assertEqual(action, "NO_ACTION")

if __name__ == '__main__':
    unittest.main()
