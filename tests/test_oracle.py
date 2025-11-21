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
        # Force MockModel by passing None or invalid path
        oracle = Oracle(model_path=None)
        # Mock the model to return high probability
        oracle.model.predict_proba = MagicMock(return_value=np.array([[0.1, 0.9]]))


        oracle.model.predict_proba = MagicMock(return_value=np.array([[0.1, 0.9]]))


        features = {
            "z_score": 2.5,
            "rsi": 70,
            "volatility": 0.001,
            "adx": 25,
            "time_sin": 0.5,
            "volume_delta": 0.1
        }
        action = oracle.predict(features)


        self.assertEqual(action, "EXECUTE")

    def test_predict_no_action(self):
        oracle = Oracle(model_path=None)
        # Mock the model to return low probability
        oracle.model.predict_proba = MagicMock(return_value=np.array([[0.8, 0.2]]))


        oracle.model.predict_proba = MagicMock(return_value=np.array([[0.8, 0.2]]))


        features = {
            "z_score": 0.5,
            "rsi": 50,
            "volatility": 0.001,
            "adx": 15,
            "time_sin": -0.5,
            "volume_delta": -0.1
        }
        action = oracle.predict(features)


        self.assertEqual(action, "NO_ACTION")

    def test_real_model_loading(self):
        # Should load the model we just trained
        oracle = Oracle()
        if isinstance(oracle.model, MockModel):
            self.skipTest("Real model not found, skipping")

        self.assertNotIsInstance(oracle.model, MockModel)
        self.assertNotIsInstance(oracle.model, MockModel)
        # Test prediction with real model
        features = {
            "z_score": 2.5,
            "rsi": 70,
            "volatility": 0.001,
            "adx": 30,
            "time_sin": 0.8,
            "volume_delta": 0.2
        }
        action = oracle.predict(features)
        self.assertIn(action, ["EXECUTE", "NO_ACTION"])



if __name__ == '__main__':
    unittest.main()
