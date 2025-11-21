import logging
import numpy as np
import os
try:
    import xgboost as xgb
except ImportError:
    xgb = None

from src.oracle.mock_model import MockModel

logger = logging.getLogger("Oracle")

class Oracle:
    def __init__(self, model_path: str = None):
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = xgb.Booster()
                self.model.load_model(model_path)
                logger.info(f"Loaded XGBoost model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")

        if self.model is None:
            logger.warning("No valid model found. Using MockModel.")
            self.model = MockModel()

    def predict(self, features: dict) -> str:
        """
        Predict action based on features.
        Features dict expected to contain keys like 'z_score', 'rsi', etc.
        """
        # Convert features dict to array expected by model
        # This mapping needs to match training data exactly
        # For now, we'll just pass a dummy array to the mock model

        # Example feature vector construction (placeholder)
        feature_vector = np.array([[
            features.get("z_score", 0),
            features.get("rsi", 50),
            features.get("volatility", 0)
        ]])

        # XGBoost expects DMatrix usually, or numpy array for scikit-learn API
        # If using native Booster:
        if isinstance(self.model, xgb.Booster):
            dtest = xgb.DMatrix(feature_vector)
            probs = self.model.predict(dtest)
            # Native predict returns raw scores or probs depending on objective
            # Assuming binary classification prob
            prob = probs[0]
        else:
            # Mock Model
            probs = self.model.predict_proba(feature_vector)
            prob = probs[0][1] # Probability of class 1 (Action)

        logger.info(f"Inference Probability: {prob:.4f}")

        if prob > 0.85:
            return "EXECUTE"
        else:
            return "NO_ACTION"
