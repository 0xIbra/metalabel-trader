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
    def __init__(self, model_path: str = "src/oracle/model.json"):
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
        Features dict expected to contain 16 features matching training.
        Returns 3-class prediction remapped from {0,1,2} to {-1,0,1}
        Then applies confidence threshold for BUY signals.
        """
        # Feature vector matching training (16 features)
        feature_vector = np.array([[
            # Technical indicators
            features.get("z_score", 0),
            features.get("rsi", 50),
            features.get("volatility", 0),
            features.get("adx", 0),
            features.get("time_sin", 0),
            features.get("volume_delta", 0),
            # Momentum
            features.get("roc_5", 0),
            features.get("roc_10", 0),
            features.get("roc_20", 0),
            features.get("macd", 0),
            features.get("velocity", 0),
            # Lag features
            features.get("close_lag1", features.get("close", 0)),
            features.get("close_lag2", features.get("close", 0)),
            features.get("close_lag3", features.get("close", 0)),
            features.get("returns_lag1", 0),
            features.get("returns_lag2", 0)
        ]])

        feature_names = [
            'z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta',
            'roc_5', 'roc_10', 'roc_20', 'macd', 'velocity',
            'close_lag1', 'close_lag2', 'close_lag3', 'returns_lag1', 'returns_lag2'
        ]

        # Model is 3-class (labels 0, 1, 2 for SELL, NO_ACTION, BUY)
        if isinstance(self.model, xgb.Booster):
            dtest = xgb.DMatrix(feature_vector, feature_names=feature_names)
            probs = self.model.predict(dtest)  # Returns probabilities for each class

            # probs shape: (1, 3) - probabilities for [SELL, NO_ACTION, BUY]
            buy_prob = probs[0][2]  # Probability of BUY class (index 2)

            logger.info(f"BUY Probability: {buy_prob:.4f}")

            # Only trade BUY signals with >65% confidence
            if buy_prob > 0.65:
                return "EXECUTE"
            else:
                return "NO_ACTION"
        else:
            # Mock Model (fallback)
            return "NO_ACTION"

