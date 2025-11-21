import numpy as np
import logging

logger = logging.getLogger("MockModel")

class MockModel:
    """
    A mock XGBoost model for testing purposes.
    """
    def __init__(self):
        logger.info("Initialized MockModel")

    def predict_proba(self, features):
        """
        Returns a dummy probability.
        If features is a 2D array, returns an array of probabilities.
        """
        # Simulate a random probability between 0 and 1
        # For consistency in testing, we could make it deterministic based on input
        # But for now, let's just return a high probability if z-score is high (simulated)

        # Assuming features is a numpy array or list
        # Let's just return a random value for now, or 0.9 to trigger a signal for testing
        return np.array([[0.1, 0.9]]) # [prob_class_0, prob_class_1]
