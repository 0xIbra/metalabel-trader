from abc import ABC, abstractmethod
import logging
import time
from src.executioner.risk_manager import RiskManager

logger = logging.getLogger("Executioner")

class Executioner(ABC):
    @abstractmethod
    def place_order(self, symbol: str, side: str, volume: float, sl: float = 0.0, tp: float = 0.0) -> bool:
        pass

    @abstractmethod
    def get_account_info(self) -> dict:
        pass

class MockExecutioner(Executioner):
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.equity = initial_balance
        self.risk_manager = RiskManager()
        self.positions = []
        logger.info(f"Initialized MockExecutioner with balance ${initial_balance}")

    def place_order(self, symbol: str, side: str, volume: float, sl: float = 0.0, tp: float = 0.0) -> bool:
        """
        Simulate placing an order.
        """
        # Risk Check
        if not self.risk_manager.check_risk(self.equity, self.balance, volume):
            logger.warning("Order rejected by Risk Manager")
            return False

        logger.info(f"MOCK ORDER PLACED: {side} {volume} {symbol} @ MKT | SL={sl} TP={tp}")

        # Simulate execution
        self.positions.append({
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "open_price": 1.0500, # Dummy price
            "sl": sl,
            "tp": tp,
            "time": time.time()
        })

        self.risk_manager.open_lots += volume
        return True

    def get_account_info(self) -> dict:
        return {
            "balance": self.balance,
            "equity": self.equity,
            "margin": 0.0, # Simplified
            "free_margin": self.equity
        }
