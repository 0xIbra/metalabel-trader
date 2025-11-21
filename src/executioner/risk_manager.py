import logging

logger = logging.getLogger("RiskManager")

class RiskManager:
    def __init__(self, max_daily_loss_percent: float = 0.03, max_lots: float = 5.0):
        self.max_daily_loss_percent = max_daily_loss_percent
        self.max_lots = max_lots
        self.daily_loss = 0.0
        self.open_lots = 0.0

    def calculate_position_size(self, account_balance: float, risk_percent: float, stop_loss_pips: float, pip_value: float = 10.0) -> float:
        """
        Calculate position size in lots.
        Formula: (Account_Balance * Risk_Percent) / (Stop_Loss_Points * Point_Value)
        """
        if stop_loss_pips <= 0:
            logger.error("Stop loss pips must be positive")
            return 0.0

        risk_amount = account_balance * risk_percent
        # Assuming standard lot size where 1 pip = $10 (for EURUSD standard lot)
        # pip_value argument allows adjustment for mini/micro lots or other pairs

        # Risk Amount = Lots * Stop_Loss_Pips * Pip_Value
        # Lots = Risk_Amount / (Stop_Loss_Pips * Pip_Value)

        lots = risk_amount / (stop_loss_pips * pip_value)

        # Round to 2 decimal places (standard for MT5)
        lots = round(lots, 2)

        return lots

    def check_risk(self, current_equity: float, initial_balance: float, proposed_lots: float) -> bool:
        """
        Check if trade is allowed based on risk parameters.
        """
        # 1. Daily Loss Check
        current_loss = initial_balance - current_equity
        loss_percent = current_loss / initial_balance

        if loss_percent > self.max_daily_loss_percent:
            logger.warning(f"Daily loss limit reached: {loss_percent:.2%} > {self.max_daily_loss_percent:.2%}")
            return False

        # 2. Max Lots Check
        if self.open_lots + proposed_lots > self.max_lots:
            logger.warning(f"Max lots limit reached: {self.open_lots + proposed_lots} > {self.max_lots}")
            return False

        return True
