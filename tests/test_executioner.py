import unittest
from src.executioner.main import MockExecutioner
from src.executioner.risk_manager import RiskManager

class TestExecutioner(unittest.TestCase):
    def test_risk_manager_position_size(self):
        rm = RiskManager()
        # Balance 10000, Risk 1% (0.01), SL 10 pips, Pip Value $10
        # Risk Amount = 100
        # Lots = 100 / (10 * 10) = 1.0
        lots = rm.calculate_position_size(10000, 0.01, 10)
        self.assertEqual(lots, 1.0)

    def test_risk_manager_max_lots(self):
        rm = RiskManager(max_lots=5.0)
        rm.open_lots = 4.0

        # Try to add 2 lots -> Should fail
        allowed = rm.check_risk(10000, 10000, 2.0)
        self.assertFalse(allowed)

        # Try to add 0.5 lots -> Should pass
        allowed = rm.check_risk(10000, 10000, 0.5)
        self.assertTrue(allowed)

    def test_mock_executioner_order(self):
        exec = MockExecutioner(initial_balance=10000)

        # Place valid order
        result = exec.place_order("EURUSD", "BUY", 1.0, sl=1.0490, tp=1.0550)
        self.assertTrue(result)
        self.assertEqual(len(exec.positions), 1)

        # Place invalid order (too large)
        result = exec.place_order("EURUSD", "BUY", 10.0)
        self.assertFalse(result)
        self.assertEqual(len(exec.positions), 1)

if __name__ == '__main__':
    unittest.main()
