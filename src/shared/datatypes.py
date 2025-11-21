from dataclasses import dataclass
from datetime import datetime

@dataclass
class TickData:
    symbol: str
    bid: float
    ask: float
    timestamp: float  # Unix timestamp
    volume: float = 1.0 # Default volume

