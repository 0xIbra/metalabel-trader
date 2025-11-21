import os
from dotenv import load_dotenv

load_dotenv()

EODHD_API_KEY = os.getenv("EODHD_API_KEY")
if not EODHD_API_KEY:
    raise ValueError("EODHD_API_KEY not found in environment variables")
