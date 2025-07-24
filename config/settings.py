from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

DATA_PATHS = {
    'fraud_data': DATA_DIR / "raw/fraud_data.csv",
    'ip_data': DATA_DIR / "raw/ipaddress_to_country.csv",
    'credit_data': DATA_DIR / "raw/creditcard.csv"
}
