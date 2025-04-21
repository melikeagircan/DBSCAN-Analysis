DB_CONFIG = {
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432',
    'database': 'GYK'
}

DB_URL = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# DBSCAN default parameters
DEFAULT_MIN_SAMPLES = 3
DEFAULT_EPS = 0.3 