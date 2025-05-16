import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_ENDPOINT_ID = os.getenv("DB_ENDPOINT_ID")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")
if not DB_NAME or not DB_USER or not DB_HOST or not DB_PORT:
    raise ValueError("Missing Database Credentials (DB_NAME, DB_USER, DB_HOST, DB_PORT) in .env file")