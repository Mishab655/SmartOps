import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

db_user = os.getenv("DB_USER", "postgres")
db_password = os.getenv("DB_PASSWORD")          
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "SmartOps")

if not db_password:
    raise EnvironmentError("DB_PASSWORD is not set.")

DB_URI = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
