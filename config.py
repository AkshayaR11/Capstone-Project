import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@localhost:5432/judicial")
MAPPINGS_FILE_PATH = os.getenv("MAPPINGS_FILE_PATH", "data/mappings.json")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:5173").split(",")
