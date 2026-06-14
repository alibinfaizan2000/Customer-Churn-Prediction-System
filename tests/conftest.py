import os
from pathlib import Path

# load .env locally if it exists (local development)
# in CI, variables come from GitHub Secrets directly
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)
