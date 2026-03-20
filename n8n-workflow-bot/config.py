import os
from dotenv import load_dotenv

load_dotenv()

N8N_API_KEY = os.getenv("N8N_API_KEY", "")
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "https://ashleymartian.app.n8n.cloud").rstrip("/")
N8N_API_BASE = f"{N8N_BASE_URL}/api/v1"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

MAX_RETRIES = 3
EXECUTION_POLL_INTERVAL = 3   # seconds between execution status polls
EXECUTION_TIMEOUT = 60        # seconds to wait for execution to complete
