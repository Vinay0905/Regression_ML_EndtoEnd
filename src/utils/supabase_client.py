
import os
from supabase import create_client, Client

def get_supabase_client() -> Client:
    url: str = os.environ.get("SUPABASE_URL", "")
    key: str = os.environ.get("SUPABASE_KEY", "")
    
    if not url or not key:
        raise ValueError("Supabase credentials not found in environment variables (SUPABASE_URL, SUPABASE_KEY)")

    return create_client(url, key)
