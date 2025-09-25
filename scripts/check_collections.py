"""
Check available ChromaDB collections
"""

import chromadb
import os
from dotenv import load_dotenv

def check_chroma_collections():
    """Check what collections exist in ChromaDB"""
    # Load environment variables
    load_dotenv()

    try:
        # Try remote first - use environment variables
        print("Checking remote ChromaDB...")
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        client = chromadb.HttpClient(host=host, port=port)
        client.heartbeat()
        print("[OK] Connected to remote ChromaDB")
    except Exception as e:
        print(f"[WARN] Remote ChromaDB failed: {e}")
        print("Using local ChromaDB...")
        client = chromadb.PersistentClient(path="chroma_db")

    try:
        collections = client.list_collections()
        print(f"\nFound {len(collections)} collections:")
        for collection in collections:
            try:
                count = collection.count()
                print(f"  - {collection.name}: {count} documents")
            except Exception as e:
                print(f"  - {collection.name}: Error getting count - {e}")

    except Exception as e:
        print(f"Error listing collections: {e}")

if __name__ == "__main__":
    check_chroma_collections()