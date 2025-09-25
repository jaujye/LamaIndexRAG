"""
Check available ChromaDB collections
"""

import chromadb

def check_chroma_collections():
    """Check what collections exist in ChromaDB"""
    try:
        # Try remote first
        print("Checking remote ChromaDB...")
        client = chromadb.HttpClient(host="192.168.0.114", port=7000)
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