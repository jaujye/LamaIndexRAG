"""
Vector index builder for legal documents
Creates and manages LlamaIndex indices with ChromaDB storage
"""

import os
import json
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext, Settings as LlamaSettings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from llama_index.core.schema import TextNode

from .document_processor import LegalDocumentProcessor, LegalChunk
from .monitoring import WandbMonitor, RAGMetrics, get_memory_usage, monitor_execution_time


class LegalIndexBuilder:
    """Builds and manages vector indices for legal documents with monitoring"""

    def __init__(self,
                 api_key: Optional[str] = None,
                 chroma_path: str = None,
                 collection_name: str = "food_safety_act",
                 embedding_model: str = "text-embedding-3-small",
                 enable_monitoring: bool = True,
                 monitor: Optional[WandbMonitor] = None):
        """
        Initialize the index builder

        Args:
            api_key: OpenAI API key (if not provided, loads from .env)
            chroma_path: Path to ChromaDB database or HTTP URL for remote ChromaDB
            collection_name: Name for the vector collection
            embedding_model: OpenAI embedding model to use
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        # Get chroma path from environment or parameter
        self.chroma_path_str = chroma_path or os.getenv("CHROMA_DB_PATH", "./chroma_db")

        # Determine if it's a local path or remote URL
        self.is_remote = self.chroma_path_str.startswith(('http://', 'https://'))

        if not self.is_remote:
            self.chroma_path = Path(self.chroma_path_str)
        else:
            self.chroma_path = None  # Not needed for remote connections
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Initialize monitoring FIRST (before other methods that use decorators)
        self.enable_monitoring = enable_monitoring
        self.monitor = monitor
        if self.enable_monitoring and not self.monitor:
            self.monitor = WandbMonitor(mode="disabled")  # Will check env for actual mode

        # Initialize components
        self._setup_llama_settings()
        self._setup_chroma_client()

        self.processor = LegalDocumentProcessor()
        self.index: Optional[VectorStoreIndex] = None
        self.vector_store: Optional[ChromaVectorStore] = None

    def _parse_host_from_url(self, url: str) -> str:
        """Extract host from URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.hostname or 'localhost'

    def _parse_port_from_url(self, url: str) -> int:
        """Extract port from URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.port or 8000

    def _setup_llama_settings(self):
        """Configure LlamaIndex global settings"""
        LlamaSettings.embed_model = OpenAIEmbedding(
            api_key=self.api_key,
            model=self.embedding_model
        )

        LlamaSettings.llm = OpenAI(
            api_key=self.api_key,
            model="gpt-3.5-turbo",
            temperature=0.1
        )

    @monitor_execution_time("chroma_setup_time")
    def _setup_chroma_client(self):
        """Initialize ChromaDB client with monitoring"""
        setup_start = time.time()

        try:
            if self.is_remote:
                # Remote ChromaDB connection
                host = self._parse_host_from_url(self.chroma_path_str)
                port = self._parse_port_from_url(self.chroma_path_str)

                print(f"Connecting to remote ChromaDB at: {host}:{port}")

                # First, test basic connectivity
                import requests
                heartbeat_response = requests.get(f"http://{host}:{port}/api/v2/heartbeat", timeout=10)
                if heartbeat_response.status_code != 200:
                    raise ConnectionError(f"ChromaDB heartbeat failed with status {heartbeat_response.status_code}")

                print("[OK] ChromaDB server is responsive")

                # Try to create client - attempt multiple approaches for compatibility
                client_created = False

                # Approach 1: Try with explicit tenant/database (newer versions)
                try:
                    self.chroma_client = chromadb.HttpClient(
                        host=host,
                        port=port,
                        tenant="default_tenant",
                        database="default_database",
                        settings=Settings(
                            anonymized_telemetry=False,
                            allow_reset=True
                        )
                    )
                    client_created = True
                    print("[OK] Connected with tenant/database specification")
                except Exception as tenant_error:
                    print(f"[WARN] Tenant/database approach failed: {tenant_error}")

                # Approach 2: Try without tenant specification (older versions)
                if not client_created:
                    try:
                        self.chroma_client = chromadb.HttpClient(
                            host=host,
                            port=port,
                            settings=Settings(
                                anonymized_telemetry=False,
                                allow_reset=True
                            )
                        )
                        client_created = True
                        print("[OK] Connected without tenant specification")
                    except Exception as no_tenant_error:
                        print(f"[WARN] No-tenant approach failed: {no_tenant_error}")

                # Approach 3: Simple HttpClient (most basic)
                if not client_created:
                    try:
                        self.chroma_client = chromadb.HttpClient(
                            host=host,
                            port=port
                        )
                        client_created = True
                        print("[OK] Connected with basic HttpClient")
                    except Exception as basic_error:
                        print(f"[ERROR] Basic HttpClient failed: {basic_error}")
                        raise basic_error

                print("[OK] ChromaDB HttpClient created successfully")

            else:
                # Local ChromaDB connection
                self.chroma_path.mkdir(parents=True, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.chroma_path),
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                print(f"[OK] Using local ChromaDB at: {self.chroma_path}")

            setup_time = time.time() - setup_start

            # Log setup metrics
            if self.monitor:
                self.monitor.log_metrics({
                    "chroma_setup_time": setup_time,
                    "chroma_connection_type": "remote" if self.is_remote else "local",
                    "chroma_setup_successful": True,
                    "memory_usage_after_setup": get_memory_usage()
                })

        except Exception as e:
            setup_time = time.time() - setup_start

            # Log setup failure
            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={
                        "chroma_setup_time": setup_time,
                        "chroma_connection_type": "remote" if self.is_remote else "local",
                        "chroma_setup_successful": False
                    }
                )

            # If remote connection failed, fallback to local
            if self.is_remote:
                print(f"[ERROR] Failed to connect to remote ChromaDB: {e}")
                print("[WARN] Falling back to local ChromaDB mode...")
                self.is_remote = False
                self.chroma_path_str = "./chroma_db"
                self.chroma_path = Path(self.chroma_path_str)
                self._setup_chroma_client()  # Retry with local setup
            else:
                raise

    def create_collection(self, reset: bool = False) -> chromadb.Collection:
        """Create or get ChromaDB collection"""
        if reset:
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                print(f"Reset collection: {self.collection_name}")
            except ValueError:
                pass  # Collection doesn't exist

        collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "law_name": "食品安全衛生管理法",
                "law_code": "L0040001",
                "embedding_model": self.embedding_model
            }
        )

        return collection

    @monitor_execution_time("index_build_total_time")
    def build_index_from_json(self, json_path: str, reset: bool = False) -> VectorStoreIndex:
        """
        Build vector index from processed legal JSON data with monitoring

        Args:
            json_path: Path to the JSON file with legal data
            reset: Whether to reset existing collection

        Returns:
            VectorStoreIndex ready for querying
        """
        build_start = time.time()
        print(f"Loading legal data from {json_path}...")

        try:
            # Load and process documents
            load_start = time.time()
            data = self.processor.load_legal_data(json_path)
            chunks = self.processor.process_all_articles(data)
            process_time = time.time() - load_start

            print(f"Processed {len(chunks)} chunks from {len(data['articles'])} articles")

            # Convert to LlamaIndex documents
            convert_start = time.time()
            documents = self.processor.convert_to_llama_documents(chunks)
            convert_time = time.time() - convert_start

            # Create ChromaDB collection
            collection_start = time.time()
            collection = self.create_collection(reset=reset)
            collection_time = time.time() - collection_start

            # Setup vector store
            self.vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # Build index
            print("Building vector index...")
            index_start = time.time()
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            index_time = time.time() - index_start

            # Save index metadata
            metadata_start = time.time()
            self._save_index_metadata(data, chunks)
            metadata_time = time.time() - metadata_start

            total_time = time.time() - build_start

            print(f"Successfully built index with {len(documents)} documents")

            # Log comprehensive metrics
            if self.monitor:
                # Calculate average chunk size
                avg_chunk_size = sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0

                metrics = RAGMetrics(
                    documents_processed=len(data['articles']),
                    chunks_created=len(chunks),
                    embedding_dimensions=1536,  # text-embedding-3-small dimensions
                    index_build_time=total_time
                )

                detailed_metrics = {
                    "data_loading_time": process_time,
                    "document_conversion_time": convert_time,
                    "collection_creation_time": collection_time,
                    "index_building_time": index_time,
                    "metadata_save_time": metadata_time,
                    "total_articles": len(data['articles']),
                    "total_chunks": len(chunks),
                    "total_documents": len(documents),
                    "avg_chunk_size_chars": avg_chunk_size,
                    "reset_collection": reset,
                    "embedding_model": self.embedding_model,
                    "memory_usage_after_build": get_memory_usage()
                }

                self.monitor.log_metrics(metrics)
                self.monitor.log_metrics(detailed_metrics)

            return self.index

        except Exception as e:
            total_time = time.time() - build_start

            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={
                        "index_build_time": total_time,
                        "json_path": json_path,
                        "reset": reset,
                        "build_stage": "index_building"
                    }
                )

            raise

    def build_index_from_chunks(self, chunks: List[LegalChunk], reset: bool = False) -> VectorStoreIndex:
        """
        Build vector index from processed legal chunks

        Args:
            chunks: List of processed legal chunks
            reset: Whether to reset existing collection

        Returns:
            VectorStoreIndex ready for querying
        """
        print(f"Building index from {len(chunks)} chunks...")

        # Convert to LlamaIndex documents
        documents = self.processor.convert_to_llama_documents(chunks)

        # Create ChromaDB collection
        collection = self.create_collection(reset=reset)

        # Setup vector store
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Build index
        print("Building vector index...")
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

        print(f"Successfully built index with {len(documents)} documents")
        return self.index

    @monitor_execution_time("index_load_time")
    def load_existing_index(self) -> Optional[VectorStoreIndex]:
        """Load existing index from ChromaDB with monitoring"""
        load_start = time.time()

        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)

            # Check if collection has data
            document_count = collection.count()
            if document_count == 0:
                print("Collection exists but is empty")

                if self.monitor:
                    self.monitor.log_metrics({
                        "index_load_successful": False,
                        "index_load_reason": "empty_collection",
                        "collection_document_count": 0
                    })

                return None

            print(f"Loading existing collection with {document_count} documents")

            # Setup vector store and load index
            self.vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=storage_context
            )

            load_time = time.time() - load_start

            # Log successful load metrics
            if self.monitor:
                self.monitor.log_metrics({
                    "index_load_time": load_time,
                    "index_load_successful": True,
                    "collection_document_count": document_count,
                    "collection_name": self.collection_name,
                    "memory_usage_after_load": get_memory_usage()
                })

            return self.index

        except ValueError as e:
            load_time = time.time() - load_start
            print(f"Collection not found: {e}")

            if self.monitor:
                self.monitor.log_metrics({
                    "index_load_time": load_time,
                    "index_load_successful": False,
                    "index_load_reason": "collection_not_found",
                    "collection_name": self.collection_name
                })

            return None

        except Exception as e:
            load_time = time.time() - load_start

            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={
                        "index_load_time": load_time,
                        "collection_name": self.collection_name,
                        "operation": "load_existing_index"
                    }
                )

            raise

    def _save_index_metadata(self, data: Dict[str, Any], chunks: List[LegalChunk]):
        """Save metadata about the index"""
        if not self.is_remote:
            metadata_path = self.chroma_path / "index_metadata.json"
        else:
            # For remote connections, save metadata in current directory
            metadata_path = Path("index_metadata.json")

        stats = self.processor.get_processing_stats(chunks)

        metadata = {
            "law_name": data["law_name"],
            "law_code": data["law_code"],
            "source_url": data["source_url"],
            "total_articles": data["total_articles"],
            "embedding_model": self.embedding_model,
            "collection_name": self.collection_name,
            "processing_stats": stats,
            "created_at": str(Path(__file__).stat().st_mtime),
            "is_remote": self.is_remote,
            "chroma_connection": self.chroma_path_str if self.is_remote else str(self.chroma_path)
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        if not self.vector_store:
            return {"error": "No index loaded"}

        collection = self.vector_store._collection

        stats = {
            "collection_name": self.collection_name,
            "document_count": collection.count(),
            "embedding_model": self.embedding_model,
            "is_remote": self.is_remote,
            "chroma_connection": self.chroma_path_str if self.is_remote else str(self.chroma_path)
        }

        # Load metadata if available
        if not self.is_remote:
            metadata_path = self.chroma_path / "index_metadata.json"
        else:
            metadata_path = Path("index_metadata.json")

        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                stats.update(metadata)

        return stats

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents (for testing)"""
        if not self.index:
            raise ValueError("Index not loaded. Call build_index_from_json() or load_existing_index() first.")

        # Create a query engine for similarity search
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="no_text"  # Only return source nodes
        )

        response = query_engine.query(query)

        results = []
        for node in response.source_nodes:
            results.append({
                "text": node.node.text,
                "metadata": node.node.metadata,
                "score": node.score
            })

        return results

    def update_documents(self, new_chunks: List[LegalChunk]):
        """Update index with new documents"""
        if not self.index:
            raise ValueError("Index not loaded")

        new_documents = self.processor.convert_to_llama_documents(new_chunks)

        # Insert new documents
        for doc in new_documents:
            self.index.insert(doc)

        print(f"Added {len(new_documents)} new documents to index")

    def delete_collection(self):
        """Delete the ChromaDB collection"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except ValueError as e:
            print(f"Collection not found: {e}")


def main():
    """Example usage"""
    try:
        # Initialize builder
        builder = LegalIndexBuilder(enable_monitoring=False)

        # Check if data file exists
        data_file = "data/food_safety_act.json"
        if not Path(data_file).exists():
            print(f"Data file not found: {data_file}")
            print("Run the data fetcher first: python -m src.data_fetcher")
            return

        # Try to load existing index
        index = builder.load_existing_index()

        if not index:
            print("Building new index...")
            index = builder.build_index_from_json(data_file, reset=True)
        else:
            print("Loaded existing index")

        # Show stats
        stats = builder.get_index_stats()
        print(f"\nIndex Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        # Test similarity search
        test_query = "食品添加物的規定"
        print(f"\nTesting similarity search with query: '{test_query}'")
        results = builder.search_similar(test_query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.3f}):")
            print(f"Article: {result['metadata']['article_number']}")
            print(f"Type: {result['metadata']['chunk_type']}")
            print(f"Preview: {result['text'][:200]}...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()