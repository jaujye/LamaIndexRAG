"""
Advanced RAG System with Hybrid Search, Query Expansion, and Reranking
Supports multiple knowledge bases (Food Safety Act, Labor Law)
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
try:
    from llama_index.embeddings.onnx import OnnxEmbedding
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
from dataclasses import dataclass
import re
from collections import defaultdict
import logging


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with scoring metadata"""
    node: NodeWithScore
    hybrid_score: float
    semantic_score: float
    keyword_score: float
    rerank_score: Optional[float] = None
    source_collection: str = ""


@dataclass
class QueryContext:
    """Query context with expansion and metadata"""
    original_query: str
    expanded_terms: List[str]
    legal_concepts: List[str]
    intent_type: str  # 'definition', 'procedure', 'penalty', 'general'
    target_collections: List[str]


class LegalQueryExpander:
    """Intelligent query expansion for legal documents"""

    def __init__(self):
        self.legal_synonyms = {
            '勞動契約': ['勞動契約', '工作契約', '僱傭契約', '聘僱關係'],
            '工資': ['工資', '薪資', '薪水', '報酬', '津貼'],
            '工作時間': ['工作時間', '工時', '勞動時間', '上班時間'],
            '休假': ['休假', '假期', '請假', '特別休假'],
            '解僱': ['解僱', '終止契約', '資遣', '開除'],
            '職業安全': ['職業安全', '工安', '勞工安全', '作業安全'],
            '食品安全': ['食品安全', '食安', '食品衛生', '食品品質'],
            '添加物': ['添加物', '食品添加劑', '化學添加物'],
            '標示': ['標示', '標籤', '包裝標示', '成分標示']
        }

        self.concept_patterns = {
            'definition': ['什麼是', '定義', '意思', '含義'],
            'procedure': ['如何', '程序', '流程', '步驟', '手續'],
            'penalty': ['罰則', '處罰', '罰金', '刑罰', '違反'],
            'rights': ['權利', '權益', '保障', '福利'],
            'obligations': ['義務', '責任', '應該', '必須']
        }

    def expand_query(self, query: str) -> QueryContext:
        """Expand query with legal synonyms and concepts"""
        expanded_terms = [query]
        legal_concepts = []
        intent_type = 'general'

        # Extract key terms using simple pattern matching
        keywords = self._extract_keywords_simple(query)

        # Expand with legal synonyms
        for keyword in keywords:
            if keyword in self.legal_synonyms:
                expanded_terms.extend(self.legal_synonyms[keyword])

        # Identify legal concepts
        for concept, indicators in self.concept_patterns.items():
            if any(indicator in query for indicator in indicators):
                legal_concepts.append(concept)
                if intent_type == 'general':
                    intent_type = concept

        # Determine target collections based on content
        target_collections = []
        if any(term in query for term in ['勞動', '勞工', '工資', '工時', '勞基法']):
            target_collections.append('labor_law')
        if any(term in query for term in ['食品', '食安', '添加物', '食品安全法']):
            target_collections.append('food_safety_act')
        if not target_collections:
            target_collections = ['labor_law', 'food_safety_act']  # Search both

        return QueryContext(
            original_query=query,
            expanded_terms=list(set(expanded_terms)),
            legal_concepts=legal_concepts,
            intent_type=intent_type,
            target_collections=target_collections
        )

    def _extract_keywords_simple(self, text: str) -> List[str]:
        """Simple keyword extraction for Chinese text without external dependencies"""
        # Common Chinese legal terms patterns
        legal_patterns = [
            r'勞動\w+',  # 勞動契約, 勞動條件
            r'工\w{1,2}',  # 工資, 工時, 工作
            r'食品\w+',  # 食品安全, 食品添加物
            r'\w*契約',   # 各種契約
            r'\w*標示',   # 標示相關
            r'\w*處罰',   # 處罰相關
            r'\w*規定',   # 規定相關
            r'\w*時間',   # 時間相關
            r'\w*休假',   # 休假相關
        ]

        keywords = []

        # Extract patterns
        for pattern in legal_patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)

        # Extract 2-4 character words (common Chinese word length)
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        keywords.extend(chinese_words)

        # Remove duplicates and return top 5
        unique_keywords = list(set(keywords))
        return unique_keywords[:5]


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining semantic and keyword search"""

    def __init__(
        self,
        vector_retrievers: Dict[str, BaseRetriever],
        chroma_client,  # Pass the properly configured ChromaDB client
        alpha: float = 0.7,  # Weight for semantic search
        top_k: int = 10
    ):
        super().__init__()
        self.vector_retrievers = vector_retrievers
        self.chroma_client = chroma_client  # Use the same client as main system
        self.alpha = alpha
        self.beta = 1.0 - alpha  # Weight for keyword search
        self.top_k = top_k
        self.query_expander = LegalQueryExpander()

    def keyword_search(self, query_context: QueryContext, collection_name: str) -> List[RetrievalResult]:
        """Perform keyword-based search using the same ChromaDB client and embedding model"""
        results = []

        # Use the properly configured ChromaDB client (same as main system)
        client = self.chroma_client

        try:
            collection = client.get_collection(collection_name)

            # Create search terms from expanded query
            search_terms = ' '.join(query_context.expanded_terms)

            # Use the same embedding model as the main system for consistent dimensions
            from llama_index.core import Settings
            embed_model = Settings.embed_model

            # Generate embedding using the configured 1536-dim model
            query_embedding = embed_model.get_text_embedding(search_terms)

            # Query ChromaDB with proper embedding
            chroma_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(self.top_k * 2, 20)  # Get more for filtering
            )

            # Convert to RetrievalResult format
            if chroma_results['documents']:
                for i, (doc, metadata, distance) in enumerate(zip(
                    chroma_results['documents'][0],
                    chroma_results['metadatas'][0],
                    chroma_results['distances'][0]
                )):
                    # Calculate keyword score based on term frequency
                    keyword_score = self._calculate_keyword_score(doc, query_context)

                    # Create proper TextNode for LlamaIndex compatibility
                    text_node = TextNode(
                        text=doc,
                        metadata=metadata,
                        id_=metadata.get('chunk_id', f'keyword_{i}')
                    )

                    # Create NodeWithScore with proper TextNode
                    node = NodeWithScore(
                        node=text_node,
                        score=1 - distance  # Convert distance to similarity
                    )

                    results.append(RetrievalResult(
                        node=node,
                        hybrid_score=0.0,  # Will be calculated later
                        semantic_score=1 - distance,
                        keyword_score=keyword_score,
                        source_collection=collection_name
                    ))

        except Exception as e:
            logging.warning(f"Keyword search failed for {collection_name}: {e}")

        return results

    def _calculate_keyword_score(self, text: str, query_context: QueryContext) -> float:
        """Calculate keyword matching score"""
        text_lower = text.lower()
        query_terms = [term.lower() for term in query_context.expanded_terms]

        # Count exact matches
        exact_matches = sum(1 for term in query_terms if term in text_lower)

        # Count partial matches (for compound terms)
        partial_matches = 0
        for term in query_terms:
            if len(term) > 2:  # Only consider terms longer than 2 characters
                term_chars = list(term)
                text_chars = list(text_lower)
                if any(char in text_chars for char in term_chars):
                    partial_matches += 0.5

        # Normalize by query length
        total_score = (exact_matches + partial_matches) / len(query_terms)
        return min(1.0, total_score)  # Cap at 1.0

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Main retrieval method combining semantic and keyword search"""
        query_context = self.query_expander.expand_query(query_bundle.query_str)

        all_results = []

        # Perform retrieval on target collections
        for collection_name in query_context.target_collections:
            if collection_name in self.vector_retrievers:
                # Semantic search
                try:
                    semantic_results = self.vector_retrievers[collection_name].retrieve(query_bundle)

                    # Convert to RetrievalResult format
                    for node_score in semantic_results:
                        result = RetrievalResult(
                            node=node_score,
                            hybrid_score=0.0,  # Will be calculated below
                            semantic_score=node_score.score or 0.0,
                            keyword_score=0.0,  # Will be calculated below
                            source_collection=collection_name
                        )

                        # Calculate keyword score for semantic results
                        result.keyword_score = self._calculate_keyword_score(
                            node_score.node.text, query_context
                        )

                        all_results.append(result)

                except Exception as e:
                    logging.warning(f"Semantic search failed for {collection_name}: {e}")

                # Keyword search
                keyword_results = self.keyword_search(query_context, collection_name)
                all_results.extend(keyword_results)

        # Calculate hybrid scores and remove duplicates
        unique_results = self._merge_and_score_results(all_results, query_context)

        # Sort by hybrid score and return top_k
        unique_results.sort(key=lambda x: x.hybrid_score, reverse=True)

        return [result.node for result in unique_results[:self.top_k]]

    def _merge_and_score_results(
        self,
        results: List[RetrievalResult],
        query_context: QueryContext
    ) -> List[RetrievalResult]:
        """Merge duplicate results and calculate final hybrid scores"""

        # Group results by content similarity
        merged_results = {}

        for result in results:
            # Use chunk_id if available, otherwise use text hash
            key = getattr(result.node.node, 'id_', None) or hash(result.node.node.text[:100])

            if key in merged_results:
                # Update with better scores
                existing = merged_results[key]
                existing.semantic_score = max(existing.semantic_score, result.semantic_score)
                existing.keyword_score = max(existing.keyword_score, result.keyword_score)
            else:
                merged_results[key] = result

        # Calculate final hybrid scores
        for result in merged_results.values():
            # Apply intent-based weighting
            alpha = self.alpha
            if query_context.intent_type == 'definition':
                alpha = 0.8  # Favor semantic search for definitions
            elif query_context.intent_type == 'procedure':
                alpha = 0.6  # Balance semantic and keyword for procedures

            result.hybrid_score = (alpha * result.semantic_score +
                                 (1 - alpha) * result.keyword_score)

        return list(merged_results.values())


class LegalReranker:
    """Legal-specific reranking based on document importance and relevance"""

    def __init__(self):
        self.importance_weights = {
            'main_provision': 1.0,
            'penalty': 0.9,
            'definition': 0.8,
            'procedure': 0.7,
            'exception': 0.6
        }

    def rerank(self, results: List[NodeWithScore], query_context: QueryContext) -> List[NodeWithScore]:
        """Rerank results based on legal document structure and importance"""

        scored_results = []

        for node_score in results:
            base_score = node_score.score or 0.0

            # Get metadata
            metadata = getattr(node_score.node, 'metadata', {})

            # Legal structure importance
            section_type = metadata.get('section_type', 'unknown')
            structure_weight = self.importance_weights.get(section_type, 0.5)

            # Article importance (from enhanced processing)
            try:
                importance_score = float(metadata.get('importance_score', 0.0))
            except (ValueError, TypeError):
                importance_score = 0.0

            # Cross-reference bonus
            cross_ref_count = int(metadata.get('cross_reference_count', 0))
            cross_ref_bonus = min(0.1, cross_ref_count * 0.02)

            # Intent matching bonus
            intent_bonus = 0.0
            text_content = node_score.node.text.lower()

            if query_context.intent_type == 'penalty' and any(term in text_content for term in ['罰', '處', '刑']):
                intent_bonus = 0.1
            elif query_context.intent_type == 'definition' and any(term in text_content for term in ['指', '係', '謂']):
                intent_bonus = 0.1
            elif query_context.intent_type == 'procedure' and any(term in text_content for term in ['應', '得', '程序']):
                intent_bonus = 0.1

            # Calculate final rerank score
            rerank_score = (
                base_score * structure_weight +
                importance_score * 0.2 +
                cross_ref_bonus +
                intent_bonus
            )

            # Update the node score
            node_score.score = rerank_score
            scored_results.append(node_score)

        # Sort by reranked score
        scored_results.sort(key=lambda x: x.score or 0.0, reverse=True)

        return scored_results


class AdvancedRAGSystem:
    """Complete advanced RAG system with all optimization strategies"""

    def __init__(
        self,
        chroma_host: str = None,
        chroma_port: int = None,
        local_db_path: str = None
    ):
        # Load environment variables
        load_dotenv()

        # Use provided values or fall back to environment variables with sensible defaults
        self.chroma_host = chroma_host or os.getenv("CHROMA_HOST") or "localhost"

        # Handle empty string for CHROMA_PORT
        port_env = os.getenv("CHROMA_PORT", "8000")
        if port_env and port_env.strip():
            self.chroma_port = chroma_port or int(port_env)
        else:
            self.chroma_port = chroma_port or 8000

        self.local_db_path = local_db_path or os.getenv("CHROMA_DB_PATH", "chroma_db")
        self.vector_stores = {}
        self.retrievers = {}
        self.hybrid_retriever = None
        self.reranker = LegalReranker()

        # Initialize connections
        self._init_vector_stores()

    def _init_vector_stores(self):
        """Initialize vector stores for different collections"""

        # Configure OpenAI embedding model to match collection dimensions (1536)
        try:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found")

            embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            embed_model = OpenAIEmbedding(
                api_key=api_key,
                model=embedding_model  # 1536 dimensions to match collections
            )
            Settings.embed_model = embed_model
            print(f"[OK] Using OpenAI {embedding_model} (1536 dims)")
        except Exception as e:
            print(f"[WARN] OpenAI embedding configuration failed: {e}")
            # Fallback to ONNX if available (but dimensions won't match existing collections)
            if ONNX_AVAILABLE:
                embed_model = OnnxEmbedding()
                Settings.embed_model = embed_model
                print("[WARN] Fallback to ONNX embeddings (384 dims) - dimension mismatch expected")
            else:
                print("[ERROR] No suitable embedding model available")
                raise

        # Try remote ChromaDB first, fallback to local
        # Configure ChromaDB settings to disable telemetry (comprehensive approach)
        chroma_settings = ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True,
            chroma_server_ssl_enabled=False,
            chroma_server_cors_allow_origins=["*"]
        )

        try:
            chroma_client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                settings=chroma_settings
            )
            # Test connection
            chroma_client.heartbeat()
            print(f"[OK] Connected to remote ChromaDB at {self.chroma_host}:{self.chroma_port}")
        except Exception as e:
            print(f"[WARN] Remote ChromaDB failed: {e}")
            print(f"[OK] Using local ChromaDB at: {self.local_db_path}")

            # For local ChromaDB, don't specify tenant/database to avoid validation issues
            try:
                chroma_client = chromadb.PersistentClient(
                    path=self.local_db_path,
                    settings=chroma_settings
                )
            except Exception as local_error:
                print(f"[WARN] Local ChromaDB with settings failed: {local_error}")
                # Last resort: minimal configuration
                chroma_client = chromadb.PersistentClient(path=self.local_db_path)

        # Initialize collections (using correct names)
        collections = ['food_safety_act', 'labor_law']

        for collection_name in collections:
            try:
                collection = chroma_client.get_collection(collection_name)

                # Create vector store
                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                # Create index
                index = VectorStoreIndex([], storage_context=storage_context)

                # Create retriever
                retriever = index.as_retriever(similarity_top_k=10)

                self.vector_stores[collection_name] = vector_store
                self.retrievers[collection_name] = retriever

                print(f"[OK] Initialized {collection_name} collection")

            except Exception as e:
                print(f"[WARN] Failed to initialize {collection_name}: {e}")

        # Create hybrid retriever with the same ChromaDB client
        if self.retrievers:
            self.hybrid_retriever = HybridRetriever(
                vector_retrievers=self.retrievers,
                chroma_client=chroma_client,  # Pass the configured client
                alpha=0.7,  # 70% semantic, 30% keyword
                top_k=10
            )
            print("[OK] Hybrid retriever initialized")

    def query(
        self,
        query: str,
        top_k: int = 5,
        enable_reranking: bool = True,
        response_mode: str = "compact"
    ) -> Dict[str, Any]:
        """Advanced query with all optimization strategies"""

        if not self.hybrid_retriever:
            return {"error": "RAG system not properly initialized"}

        try:
            # Create query bundle
            query_bundle = QueryBundle(query_str=query)

            # Expand query for context
            query_context = self.hybrid_retriever.query_expander.expand_query(query)

            # Hybrid retrieval
            retrieved_nodes = self.hybrid_retriever.retrieve(query_bundle)

            # Reranking
            if enable_reranking and retrieved_nodes:
                retrieved_nodes = self.reranker.rerank(retrieved_nodes, query_context)

            # Limit to top_k
            final_nodes = retrieved_nodes[:top_k]

            # Generate response
            response_synthesizer = get_response_synthesizer(response_mode=response_mode)
            query_engine = RetrieverQueryEngine.from_args(
                retriever=self.hybrid_retriever,
                response_synthesizer=response_synthesizer
            )

            response = query_engine.query(query_bundle)

            # Prepare detailed results
            results = {
                "response": str(response),
                "query_context": {
                    "original_query": query_context.original_query,
                    "expanded_terms": query_context.expanded_terms,
                    "legal_concepts": query_context.legal_concepts,
                    "intent_type": query_context.intent_type,
                    "target_collections": query_context.target_collections
                },
                "retrieved_documents": [],
                "metadata": {
                    "total_retrieved": len(final_nodes),
                    "reranking_enabled": enable_reranking,
                    "hybrid_search": True
                }
            }

            # Add document details
            for i, node in enumerate(final_nodes):
                doc_info = {
                    "rank": i + 1,
                    "score": node.score or 0.0,
                    "text_preview": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                    "metadata": getattr(node.node, 'metadata', {}),
                    "source": getattr(node.node, 'metadata', {}).get('source_url', 'Unknown')
                }
                results["retrieved_documents"].append(doc_info)

            return results

        except Exception as e:
            return {"error": f"Query failed: {e}"}


def main():
    """Example usage of the Advanced RAG System"""
    print("Initializing Advanced RAG System...")

    rag_system = AdvancedRAGSystem()

    # Test queries
    test_queries = [
        "勞動契約的規定",
        "食品添加物的標示要求",
        "違反勞基法的罰則",
        "什麼是工作時間"
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)

        result = rag_system.query(query, top_k=3)

        if "error" in result:
            print(f"[ERROR] {result['error']}")
        else:
            print(f"Intent: {result['query_context']['intent_type']}")
            print(f"Target Collections: {result['query_context']['target_collections']}")
            print(f"Expanded Terms: {result['query_context']['expanded_terms'][:3]}")
            print(f"\nResponse: {result['response'][:300]}...")
            print(f"\nTop Retrieved Documents:")
            for doc in result['retrieved_documents']:
                print(f"  Rank {doc['rank']}: {doc['text_preview'][:100]}... (Score: {doc['score']:.3f})")


if __name__ == "__main__":
    main()