"""
RAG (Retrieval-Augmented Generation) system for legal documents
Handles queries and generates responses using indexed legal knowledge
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.schema import NodeWithScore
from llama_index.core.prompts.base import PromptTemplate
from llama_index.llms.openai import OpenAI

from .index_builder import LegalIndexBuilder
from .monitoring import WandbMonitor, RAGMetrics, get_memory_usage, monitor_execution_time


@dataclass
class QueryResult:
    """Represents a query result with metadata"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    query_type: str


class LegalRAGSystem:
    """RAG system for legal document queries with W&B monitoring"""

    def __init__(self,
                 api_key: Optional[str] = None,
                 chroma_path: str = None,
                 collection_name: str = "food_safety_act",
                 temperature: float = 0.1,
                 enable_monitoring: bool = True,
                 monitor: Optional[WandbMonitor] = None):
        """
        Initialize the RAG system

        Args:
            api_key: OpenAI API key
            chroma_path: Path to ChromaDB or HTTP URL for remote ChromaDB
            collection_name: ChromaDB collection name
            temperature: LLM temperature for generation
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        # Get chroma path from environment or parameter
        self.chroma_path = chroma_path or os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.collection_name = collection_name
        self.temperature = temperature

        # Initialize monitoring FIRST
        self.enable_monitoring = enable_monitoring
        self.monitor = monitor
        if self.enable_monitoring and not self.monitor:
            # Create a default monitor if none provided
            self.monitor = WandbMonitor(mode="disabled")  # Will check env for actual mode

        self.query_count = 0
        self.total_queries_time = 0.0

        # Initialize components
        self.index_builder = LegalIndexBuilder(
            api_key=self.api_key,
            chroma_path=chroma_path,
            collection_name=collection_name,
            enable_monitoring=self.enable_monitoring,
            monitor=self.monitor
        )

        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None

        # Load existing index
        self._load_index()

    def _load_index(self):
        """Load the vector index"""
        start_time = time.time()

        self.index = self.index_builder.load_existing_index()
        if not self.index:
            raise ValueError(
                "No index found. Please build an index first using index_builder.py"
            )

        load_time = time.time() - start_time

        # Log index loading metrics
        if self.monitor:
            self.monitor.log_metrics({
                "index_load_time": load_time,
                "index_loaded_successfully": True,
                "memory_usage_after_load": get_memory_usage()
            })

        print("RAG system initialized with existing index")

    def _create_custom_prompt_template(self) -> PromptTemplate:
        """Create custom prompt template for legal queries"""
        template_str = """你是一個專業的食品安全法律顧問助手。請基於提供的台灣食品安全衛生管理法條文，回答使用者的問題。

指導原則：
1. 只使用提供的法條內容來回答問題
2. 明確引用相關的法條編號
3. 如果法條內容不足以回答問題，請明確說明
4. 使用繁體中文回答
5. 保持專業和準確的語調

法條內容：
{context_str}

問題：{query_str}

請提供詳細且準確的回答："""

        return PromptTemplate(template=template_str)

    def setup_query_engine(self,
                          similarity_top_k: int = 5,
                          similarity_cutoff: float = 0.7,
                          response_mode: str = "compact") -> RetrieverQueryEngine:
        """
        Setup the query engine with custom parameters

        Args:
            similarity_top_k: Number of similar documents to retrieve
            similarity_cutoff: Minimum similarity score for relevance
            response_mode: Response synthesis mode

        Returns:
            Configured query engine
        """
        if not self.index:
            raise ValueError("Index not loaded")

        # Create retriever with custom parameters
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k
        )

        # Add postprocessor to filter by similarity
        postprocessor = SimilarityPostprocessor(
            similarity_cutoff=similarity_cutoff
        )

        # Create query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor],
            response_synthesizer=self.index.as_query_engine(
                response_mode=response_mode,
                text_qa_template=self._create_custom_prompt_template()
            )._response_synthesizer
        )

        return self.query_engine

    def classify_query_type(self, query: str) -> str:
        """Classify the type of legal query"""
        query_lower = query.lower()

        # Query type classification
        if any(keyword in query_lower for keyword in ['罰', '處罰', '違反', '刑責']):
            return 'penalty'
        elif any(keyword in query_lower for keyword in ['標示', '標籤', '包裝']):
            return 'labeling'
        elif any(keyword in query_lower for keyword in ['添加物', '防腐劑', '色素']):
            return 'additives'
        elif any(keyword in query_lower for keyword in ['衛生', '清潔', '消毒']):
            return 'hygiene'
        elif any(keyword in query_lower for keyword in ['檢驗', '檢查', '稽查']):
            return 'inspection'
        elif any(keyword in query_lower for keyword in ['進口', '輸入', '邊境']):
            return 'import'
        elif any(keyword in query_lower for keyword in ['製造', '加工', '生產']):
            return 'manufacturing'
        else:
            return 'general'

    def enhance_query(self, query: str, query_type: str) -> str:
        """Enhance query based on its type"""
        enhancements = {
            'penalty': f"{query} 相關的罰則和處罰規定",
            'labeling': f"{query} 相關的標示和標籤要求",
            'additives': f"{query} 相關的食品添加物規定和限制",
            'hygiene': f"{query} 相關的衛生安全標準和要求",
            'inspection': f"{query} 相關的檢驗和稽查程序",
            'import': f"{query} 相關的進口和邊境管制規定",
            'manufacturing': f"{query} 相關的製造和加工標準"
        }

        return enhancements.get(query_type, query)

    def calculate_relevance_score(self, query: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate custom relevance scores for retrieved sources

        TODO(human): Implement custom relevance scoring logic that considers:
        - Similarity score from vector search
        - Article type relevance (main_provision vs items vs penalties)
        - Chapter relevance based on query topic
        - Recency or importance of specific legal provisions

        Args:
            query: The original user query
            sources: List of retrieved source documents with metadata

        Returns:
            sources: Same list but with updated relevance_score field
        """
        # Current implementation just uses similarity score
        # TODO(human): Enhance this with custom scoring logic

        for source in sources:
            # Base score from vector similarity
            base_score = source.get('similarity_score', 0.0)

            # TODO(human): Add your custom relevance factors here
            # Consider factors like:
            # - Article type weighting (penalties might be more important for penalty queries)
            # - Chapter relevance
            # - Legal hierarchy (general provisions vs specific rules)
            # - Article number patterns (certain ranges might be more relevant)

            # For now, just use base similarity score
            source['relevance_score'] = base_score

        # TODO(human): Sort sources by your custom relevance score
        # sources.sort(key=lambda x: x['relevance_score'], reverse=True)

        return sources

    @monitor_execution_time("query_total_time")
    def query(self,
              question: str,
              similarity_top_k: int = 5,
              similarity_cutoff: float = 0.7,
              enhance_query: bool = True) -> QueryResult:
        """
        Query the legal knowledge base with monitoring

        Args:
            question: User's question
            similarity_top_k: Number of documents to retrieve
            similarity_cutoff: Minimum similarity threshold
            enhance_query: Whether to enhance the query based on type

        Returns:
            QueryResult with answer and source information
        """
        start_time = time.time()
        self.query_count += 1

        try:
            if not self.query_engine:
                self.setup_query_engine(similarity_top_k, similarity_cutoff)

            # Classify and enhance query
            classify_start = time.time()
            query_type = self.classify_query_type(question)
            enhanced_question = self.enhance_query(question, query_type) if enhance_query else question
            classify_time = time.time() - classify_start

            print(f"Query type: {query_type}")
            if enhance_query and enhanced_question != question:
                print(f"Enhanced query: {enhanced_question}")

            # Execute query with timing
            llm_start = time.time()
            response = self.query_engine.query(enhanced_question)
            llm_time = time.time() - llm_start

            # Process sources
            process_start = time.time()
            sources = []
            similarity_scores = []

            for node_with_score in response.source_nodes:
                similarity_score = float(node_with_score.score) if node_with_score.score else 0.0
                similarity_scores.append(similarity_score)

                source_info = {
                    'article_number': node_with_score.node.metadata.get('article_number', 'Unknown'),
                    'article_title': node_with_score.node.metadata.get('article_title', ''),
                    'chapter': node_with_score.node.metadata.get('chapter', ''),
                    'chunk_type': node_with_score.node.metadata.get('chunk_type', ''),
                    'similarity_score': similarity_score,
                    'text_preview': node_with_score.node.text[:200] + "..." if len(node_with_score.node.text) > 200 else node_with_score.node.text,
                    'source_url': node_with_score.node.metadata.get('source_url', '')
                }
                sources.append(source_info)

            # Apply custom relevance scoring
            sources = self.calculate_relevance_score(enhanced_question, sources)
            process_time = time.time() - process_start

            # Calculate average confidence score using relevance scores
            total_relevance = sum(source.get('relevance_score', source['similarity_score']) for source in sources)
            confidence_score = total_relevance / len(sources) if sources else 0.0

            # Calculate total time
            total_time = time.time() - start_time
            self.total_queries_time += total_time

            # Create result
            result = QueryResult(
                question=question,
                answer=str(response),
                sources=sources,
                confidence_score=confidence_score,
                query_type=query_type
            )

            # Log metrics to W&B
            if self.monitor:
                # Estimate token usage (rough approximation)
                estimated_tokens = len(question.split()) + len(str(response).split()) * 2

                metrics = RAGMetrics(
                    query_text=question[:100] + "..." if len(question) > 100 else question,
                    query_type=query_type,
                    query_enhanced=enhance_query and enhanced_question != question,
                    total_time=total_time,
                    retrieval_time=classify_time + process_time,
                    llm_time=llm_time,
                    documents_retrieved=len(sources),
                    similarity_scores=similarity_scores,
                    response_length=len(str(response)),
                    confidence_score=confidence_score,
                    tokens_used=estimated_tokens,
                    memory_usage_mb=get_memory_usage()
                )

                self.monitor.log_metrics(metrics)
                self.monitor.log_query_result(question, str(response), sources, metrics)

                # Log cumulative statistics
                self.monitor.log_metrics({
                    "total_queries": self.query_count,
                    "avg_query_time": self.total_queries_time / self.query_count,
                    "cumulative_query_time": self.total_queries_time
                })

            return result

        except Exception as e:
            total_time = time.time() - start_time

            # Log error to monitoring
            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={
                        "query_text": question[:100] + "..." if len(question) > 100 else question,
                        "query_type": query_type if 'query_type' in locals() else "unknown",
                        "execution_time": total_time,
                        "query_count": self.query_count
                    }
                )

            raise  # Re-raise the exception

    def batch_query(self, questions: List[str]) -> List[QueryResult]:
        """Process multiple questions in batch with monitoring"""
        batch_start = time.time()
        results = []
        successful_queries = 0
        failed_queries = 0

        for i, question in enumerate(questions):
            try:
                result = self.query(question)
                results.append(result)
                successful_queries += 1

                # Log batch progress
                if self.monitor and (i + 1) % 10 == 0:  # Log every 10 queries
                    self.monitor.log_metrics({
                        "batch_progress": (i + 1) / len(questions),
                        "batch_successful": successful_queries,
                        "batch_failed": failed_queries
                    })

            except Exception as e:
                failed_queries += 1
                print(f"Error processing question '{question}': {e}")

                # Create error result
                error_result = QueryResult(
                    question=question,
                    answer=f"處理問題時發生錯誤: {e}",
                    sources=[],
                    confidence_score=0.0,
                    query_type="error"
                )
                results.append(error_result)

        batch_time = time.time() - batch_start

        # Log batch summary
        if self.monitor:
            self.monitor.log_metrics({
                "batch_total_time": batch_time,
                "batch_total_questions": len(questions),
                "batch_successful_queries": successful_queries,
                "batch_failed_queries": failed_queries,
                "batch_success_rate": successful_queries / len(questions) if questions else 0,
                "batch_avg_time_per_query": batch_time / len(questions) if questions else 0
            })

        return results

    def get_related_articles(self, article_number: str) -> List[Dict[str, Any]]:
        """Get articles related to a specific article number"""
        query = f"與第{article_number}條相關的規定"
        result = self.query(query, similarity_top_k=10)

        # Filter out the original article and return related ones
        related = []
        for source in result.sources:
            if source['article_number'] != article_number:
                related.append(source)

        return related[:5]  # Return top 5 related articles

    def explain_article(self, article_number: str) -> Optional[QueryResult]:
        """Get detailed explanation of a specific article"""
        query = f"請解釋第{article_number}條的內容和規定"
        return self.query(query)

    def search_by_keyword(self, keyword: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search documents by keyword"""
        results = self.index_builder.search_similar(keyword, top_k=top_k)
        return results

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics with monitoring data"""
        index_stats = self.index_builder.get_index_stats()

        system_stats = {
            "rag_system_status": "active",
            "query_engine_configured": self.query_engine is not None,
            "temperature": self.temperature,
            "monitoring_enabled": self.enable_monitoring and self.monitor is not None,
            "total_queries": self.query_count,
            "avg_query_time": self.total_queries_time / self.query_count if self.query_count > 0 else 0.0,
            "cumulative_query_time": self.total_queries_time,
            "current_memory_usage": get_memory_usage(),
            "index_stats": index_stats
        }

        # Log system stats if monitoring is enabled
        if self.monitor:
            self.monitor.log_system_stats(system_stats)

        return system_stats


def main():
    """Example usage and testing"""
    try:
        # Initialize RAG system
        print("Initializing Legal RAG System...")
        rag_system = LegalRAGSystem()

        # Setup query engine
        rag_system.setup_query_engine(
            similarity_top_k=5,
            similarity_cutoff=0.6
        )

        # Test queries
        test_questions = [
            "食品添加物有什麼限制？",
            "違反食品標示規定會受到什麼處罰？",
            "食品製造業者需要符合哪些衛生條件？",
            "進口食品需要經過哪些檢驗？"
        ]

        print("\n" + "="*60)
        print("測試查詢系統")
        print("="*60)

        for question in test_questions:
            print(f"\n問題: {question}")
            print("-" * 40)

            result = rag_system.query(question)

            print(f"回答:\n{result.answer}")
            print(f"\n信心度: {result.confidence_score:.3f}")
            print(f"查詢類型: {result.query_type}")

            print(f"\n相關法條:")
            for i, source in enumerate(result.sources[:3], 1):
                print(f"  {i}. 第{source['article_number']}條 (相似度: {source['similarity_score']:.3f})")
                print(f"     {source['text_preview']}")

            print("\n" + "-" * 60)

        # System statistics
        stats = rag_system.get_system_stats()
        print(f"\n系統統計:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()