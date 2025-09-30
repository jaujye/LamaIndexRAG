"""
Query Handler Module
Handles all query-related business logic for the Legal RAG CLI.
Separated from main CLI to improve code organization and maintainability.
"""

import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.legal_single_domain_rag import LegalRAGSystem, QueryResult
from src.legal_multi_domain_rag import AdvancedRAGSystem
from src.query_router import QueryRouter, QueryResponse
from src.cli.index_manager import IndexManager
from src.cli.cli_renderer import CLIRenderer
from src.cli.data_manager import LawType
from src.monitoring import WandbMonitor


class QueryHandler:
    """
    Handles query processing for the Legal RAG system.

    Responsibilities:
    - Initialize single-domain and multi-domain RAG systems
    - Provide interactive query interfaces
    - Handle batch query processing
    - Format and display query results
    - Integrate with monitoring system
    """

    def __init__(
        self,
        index_manager: IndexManager,
        renderer: CLIRenderer,
        enable_monitoring: bool = False,
        monitor: Optional[WandbMonitor] = None
    ):
        """
        Initialize query handler.

        Args:
            index_manager: IndexManager instance for checking indices
            renderer: CLIRenderer for displaying results
            enable_monitoring: Whether to enable W&B monitoring
            monitor: Optional WandbMonitor instance
        """
        self.index_manager = index_manager
        self.renderer = renderer
        self.enable_monitoring = enable_monitoring
        self.monitor = monitor

        # RAG system instances (initialized on demand)
        self.single_rag_system: Optional[LegalRAGSystem] = None
        self.query_router: Optional[QueryRouter] = None
        self.advanced_rag_system: Optional[AdvancedRAGSystem] = None

        # Query statistics
        self.session_queries = 0
        self.session_start_time = time.time()

    def initialize_single_domain_system(self, law_type: LawType) -> bool:
        """
        Initialize single-domain RAG system for a specific law type.

        Args:
            law_type: Type of law to initialize system for

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Check if index exists
            if not self.index_manager.check_index_exists(law_type):
                law_name = self.index_manager.data_manager.get_law_name(law_type)
                self.renderer.display_error(f"{law_name}索引不存在，請先建立索引")
                return False

            self.renderer.display_info("\n初始化單一法規RAG查詢系統...")

            # Get collection name for the law type
            collection_name = IndexManager.COLLECTION_NAMES[law_type]

            # Initialize single-domain RAG system
            self.single_rag_system = LegalRAGSystem(
                collection_name=collection_name,
                enable_monitoring=self.enable_monitoring,
                monitor=self.monitor
            )

            # Setup query engine with default parameters
            self.single_rag_system.setup_query_engine(
                similarity_top_k=10,
                similarity_cutoff=0.3
            )

            law_name = self.index_manager.data_manager.get_law_name(law_type)
            self.renderer.display_success(f"{law_name}RAG系統初始化完成")

            # Log initialization metrics
            if self.monitor:
                self.monitor.log_metrics({
                    "rag_system_initialized": True,
                    "law_type": law_type.value,
                    "collection_name": collection_name,
                    "similarity_top_k": 10,
                    "similarity_cutoff": 0.3
                })

            return True

        except Exception as e:
            self.renderer.display_error(f"單一法規RAG系統初始化失敗: {e}")

            # Log error
            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"operation": "initialize_single_domain_system", "law_type": law_type.value}
                )

            return False

    def initialize_multi_domain_system(self) -> bool:
        """
        Initialize multi-domain RAG system with query routing.
        Requires all three law indices to exist.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Check if all required indices exist
            required_law_types = [LawType.FOOD_SAFETY, LawType.LABOR_LAW, LawType.CIVIL_LAW]
            missing_indices = []

            for law_type in required_law_types:
                if not self.index_manager.check_index_exists(law_type):
                    law_name = self.index_manager.data_manager.get_law_name(law_type)
                    missing_indices.append(law_name)

            if missing_indices:
                self.renderer.display_error(
                    f"多法規整合系統需要所有法規索引，缺少: {', '.join(missing_indices)}"
                )
                self.renderer.display_info("請先建立所有法規索引")
                return False

            self.renderer.display_info("\n初始化多法規整合查詢系統...")

            # Initialize query router (will initialize AdvancedRAGSystem internally)
            self.query_router = QueryRouter()

            # Store reference to advanced RAG system
            self.advanced_rag_system = self.query_router.rag_system

            self.renderer.display_success("多法規整合系統初始化完成")

            # Log initialization metrics
            if self.monitor:
                self.monitor.log_metrics({
                    "multi_domain_system_initialized": True,
                    "query_router_enabled": True,
                    "advanced_rag_enabled": True,
                    "supported_law_types": [lt.value for lt in required_law_types]
                })

            return True

        except Exception as e:
            self.renderer.display_error(f"多法規整合系統初始化失敗: {e}")

            # Log error
            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"operation": "initialize_multi_domain_system"}
                )

            return False

    def interactive_single_query(self):
        """
        Interactive single-domain query interface.
        Allows users to ask questions until they quit.
        """
        if not self.single_rag_system:
            self.renderer.display_error("RAG系統尚未初始化")
            return

        # Display welcome message
        self.renderer.display_welcome(enable_monitoring=self.enable_monitoring and self.monitor is not None)

        # Reset session statistics
        self.session_queries = 0
        session_start = time.time()

        while True:
            try:
                # Get user question
                question = Prompt.ask("\n[bold cyan]請輸入您的問題[/bold cyan]")

                # Check for exit commands
                if question.lower() in ['quit', 'exit', '退出']:
                    break

                if not question.strip():
                    continue

                # Execute query
                self.renderer.display_searching()

                query_start = time.time()
                result = self.single_rag_system.query(question)
                query_time = time.time() - query_start

                self.session_queries += 1

                # Display result
                self.renderer.display_query_result(result)

                # Log query metrics
                if self.monitor:
                    self.monitor.log_metrics({
                        "session_queries": self.session_queries,
                        "session_duration": time.time() - session_start,
                        "interactive_mode": True,
                        "query_response_time": query_time
                    })

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.renderer.display_error(f"查詢錯誤: {e}")

                # Log error
                if self.monitor:
                    self.monitor.log_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        context={"operation": "interactive_single_query", "session_queries": self.session_queries}
                    )

        # Display goodbye message
        session_time = time.time() - session_start
        self.renderer.display_goodbye()

        # Log session summary
        if self.monitor:
            self.monitor.create_summary({
                "session_total_queries": self.session_queries,
                "session_total_time": session_time,
                "session_avg_query_time": session_time / self.session_queries if self.session_queries > 0 else 0,
                "session_type": "single_domain_interactive"
            })

    def interactive_multi_query(self):
        """
        Interactive multi-domain query interface.
        Routes queries to appropriate knowledge bases.
        """
        if not self.query_router:
            self.renderer.display_error("多法規查詢路由器尚未初始化")
            return

        # Display welcome message
        self.renderer.display_multi_domain_welcome(enable_monitoring=self.enable_monitoring and self.monitor is not None)

        # Reset session statistics
        self.session_queries = 0
        session_start = time.time()

        while True:
            try:
                # Get user question
                question = Prompt.ask("\n[bold cyan]請輸入您的問題[/bold cyan]")

                # Check for exit commands
                if question.lower() in ['quit', 'exit', '退出']:
                    break

                if not question.strip():
                    continue

                # Execute multi-domain query
                self.renderer.display_multi_domain_searching()

                query_start = time.time()
                response = self.query_router.route_query(question, top_k=5)
                query_time = time.time() - query_start

                self.session_queries += 1

                # Display result
                self.renderer.display_multi_domain_result(response)

                # Log query metrics
                if self.monitor:
                    self.monitor.log_metrics({
                        "session_queries": self.session_queries,
                        "session_duration": time.time() - session_start,
                        "interactive_mode": True,
                        "multi_domain_mode": True,
                        "query_response_time": query_time,
                        "primary_kb": response.route_decision.primary_kb.value if response.route_decision.primary_kb else "conversational",
                        "routing_confidence": response.route_decision.confidence_score
                    })

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.renderer.display_error(f"查詢錯誤: {e}")

                # Log error
                if self.monitor:
                    self.monitor.log_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        context={"operation": "interactive_multi_query", "session_queries": self.session_queries}
                    )

        # Display goodbye message
        session_time = time.time() - session_start
        self.renderer.display_goodbye(message="感謝使用多法規整合知識檢索系統！")

        # Log session summary
        if self.monitor:
            self.monitor.create_summary({
                "session_total_queries": self.session_queries,
                "session_total_time": session_time,
                "session_avg_query_time": session_time / self.session_queries if self.session_queries > 0 else 0,
                "session_type": "multi_domain_interactive"
            })

    def batch_query(
        self,
        questions_file: str,
        output_file: Optional[str] = None,
        multi_domain: bool = False
    ):
        """
        Process batch queries from a file.

        Args:
            questions_file: Path to file containing questions (one per line or JSON array)
            output_file: Optional output file path for results (default: auto-generated)
            multi_domain: Whether to use multi-domain routing
        """
        try:
            # Load questions from file
            questions = self._load_questions_from_file(questions_file)

            if not questions:
                self.renderer.display_error("未找到問題或檔案格式錯誤")
                return

            # Determine output file name
            if not output_file:
                prefix = "multi_domain_batch" if multi_domain else "batch"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"{prefix}_results_{timestamp}.json"

            # Display processing info
            mode_text = "多法規批次" if multi_domain else "批次"
            self.renderer.display_info(f"\n{mode_text}處理 {len(questions)} 個問題...")

            # Process queries with progress bar
            results = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.renderer.console
            ) as progress:
                task = progress.add_task("處理問題...", total=len(questions))

                for question in questions:
                    try:
                        if multi_domain:
                            response = self.query_router.route_query(question, top_k=5)
                            results.append(self._format_multi_domain_result(question, response))
                        else:
                            result = self.single_rag_system.query(question)
                            results.append(self._format_single_domain_result(result))

                        progress.advance(task)

                    except Exception as e:
                        self.renderer.display_warning(f"處理問題失敗: {question[:50]}... - {e}")
                        results.append({
                            "question": question,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        })
                        progress.advance(task)

            # Save results to file
            self._save_batch_results(results, output_file)

            self.renderer.display_success(f"{mode_text}查詢完成，結果已儲存至 {output_file}")

            # Log batch completion
            if self.monitor:
                self.monitor.log_metrics({
                    "batch_mode": True,
                    "multi_domain": multi_domain,
                    "total_questions": len(questions),
                    "successful_queries": len([r for r in results if "error" not in r]),
                    "failed_queries": len([r for r in results if "error" in r])
                })

        except Exception as e:
            self.renderer.display_error(f"批次查詢失敗: {e}")

            # Log error
            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"operation": "batch_query", "questions_file": questions_file}
                )

    def _load_questions_from_file(self, file_path: str) -> List[str]:
        """
        Load questions from text file or JSON file.

        Args:
            file_path: Path to questions file

        Returns:
            List of question strings
        """
        questions = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

                # Try to parse as JSON first
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        questions = [str(q) for q in data if q]
                    elif isinstance(data, dict) and "questions" in data:
                        questions = [str(q) for q in data["questions"] if q]
                except json.JSONDecodeError:
                    # Fall back to line-by-line text file
                    questions = [line.strip() for line in content.split('\n') if line.strip()]

        except Exception as e:
            self.renderer.display_error(f"讀取問題檔案失敗: {e}")

        return questions

    def _format_single_domain_result(self, result: QueryResult) -> Dict[str, Any]:
        """Format single-domain query result for batch output."""
        return {
            "question": result.question,
            "answer": result.answer,
            "confidence_score": result.confidence_score,
            "query_type": result.query_type,
            "sources": result.sources,
            "timestamp": datetime.now().isoformat()
        }

    def _format_multi_domain_result(self, question: str, response: QueryResponse) -> Dict[str, Any]:
        """Format multi-domain query result for batch output."""
        return {
            "question": question,
            "primary_kb": response.route_decision.primary_kb.value if response.route_decision.primary_kb else "conversational",
            "confidence_score": response.route_decision.confidence_score,
            "reasoning": response.route_decision.reasoning,
            "fused_response": response.fused_response,
            "responses": {
                kb: resp for kb, resp in response.responses.items() if "error" not in resp
            },
            "metadata": response.metadata,
            "timestamp": datetime.now().isoformat()
        }

    def _save_batch_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save batch query results to JSON file.

        Args:
            results: List of result dictionaries
            output_file: Output file path
        """
        try:
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save with metadata
            output_data = {
                "metadata": {
                    "total_queries": len(results),
                    "timestamp": datetime.now().isoformat(),
                    "monitoring_enabled": self.enable_monitoring
                },
                "results": results
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.renderer.display_error(f"儲存結果失敗: {e}")

    def execute_single_query(self, question: str) -> bool:
        """
        Execute a single query (non-interactive mode).

        Args:
            question: Question to query

        Returns:
            True if query successful
        """
        if not self.single_rag_system:
            self.renderer.display_error("RAG系統尚未初始化")
            return False

        try:
            self.renderer.display_searching()

            query_start = time.time()
            result = self.single_rag_system.query(question)
            query_time = time.time() - query_start

            # Display result
            self.renderer.display_query_result(result)

            # Log query
            if self.monitor:
                self.monitor.log_metrics({
                    "single_query_mode": True,
                    "query_response_time": query_time
                })

                self.monitor.create_summary({
                    "session_type": "single_query",
                    "query_text": question[:100] + "..." if len(question) > 100 else question
                })

            return True

        except Exception as e:
            self.renderer.display_error(f"查詢失敗: {e}")

            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"operation": "execute_single_query"}
                )

            return False

    def execute_multi_domain_query(self, question: str) -> bool:
        """
        Execute a single multi-domain query (non-interactive mode).

        Args:
            question: Question to query

        Returns:
            True if query successful
        """
        if not self.query_router:
            self.renderer.display_error("多法規查詢路由器尚未初始化")
            return False

        try:
            self.renderer.display_multi_domain_searching()

            query_start = time.time()
            response = self.query_router.route_query(question, top_k=5)
            query_time = time.time() - query_start

            # Display result
            self.renderer.display_multi_domain_result(response)

            # Log query
            if self.monitor:
                self.monitor.log_metrics({
                    "multi_domain_single_query_mode": True,
                    "query_response_time": query_time,
                    "primary_kb": response.route_decision.primary_kb.value if response.route_decision.primary_kb else "conversational",
                    "routing_confidence": response.route_decision.confidence_score
                })

                self.monitor.create_summary({
                    "session_type": "multi_domain_single_query",
                    "query_text": question[:100] + "..." if len(question) > 100 else question,
                    "primary_kb": response.route_decision.primary_kb.value if response.route_decision.primary_kb else "conversational",
                    "routing_confidence": response.route_decision.confidence_score
                })

            return True

        except Exception as e:
            self.renderer.display_error(f"多法規查詢失敗: {e}")

            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"operation": "execute_multi_domain_query"}
                )

            return False