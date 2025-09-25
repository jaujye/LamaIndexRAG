"""
Monitoring and logging module for RAG system using Weights & Biases
Tracks system performance, query metrics, and provides visualization capabilities
"""

import os
import time
import functools
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    # Silently handle missing wandb - will be handled gracefully by WandbMonitor


@dataclass
class RAGMetrics:
    """Data class for RAG system metrics"""
    # Query metrics
    query_text: str = ""
    query_type: str = ""
    query_enhanced: bool = False

    # Timing metrics
    total_time: float = 0.0
    retrieval_time: float = 0.0
    llm_time: float = 0.0
    embedding_time: float = 0.0

    # Retrieval metrics
    documents_retrieved: int = 0
    similarity_scores: List[float] = None
    avg_similarity: float = 0.0
    min_similarity: float = 0.0
    max_similarity: float = 0.0

    # Response metrics
    response_length: int = 0
    confidence_score: float = 0.0
    tokens_used: int = 0

    # System metrics
    memory_usage_mb: float = 0.0
    error_occurred: bool = False
    error_message: str = ""

    # Index metrics (for index building operations)
    documents_processed: int = 0
    chunks_created: int = 0
    embedding_dimensions: int = 0
    index_build_time: float = 0.0

    def __post_init__(self):
        if self.similarity_scores is None:
            self.similarity_scores = []

        if self.similarity_scores:
            self.avg_similarity = sum(self.similarity_scores) / len(self.similarity_scores)
            self.min_similarity = min(self.similarity_scores)
            self.max_similarity = max(self.similarity_scores)


class WandbMonitor:
    """Weights & Biases monitoring integration for RAG system"""

    def __init__(self,
                 project_name: Optional[str] = None,
                 entity: Optional[str] = None,
                 api_key: Optional[str] = None,
                 mode: str = "online",
                 tags: Optional[List[str]] = None):
        """
        Initialize W&B monitoring

        Args:
            project_name: W&B project name
            entity: W&B entity (username/team)
            api_key: W&B API key
            mode: W&B mode ("online", "offline", "disabled")
            tags: List of tags for the run
        """
        load_dotenv()

        self.enabled = WANDB_AVAILABLE and mode != "disabled"

        if not self.enabled:
            if not WANDB_AVAILABLE:
                print("ℹ️  W&B 未安裝，監控功能已停用。安裝方式: pip install wandb")
            elif mode == "disabled":
                print("ℹ️  W&B 監控已停用")
            return

        # Load configuration from environment or parameters
        self.project_name = project_name or os.getenv("WANDB_PROJECT", "food-safety-rag")
        self.entity = entity or os.getenv("WANDB_ENTITY")
        self.mode = mode or os.getenv("WANDB_MODE", "online")

        # Set API key if provided
        if api_key or os.getenv("WANDB_API_KEY"):
            wandb.login(key=api_key or os.getenv("WANDB_API_KEY"))

        self.run = None
        self.metrics_buffer = []

        # Default tags
        default_tags = ["rag", "food-safety", "taiwan-law"]
        self.tags = (tags or []) + default_tags

    def init_run(self,
                 run_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 resume: bool = False):
        """Initialize W&B run"""
        if not self.enabled:
            return

        try:
            # Generate run name if not provided
            if not run_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"rag_session_{timestamp}"

            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                mode=self.mode,
                tags=self.tags,
                config=config or {},
                resume=resume
            )

            print(f"W&B run initialized: {self.run.name}")

        except Exception as e:
            print(f"Failed to initialize W&B run: {e}")
            self.enabled = False

    def log_metrics(self, metrics: Union[RAGMetrics, Dict[str, Any]], step: Optional[int] = None):
        """Log metrics to W&B"""
        if not self.enabled or not self.run:
            return

        try:
            if isinstance(metrics, RAGMetrics):
                metrics_dict = asdict(metrics)
            else:
                metrics_dict = metrics

            # Filter out None values and empty lists
            filtered_metrics = {}
            for key, value in metrics_dict.items():
                if value is not None and value != [] and value != "":
                    if isinstance(value, list) and len(value) > 0:
                        # Log list statistics
                        if all(isinstance(x, (int, float)) for x in value):
                            filtered_metrics[f"{key}_mean"] = sum(value) / len(value)
                            filtered_metrics[f"{key}_min"] = min(value)
                            filtered_metrics[f"{key}_max"] = max(value)
                            filtered_metrics[f"{key}_count"] = len(value)
                    else:
                        filtered_metrics[key] = value

            wandb.log(filtered_metrics, step=step)

        except Exception as e:
            print(f"Failed to log metrics to W&B: {e}")

    def log_query_result(self,
                        query: str,
                        answer: str,
                        sources: List[Dict[str, Any]],
                        metrics: RAGMetrics):
        """Log query result with rich context"""
        if not self.enabled or not self.run:
            return

        try:
            # Create a table for sources
            source_data = []
            for i, source in enumerate(sources[:5]):  # Limit to top 5 sources
                source_data.append([
                    i + 1,
                    source.get('article_number', 'Unknown'),
                    source.get('similarity_score', 0.0),
                    source.get('chunk_type', 'unknown'),
                    source.get('text_preview', '')[:100] + "..."
                ])

            source_table = wandb.Table(
                columns=["Rank", "Article", "Similarity", "Type", "Preview"],
                data=source_data
            )

            # Log query details
            self.run.log({
                "query_table": wandb.Table(
                    columns=["Query", "Answer", "Confidence", "Sources_Count"],
                    data=[[query, answer[:200] + "...", metrics.confidence_score, len(sources)]]
                ),
                "sources_table": source_table
            })

        except Exception as e:
            print(f"Failed to log query result: {e}")

    def log_system_stats(self, stats: Dict[str, Any]):
        """Log system statistics"""
        if not self.enabled:
            return

        try:
            # Flatten nested dictionaries
            flat_stats = {}
            for key, value in stats.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_stats[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_stats[key] = value

            self.log_metrics(flat_stats)

        except Exception as e:
            print(f"Failed to log system stats: {e}")

    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Log error information"""
        if not self.enabled:
            return

        try:
            error_data = {
                "error_type": error_type,
                "error_message": error_message[:500],  # Truncate long messages
                "error_occurred": True,
                "timestamp": datetime.now().isoformat()
            }

            if context:
                error_data.update(context)

            self.log_metrics(error_data)

        except Exception as e:
            print(f"Failed to log error: {e}")

    def finish_run(self):
        """Finish W&B run"""
        if not self.enabled or not self.run:
            return

        try:
            wandb.finish()
            print("W&B run finished")
        except Exception as e:
            print(f"Failed to finish W&B run: {e}")

    def create_summary(self, summary_data: Dict[str, Any]):
        """Create run summary"""
        if not self.enabled or not self.run:
            return

        try:
            for key, value in summary_data.items():
                wandb.run.summary[key] = value
        except Exception as e:
            print(f"Failed to create summary: {e}")


def monitor_execution_time(metric_name: str = "execution_time"):
    """Decorator to monitor function execution time"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Try to log to W&B if monitor instance is available
                if hasattr(args[0], 'monitor') and args[0].monitor:
                    args[0].monitor.log_metrics({
                        f"{func.__name__}_{metric_name}": execution_time,
                        f"{func.__name__}_success": True
                    })

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                # Log error if monitor available
                if hasattr(args[0], 'monitor') and args[0].monitor:
                    args[0].monitor.log_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        context={
                            f"{func.__name__}_{metric_name}": execution_time,
                            f"{func.__name__}_success": False,
                            "function_name": func.__name__
                        }
                    )

                raise  # Re-raise the exception

        return wrapper
    return decorator


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def create_config_from_env() -> Dict[str, Any]:
    """Create W&B config from environment variables"""
    load_dotenv()

    config = {
        "chunk_size": int(os.getenv("CHUNK_SIZE", 512)),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 50)),
        "top_k_retrieval": int(os.getenv("TOP_K_RETRIEVAL", 5)),
        "chroma_db_path": os.getenv("CHROMA_DB_PATH", "./chroma_db"),
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-3.5-turbo",
        "law_source": "Taiwan Food Safety and Hygiene Act"
    }

    return config


# Global monitor instance
_global_monitor: Optional[WandbMonitor] = None


def get_global_monitor() -> Optional[WandbMonitor]:
    """Get the global monitor instance"""
    return _global_monitor


def initialize_global_monitor(**kwargs) -> WandbMonitor:
    """Initialize the global monitor instance"""
    global _global_monitor
    _global_monitor = WandbMonitor(**kwargs)
    return _global_monitor


def main():
    """Example usage and testing"""
    print("Testing W&B monitoring module...")

    # Test monitor initialization
    monitor = WandbMonitor(mode="disabled")  # Use disabled mode for testing

    # Test metrics creation
    metrics = RAGMetrics(
        query_text="食品添加物的規定是什麼？",
        query_type="additives",
        total_time=1.5,
        documents_retrieved=5,
        similarity_scores=[0.85, 0.78, 0.72, 0.65, 0.58],
        response_length=250,
        confidence_score=0.75
    )

    print("Sample metrics:")
    print(f"  Query: {metrics.query_text}")
    print(f"  Average similarity: {metrics.avg_similarity:.3f}")
    print(f"  Confidence: {metrics.confidence_score:.3f}")

    # Test decorator
    @monitor_execution_time("test_time")
    def test_function():
        time.sleep(0.1)
        return "test result"

    class TestClass:
        def __init__(self):
            self.monitor = monitor

    test_obj = TestClass()
    result = test_function.__get__(test_obj, TestClass)()
    print(f"Test function result: {result}")

    print("Monitoring module test completed successfully!")


if __name__ == "__main__":
    main()