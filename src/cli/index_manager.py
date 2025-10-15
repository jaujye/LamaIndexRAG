"""
Index Manager
Manages vector index creation for different legal document types
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.index_builder import LegalIndexBuilder
from src.legal_basic_processor import LegalDocumentProcessor
from src.legal_enhanced_processor import EnhancedLegalProcessor
from src.cli.data_manager import DataManager, LawType
from src.monitoring import WandbMonitor

# Import necessary LlamaIndex components for enhanced processing
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext


class IndexManager:
    """
    Manages vector index creation for legal documents.

    Responsibilities:
    - Build indices for different law types (food safety, labor law, civil law)
    - Use appropriate processors (basic vs enhanced)
    - Integrate with LegalIndexBuilder
    - Support index reset functionality
    - Provide index statistics
    """

    # Collection names for each law type
    COLLECTION_NAMES = {
        LawType.FOOD_SAFETY: "food_safety_act",
        LawType.LABOR_LAW: "labor_law",
        LawType.CIVIL_LAW: "civil_law",
    }

    # Metadata file paths for enhanced processing
    METADATA_PATHS = {
        LawType.LABOR_LAW: Path("data/labor_law_index_metadata.json"),
        LawType.CIVIL_LAW: Path("data/civil_law_index_metadata.json"),
    }

    def __init__(
        self,
        data_manager: DataManager,
        enable_monitoring: bool = False,
        monitor: Optional[WandbMonitor] = None,
        console: Optional[Console] = None
    ):
        """
        Initialize index manager.

        Args:
            data_manager: DataManager instance for accessing law data
            enable_monitoring: Whether to enable performance monitoring
            monitor: Optional WandbMonitor instance for monitoring
            console: Optional Rich Console for output
        """
        self.data_manager = data_manager
        self.enable_monitoring = enable_monitoring
        self.monitor = monitor
        self.console = console or Console()

        # Index builder will be created per law type
        self.index_builder: Optional[LegalIndexBuilder] = None

    def build_index(
        self,
        law_type: LawType,
        reset: bool = False,
        use_enhanced: bool = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Build vector index for a specific law type.

        Args:
            law_type: Type of law to build index for
            reset: Whether to reset existing index
            use_enhanced: Whether to use enhanced processor (auto-detect if None)

        Returns:
            Tuple of (success: bool, stats: Optional[Dict])
        """
        # Determine processor type if not specified
        if use_enhanced is None:
            # Food safety uses basic processor, others use enhanced
            use_enhanced = law_type != LawType.FOOD_SAFETY

        # Check if data exists
        if not self.data_manager.data_exists(law_type):
            self.console.print(f"[FAIL] {self.data_manager.get_law_name(law_type)}資料檔案不存在，請先下載資料")
            return False, None

        # Build index based on law type
        if law_type == LawType.FOOD_SAFETY:
            return self._build_food_safety_index(reset)
        elif law_type == LawType.LABOR_LAW:
            return self._build_enhanced_index(law_type, reset)
        elif law_type == LawType.CIVIL_LAW:
            return self._build_enhanced_index(law_type, reset)
        else:
            self.console.print(f"[FAIL] 不支援的法規類型: {law_type}")
            return False, None

    def _build_food_safety_index(self, reset: bool = False) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Build food safety law index using basic processor"""
        try:
            law_name = self.data_manager.get_law_name(LawType.FOOD_SAFETY)

            # Initialize index builder with monitoring
            self.index_builder = LegalIndexBuilder(
                collection_name=self.COLLECTION_NAMES[LawType.FOOD_SAFETY],
                enable_monitoring=self.enable_monitoring,
                monitor=self.monitor
            )

            # Check for existing index
            existing_index = self.index_builder.load_existing_index()

            if existing_index and not reset:
                self.console.print(f"[OK] 找到現有{law_name}索引，跳過建立步驟")

                # Log skipped index build
                if self.monitor:
                    self.monitor.log_metrics({
                        "food_safety_index_build_skipped": True,
                        "index_exists": True,
                        "reset_requested": reset
                    })

                # Return existing stats
                stats = self.index_builder.get_index_stats()
                return True, stats

            self.console.print(f"\n[yellow]建立{law_name}向量索引...[/yellow]")

            # Build index with progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(f"處理{law_name}文件並建立索引...", total=None)

                data_path = self.data_manager.get_data_path(LawType.FOOD_SAFETY)
                index = self.index_builder.build_index_from_json(
                    str(data_path),
                    reset=reset
                )

                progress.update(task, description=f"{law_name}索引建立完成！")

            # Get and return statistics
            stats = self.index_builder.get_index_stats()
            return True, stats

        except Exception as e:
            self.console.print(f"[FAIL] {law_name}索引建立失敗: {e}")

            # Log error
            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"operation": "build_food_safety_index", "reset": reset}
                )

            return False, None

    def _build_enhanced_index(
        self,
        law_type: LawType,
        reset: bool = False
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Build index using enhanced processor (for labor law and civil law)"""
        try:
            law_name = self.data_manager.get_law_name(law_type)
            collection_name = self.COLLECTION_NAMES[law_type]

            self.console.print(f"\n[yellow]建立{law_name}向量索引...[/yellow]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(f"處理{law_name}文件並建立索引...", total=None)

                # Initialize enhanced processor
                processor = EnhancedLegalProcessor(chunk_size=512, chunk_overlap=50)

                # Load law data
                data = self.data_manager.load_data(law_type)
                if not data:
                    self.console.print(f"[FAIL] 無法載入{law_name}資料")
                    return False, None

                # Process articles into semantic chunks
                chunks = processor.create_semantic_chunks(data['articles'])
                progress.update(task, description=f"創建了 {len(chunks)} 個語意塊")

                # Initialize index builder (disable monitoring to avoid encoding issues)
                index_builder = LegalIndexBuilder(
                    collection_name=collection_name,
                    enable_monitoring=False
                )

                # Convert chunks to documents
                documents = processor.convert_to_llama_documents(chunks)
                progress.update(task, description=f"轉換了 {len(documents)} 個文檔")

                # Create ChromaDB collection
                collection = index_builder.create_collection(reset=reset)

                # Build vector index
                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    show_progress=False
                )

                progress.update(task, description=f"{law_name}索引建立完成！")

            # Save metadata
            stats = processor.get_processing_stats(chunks)
            metadata = {
                'law_name': data['law_name'],
                'law_code': data['law_code'],
                'source_url': data['source_url'],
                'total_articles': data['total_articles'],
                'collection_name': collection_name,
                'processing_stats': stats
            }

            # Save to appropriate metadata file
            metadata_path = self.METADATA_PATHS.get(law_type)
            if metadata_path:
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.console.print(
                f"[OK] {law_name}索引建立完成！集合: {collection_name}，文檔: {len(documents)}"
            )

            return True, stats

        except Exception as e:
            law_name = self.data_manager.get_law_name(law_type)
            self.console.print(f"[FAIL] {law_name}索引建立失敗: {e}")

            import traceback
            self.console.print(traceback.format_exc())

            return False, None

    def check_index_exists(self, law_type: LawType) -> bool:
        """
        Check if an index exists for a law type.

        Args:
            law_type: Type of law to check

        Returns:
            True if index exists
        """
        try:
            collection_name = self.COLLECTION_NAMES[law_type]
            index_builder = LegalIndexBuilder(
                collection_name=collection_name,
                enable_monitoring=False
            )

            existing_index = index_builder.load_existing_index()
            return existing_index is not None

        except Exception:
            return False

    def get_index_stats(self, law_type: LawType) -> Optional[Dict[str, Any]]:
        """
        Get statistics for an existing index.

        Args:
            law_type: Type of law

        Returns:
            Dictionary of statistics, or None if index doesn't exist
        """
        try:
            collection_name = self.COLLECTION_NAMES[law_type]
            index_builder = LegalIndexBuilder(
                collection_name=collection_name,
                enable_monitoring=False
            )

            # Load existing index
            existing_index = index_builder.load_existing_index()
            if not existing_index:
                return None

            # Get basic stats
            stats = index_builder.get_index_stats()

            # For enhanced indices, try to load additional metadata
            metadata_path = self.METADATA_PATHS.get(law_type)
            if metadata_path and metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    stats.update(metadata)

            return stats

        except Exception as e:
            self.console.print(f"[WARN] 無法取得索引統計: {e}")
            return None