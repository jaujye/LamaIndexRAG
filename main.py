#!/usr/bin/env python3
"""
台灣法規 RAG 知識檢索系統
支持食品安全衛生管理法、勞動基準法及多法規整合查詢
主要CLI介面程式
"""

import os
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List
import json
from datetime import datetime

# Fix Windows console encoding issues
if sys.platform == "win32":
    import locale
    # Set console encoding to UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    # Set environment variable for subprocess calls
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Disable ChromaDB telemetry completely (multiple methods for compatibility)
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'
os.environ['CHROMA_SERVER_NOFILE'] = '65536'
os.environ['CHROMA_SERVER_CORS_ALLOW_ORIGINS'] = '["*"]'

# Additional telemetry disabling and log level control
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    import logging

    # Set global ChromaDB settings to disable telemetry
    chromadb.configure(anonymized_telemetry=False)

    # Suppress ChromaDB telemetry and verbose logging
    logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
    logging.getLogger('chromadb').setLevel(logging.WARNING)

except Exception:
    pass  # Continue if chromadb not available yet

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich import print as rprint

from src.legal_food_safety_fetcher import FoodSafetyActFetcher
from src.labor_law_fetcher import LaborLawFetcher
from src.civil_law_fetcher import CivilLawFetcher
from src.legal_basic_processor import LegalDocumentProcessor
from src.legal_enhanced_processor import EnhancedLegalProcessor
from src.index_builder import LegalIndexBuilder
from src.legal_single_domain_rag import LegalRAGSystem
from src.query_router import QueryRouter
from src.legal_multi_domain_rag import AdvancedRAGSystem
from src.monitoring import WandbMonitor, initialize_global_monitor, create_config_from_env


class LegalRAGCLI:
    """台灣法規RAG系統的CLI介面，支持食品安全法、勞基法及多法規整合查詢，整合 W&B 監控"""

    def __init__(self, enable_monitoring: bool = True):
        # Initialize console with UTF-8 for Windows compatibility
        try:
            self.console = Console(force_terminal=True, width=None)
            # Try to configure Rich console for UTF-8 on Windows
            if sys.platform == "win32":
                self.console._file = sys.stdout
        except Exception:
            self.console = Console()
        self.food_safety_data_file = Path("data/food_safety_act.json")
        self.labor_law_data_file = Path("data/labor_standards_act.json")
        self.civil_law_data_file = Path("data/civil_code.json")
        self.env_file = Path(".env")

        # 監控設置
        self.enable_monitoring = enable_monitoring
        self.monitor: Optional[WandbMonitor] = None
        self.session_start_time = time.time()

        # 系統組件
        self.rag_system: Optional[LegalRAGSystem] = None
        self.query_router: Optional[QueryRouter] = None
        self.advanced_rag_system: Optional[AdvancedRAGSystem] = None
        self.index_builder: Optional[LegalIndexBuilder] = None

    def setup_monitoring(self):
        """設置 W&B 監控"""
        if not self.enable_monitoring:
            return

        try:
            # 載入環境變數
            from dotenv import load_dotenv
            load_dotenv()

            # 檢查 W&B 設定
            wandb_mode = os.getenv("WANDB_MODE", "online")
            wandb_project = os.getenv("WANDB_PROJECT", "food-safety-rag")

            if wandb_mode == "disabled":
                self.console.print("[yellow]W&B 監控已停用[/yellow]")
                self.enable_monitoring = False
                return

            # 初始化監控器
            try:
                self.monitor = WandbMonitor(
                    project_name=wandb_project,
                    mode=wandb_mode,
                    tags=["cli-session", "food-safety"]
                )
            except Exception as e:
                self.console.print(f"[yellow]監控器初始化失敗: {e}[/yellow]")
                self.enable_monitoring = False
                self.monitor = None
                return

            # 設定為全域監控器
            try:
                initialize_global_monitor(
                    project_name=wandb_project,
                    mode=wandb_mode,
                    tags=["cli-session", "food-safety"]
                )
            except Exception as e:
                self.console.print(f"[yellow]全域監控器設定失敗: {e}[/yellow]")

            # 初始化 W&B run
            if self.monitor:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                config = create_config_from_env()
                config["session_type"] = "cli"

                self.monitor.init_run(
                    run_name=f"cli_session_{timestamp}",
                    config=config
                )

                self.console.print(f"[green]W&B 監控已啟用 - 專案: {wandb_project}[/green]")

        except Exception as e:
            self.console.print(f"[yellow]W&B 監控初始化失敗: {e}[/yellow]")
            self.enable_monitoring = False
            self.monitor = None

    def check_environment(self) -> bool:
        """檢查環境設置"""
        issues = []

        # 檢查.env檔案
        if not self.env_file.exists():
            issues.append("[FAIL] 未找到 .env 檔案。請複製 .env.template 並設定您的 OpenAI API key")
        else:
            # 檢查API key
            from dotenv import load_dotenv
            load_dotenv()
            if not os.getenv("OPENAI_API_KEY"):
                issues.append("[FAIL] .env 檔案中未設定 OPENAI_API_KEY")

        # 檢查資料檔案
        if not self.food_safety_data_file.exists() and not self.labor_law_data_file.exists() and not self.civil_law_data_file.exists():
            issues.append("[FAIL] 未找到法規資料檔案，需要先下載法規內容（食品安全法、勞基法或民法）")

        if issues:
            self.console.print("\n[red]環境設置問題：[/red]")
            for issue in issues:
                self.console.print(f"  {issue}")

            # 記錄環境檢查失敗
            if self.monitor:
                self.monitor.log_metrics({
                    "environment_check_passed": False,
                    "environment_issues_count": len(issues)
                })

            return False

        self.console.print("[OK] 環境檢查通過")

        # 記錄環境檢查成功
        if self.monitor:
            self.monitor.log_metrics({
                "environment_check_passed": True,
                "environment_issues_count": 0
            })

        return True

    def setup_food_safety_data(self) -> bool:
        """設置食品安全法資料（下載法規）"""
        if self.food_safety_data_file.exists():
            if not Confirm.ask("食品安全法資料檔案已存在，是否重新下載？"):
                return True

        self.console.print("\n[yellow]開始下載台灣食品安全衛生管理法...[/yellow]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:

                task = progress.add_task("下載法規內容...", total=None)

                fetcher = FoodSafetyActFetcher(delay=1.0)
                articles = fetcher.fetch_all_articles()

                progress.update(task, description="儲存資料...")
                fetcher.save_to_json(str(self.food_safety_data_file))

                progress.update(task, description="完成！")

            self.console.print(f"[OK] 成功下載 {len(articles)} 條食品安全法")
            return True

        except Exception as e:
            self.console.print(f"[FAIL] 食品安全法下載失敗: {e}")
            return False

    def setup_labor_law_data(self) -> bool:
        """設置勞基法資料（下載法規）"""
        if self.labor_law_data_file.exists():
            if not Confirm.ask("勞基法資料檔案已存在，是否重新下載？"):
                return True

        self.console.print("\n[yellow]開始下載台灣勞動基準法...[/yellow]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:

                task = progress.add_task("下載法規內容...", total=None)

                fetcher = LaborLawFetcher(delay=1.5)  # 稍長的延遲以示尊重
                articles = fetcher.fetch_all_articles()

                progress.update(task, description="儲存資料...")
                fetcher.save_to_json(str(self.labor_law_data_file))

                progress.update(task, description="完成！")

            self.console.print(f"[OK] 成功下載 {len(articles)} 條勞基法")
            return True

        except Exception as e:
            self.console.print(f"[FAIL] 勞基法下載失敗: {e}")
            return False

    def setup_civil_law_data(self) -> bool:
        """設置民法資料（下載法規）"""
        if self.civil_law_data_file.exists():
            if not Confirm.ask("民法資料檔案已存在，是否重新下載？"):
                return True

        self.console.print("\n[yellow]開始下載台灣民法...[/yellow]")
        self.console.print("[dim]注意：民法有1229條，預計需要40分鐘以上時間[/dim]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:

                task = progress.add_task("下載法規內容...", total=None)

                fetcher = CivilLawFetcher(delay=2.0)  # 較長延遲以尊重伺服器
                articles = fetcher.fetch_all_articles()

                progress.update(task, description="儲存資料...")
                fetcher.save_to_json(str(self.civil_law_data_file))

                progress.update(task, description="完成！")

            self.console.print(f"[OK] 成功下載 {len(articles)} 條民法")

            # 顯示各編摘要
            book_summary = fetcher.get_book_summary()
            if book_summary:
                self.console.print("\n各編摘要:")
                for book, count in book_summary.items():
                    self.console.print(f"  - {book}: {count} 條")

            return True

        except Exception as e:
            self.console.print(f"[FAIL] 民法下載失敗: {e}")
            return False

    def build_food_safety_index(self, reset: bool = False) -> bool:
        """建立食品安全法向量索引"""
        try:
            # 傳遞監控器到索引建構器
            self.index_builder = LegalIndexBuilder(
                collection_name="food_safety_act",
                enable_monitoring=self.enable_monitoring,
                monitor=self.monitor
            )

            # 檢查是否已有索引
            existing_index = self.index_builder.load_existing_index()

            if existing_index and not reset:
                self.console.print("[OK] 找到現有食品安全法索引，跳過建立步驟")

                # 記錄跳過索引建立
                if self.monitor:
                    self.monitor.log_metrics({
                        "food_safety_index_build_skipped": True,
                        "index_exists": True,
                        "reset_requested": reset
                    })

                return True

            self.console.print("\n[yellow]建立食品安全法向量索引...[/yellow]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:

                task = progress.add_task("處理食品安全法文件並建立索引...", total=None)

                index = self.index_builder.build_index_from_json(
                    str(self.food_safety_data_file),
                    reset=reset
                )

                progress.update(task, description="食品安全法索引建立完成！")

            # 顯示統計資訊
            stats = self.index_builder.get_index_stats()
            self.show_index_stats(stats, "食品安全法")

            return True

        except Exception as e:
            self.console.print(f"[FAIL] 食品安全法索引建立失敗: {e}")

            # 記錄索引建立失敗
            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"operation": "build_food_safety_index", "reset": reset}
                )

            return False

    def build_labor_law_index(self, reset: bool = False) -> bool:
        """建立勞基法向量索引"""
        try:
            if not self.labor_law_data_file.exists():
                self.console.print("[FAIL] 勞基法資料檔案不存在，請先下載資料")
                return False

            self.console.print("\n[yellow]建立勞基法向量索引...[/yellow]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:

                task = progress.add_task("處理勞基法文件並建立索引...", total=None)

                # 使用增強處理器
                processor = EnhancedLegalProcessor(chunk_size=512, chunk_overlap=50)

                # 載入勞基法資料
                with open(self.labor_law_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 處理文章
                chunks = processor.create_semantic_chunks(data['articles'])
                progress.update(task, description=f"創建了 {len(chunks)} 個語意塊")

                # 初始化勞基法索引建構器
                index_builder = LegalIndexBuilder(
                    collection_name="labor_law",
                    enable_monitoring=False  # 避免編碼問題
                )

                # 轉換為文檔
                documents = processor.convert_to_llama_documents(chunks)
                progress.update(task, description=f"轉換了 {len(documents)} 個文檔")

                # 建立ChromaDB集合
                collection = index_builder.create_collection(reset=reset)

                # 建立向量索引
                from llama_index.vector_stores.chroma import ChromaVectorStore
                from llama_index.core import VectorStoreIndex, StorageContext

                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    show_progress=False
                )

                progress.update(task, description="勞基法索引建立完成！")

            # 保存元數據
            stats = processor.get_processing_stats(chunks)
            metadata = {
                'law_name': data['law_name'],
                'law_code': data['law_code'],
                'source_url': data['source_url'],
                'total_articles': data['total_articles'],
                'collection_name': 'labor_law',
                'processing_stats': stats
            }

            with open('data/labor_law_index_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.console.print(f"[OK] 勞基法索引建立完成！集合: labor_law，文檔: {len(documents)}")

            # 顯示統計資訊
            self.show_enhanced_stats(stats, "勞基法")

            return True

        except Exception as e:
            self.console.print(f"[FAIL] 勞基法索引建立失敗: {e}")
            import traceback
            self.console.print(traceback.format_exc())
            return False

    def build_civil_law_index(self, reset: bool = False) -> bool:
        """建立民法向量索引"""
        try:
            if not self.civil_law_data_file.exists():
                self.console.print("[FAIL] 民法資料檔案不存在，請先下載資料")
                return False

            self.console.print("\n[yellow]建立民法向量索引...[/yellow]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:

                task = progress.add_task("處理民法文件並建立索引...", total=None)

                # 使用增強處理器
                processor = EnhancedLegalProcessor(chunk_size=512, chunk_overlap=50)

                # 載入民法資料
                with open(self.civil_law_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 處理文章
                chunks = processor.create_semantic_chunks(data['articles'])
                progress.update(task, description=f"創建了 {len(chunks)} 個語意塊")

                # 初始化民法索引建構器
                index_builder = LegalIndexBuilder(
                    collection_name="civil_law",
                    enable_monitoring=False  # 避免編碼問題
                )

                # 轉換為文檔
                documents = processor.convert_to_llama_documents(chunks)
                progress.update(task, description=f"轉換了 {len(documents)} 個文檔")

                # 建立ChromaDB集合
                collection = index_builder.create_collection(reset=reset)

                # 建立向量索引
                from llama_index.vector_stores.chroma import ChromaVectorStore
                from llama_index.core import VectorStoreIndex, StorageContext

                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    show_progress=False
                )

                progress.update(task, description="民法索引建立完成！")

            # 保存元數據
            stats = processor.get_processing_stats(chunks)
            metadata = {
                'law_name': data['law_name'],
                'law_code': data['law_code'],
                'source_url': data['source_url'],
                'total_articles': data['total_articles'],
                'collection_name': 'civil_law',
                'processing_stats': stats
            }

            with open('data/civil_law_index_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.console.print(f"[OK] 民法索引建立完成！集合: civil_law，文檔: {len(documents)}")

            # 顯示統計資訊
            self.show_enhanced_stats(stats, "民法")

            return True

        except Exception as e:
            self.console.print(f"[FAIL] 民法索引建立失敗: {e}")
            import traceback
            self.console.print(traceback.format_exc())
            return False

    def initialize_single_rag_system(self) -> bool:
        """初始化單一法規RAG系統（向後相容）"""
        try:
            self.console.print("\n[yellow]初始化單一法規RAG查詢系統...[/yellow]")

            # 傳遞監控器到 RAG 系統
            self.rag_system = LegalRAGSystem(
                enable_monitoring=self.enable_monitoring,
                monitor=self.monitor
            )
            self.rag_system.setup_query_engine(
                similarity_top_k=10,
                similarity_cutoff=0.3
            )

            self.console.print("[OK] 單一法規RAG系統初始化完成")

            # 記錄 RAG 系統初始化成功
            if self.monitor:
                self.monitor.log_metrics({
                    "rag_system_initialized": True,
                    "similarity_top_k": 10,
                    "similarity_cutoff": 0.3
                })

            return True

        except Exception as e:
            self.console.print(f"[FAIL] 單一法規RAG系統初始化失敗: {e}")

            # 記錄 RAG 系統初始化失敗
            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"operation": "initialize_single_rag_system"}
                )

            return False

    def initialize_multi_domain_system(self) -> bool:
        """初始化多法規整合系統"""
        try:
            self.console.print("\n[yellow]初始化多法規整合查詢系統...[/yellow]")

            # 初始化查詢路由器（使用環境變數設定）
            self.query_router = QueryRouter()

            # 初始化進階RAG系統（使用環境變數設定）
            self.advanced_rag_system = AdvancedRAGSystem()

            self.console.print("[OK] 多法規整合系統初始化完成")

            # 記錄多法規系統初始化成功
            if self.monitor:
                self.monitor.log_metrics({
                    "multi_domain_system_initialized": True,
                    "query_router_enabled": True,
                    "advanced_rag_enabled": True
                })

            return True

        except Exception as e:
            self.console.print(f"[FAIL] 多法規整合系統初始化失敗: {e}")

            # 記錄多法規系統初始化失敗
            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"operation": "initialize_multi_domain_system"}
                )

            return False

    def show_index_stats(self, stats: dict, law_name: str = ""):
        """顯示索引統計資訊"""
        title = f"{law_name}索引統計資訊" if law_name else "索引統計資訊"
        table = Table(title=title)
        table.add_column("項目", style="cyan")
        table.add_column("數值", style="green")

        # 基本統計
        table.add_row("文件總數", str(stats.get("document_count", "未知")))
        table.add_row("嵌入模型", stats.get("embedding_model", "未知"))
        table.add_row("法規名稱", stats.get("law_name", "未知"))

        # 處理統計
        if "processing_stats" in stats:
            ps = stats["processing_stats"]
            table.add_row("處理的條文數", str(ps.get("articles_processed", "未知")))
            table.add_row("總chunk數", str(ps.get("total_chunks", "未知")))
            table.add_row("平均chunk大小", f"{ps.get('avg_chunk_size', 0):.1f} tokens")

        self.console.print(table)

    def show_enhanced_stats(self, stats: dict, law_name: str = ""):
        """顯示增強處理統計資訊"""
        title = f"{law_name}增強處理統計" if law_name else "增強處理統計"
        table = Table(title=title)
        table.add_column("項目", style="cyan")
        table.add_column("數值", style="green")

        table.add_row("總關鍵字數", str(stats.get("total_keywords", 0)))
        table.add_row("總概念數", str(stats.get("total_concepts", 0)))
        table.add_row("總交叉引用數", str(stats.get("total_cross_references", 0)))
        table.add_row("平均重要性分數", f"{stats.get('avg_importance_score', 0):.3f}")
        table.add_row("高重要性塊數", str(stats.get("high_importance_chunks", 0)))
        table.add_row("引用圖節點數", str(stats.get("graph_nodes", 0)))
        table.add_row("引用圖邊數", str(stats.get("graph_edges", 0)))

        self.console.print(table)

    def single_domain_query_interface(self):
        """單一法規互動式查詢介面（向後相容）"""
        self.console.print("\n" + "="*60)
        panel_text = "[bold blue]台灣法規 RAG 知識檢索系統[/bold blue]\n" \
                    "輸入您的問題，系統將基於法規內容為您解答\n" \
                    "[dim]輸入 'quit' 或 'exit' 結束程式[/dim]"

        if self.enable_monitoring and self.monitor:
            panel_text += "\n[dim]🔍 W&B 監控已啟用[/dim]"

        self.console.print(Panel.fit(panel_text, border_style="blue"))

        session_queries = 0
        session_start = time.time()

        while True:
            try:
                # 獲取使用者問題
                question = Prompt.ask("\n[bold cyan]請輸入您的問題[/bold cyan]")

                if question.lower() in ['quit', 'exit', '退出']:
                    break

                if not question.strip():
                    continue

                # 執行查詢
                self.console.print("\n[yellow]🔍 搜尋相關法規...[/yellow]")

                query_start = time.time()
                result = self.rag_system.query(question)
                query_time = time.time() - query_start

                session_queries += 1

                # 顯示結果
                self.display_query_result(result)

                # 記錄互動式查詢統計
                if self.monitor:
                    self.monitor.log_metrics({
                        "session_queries": session_queries,
                        "session_duration": time.time() - session_start,
                        "interactive_mode": True,
                        "query_response_time": query_time
                    })

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[FAIL] 查詢錯誤: {e}")

                # 記錄查詢錯誤
                if self.monitor:
                    self.monitor.log_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        context={"operation": "interactive_query", "session_queries": session_queries}
                    )

        session_time = time.time() - session_start
        self.console.print("\n[green]感謝使用法規知識檢索系統！[/green]")

        # 記錄會話摘要
        if self.monitor:
            self.monitor.create_summary({
                "session_total_queries": session_queries,
                "session_total_time": session_time,
                "session_avg_query_time": session_time / session_queries if session_queries > 0 else 0,
                "session_type": "single_domain_interactive"
            })

    def multi_domain_query_interface(self):
        """多法規整合互動式查詢介面"""
        self.console.print("\n" + "="*60)
        panel_text = "[bold blue]台灣多法規整合 RAG 知識檢索系統[/bold blue]\n" \
                    "支持食品安全法、勞基法及跨法規查詢\n" \
                    "系統將智能路由您的問題到最適合的法規知識庫\n" \
                    "[dim]輸入 'quit' 或 'exit' 結束程式[/dim]"

        if self.enable_monitoring and self.monitor:
            panel_text += "\n[dim]🔍 W&B 監控已啟用[/dim]"

        self.console.print(Panel.fit(panel_text, border_style="blue"))

        session_queries = 0
        session_start = time.time()

        while True:
            try:
                # 獲取使用者問題
                question = Prompt.ask("\n[bold cyan]請輸入您的問題[/bold cyan]")

                if question.lower() in ['quit', 'exit', '退出']:
                    break

                if not question.strip():
                    continue

                # 執行多法規查詢
                self.console.print("\n[yellow]🔍 智能路由查詢到相關法規...[/yellow]")

                query_start = time.time()
                response = self.query_router.route_query(question, top_k=5)
                query_time = time.time() - query_start

                session_queries += 1

                # 顯示多法規查詢結果
                self.display_multi_domain_result(response)

                # 記錄互動式查詢統計
                if self.monitor:
                    self.monitor.log_metrics({
                        "session_queries": session_queries,
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
                self.console.print(f"[FAIL] 查詢錯誤: {e}")

                # 記錄查詢錯誤
                if self.monitor:
                    self.monitor.log_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        context={"operation": "multi_domain_interactive_query", "session_queries": session_queries}
                    )

        session_time = time.time() - session_start
        self.console.print("\n[green]感謝使用多法規整合知識檢索系統！[/green]")

        # 記錄會話摘要
        if self.monitor:
            self.monitor.create_summary({
                "session_total_queries": session_queries,
                "session_total_time": session_time,
                "session_avg_query_time": session_time / session_queries if session_queries > 0 else 0,
                "session_type": "multi_domain_interactive"
            })

    def display_query_result(self, result):
        """顯示查詢結果"""
        # 回答
        self.console.print(Panel(
            result.answer,
            title="[bold green] 法規解答[/bold green]",
            border_style="green"
        ))

        # 信心度和查詢類型
        confidence_color = "green" if result.confidence_score > 0.7 else "yellow" if result.confidence_score > 0.5 else "red"
        self.console.print(f"\n[{confidence_color}]信心度: {result.confidence_score:.3f}[/{confidence_color}] | 查詢類型: [blue]{result.query_type}[/blue]")

        # 相關法條
        if result.sources:
            self.console.print("\n[bold cyan]📚 相關法條:[/bold cyan]")

            sources_table = Table()
            sources_table.add_column("條文", style="cyan")
            sources_table.add_column("相似度", style="yellow")
            sources_table.add_column("類型", style="green")
            sources_table.add_column("內容預覽", style="white")

            for source in result.sources[:3]:  # 只顯示前3個最相關的
                similarity = f"{source['similarity_score']:.3f}"
                preview = source['text_preview'][:100] + "..." if len(source['text_preview']) > 100 else source['text_preview']

                sources_table.add_row(
                    f"第{source['article_number']}條",
                    similarity,
                    source['chunk_type'].replace('article_', ''),
                    preview
                )

            self.console.print(sources_table)

    def display_multi_domain_result(self, response):
        """顯示多法規查詢結果"""
        # 路由資訊
        kb_name = response.route_decision.primary_kb.value if response.route_decision.primary_kb else "conversational"
        self.console.print(f"\n[dim]🧠 查詢路由: {kb_name} " +
                          f"(信心度: {response.route_decision.confidence_score:.2f})[/dim]")
        if response.route_decision.secondary_kbs:
            secondary = ", ".join([kb.value for kb in response.route_decision.secondary_kbs])
            self.console.print(f"[dim]次要查詢: {secondary}[/dim]")

        # 融合回答
        if response.fused_response:
            self.console.print(Panel(
                response.fused_response,
                title="[bold green] 法規解答[/bold green]",
                border_style="green"
            ))
        else:
            self.console.print("[yellow]未找到相關資訊[/yellow]")

        # 顯示各法規的查詢結果摘要
        if len(response.responses) > 1:
            # 檢查是否有任何有意義的結果
            meaningful_results = []
            total_docs = 0

            for kb_name, kb_response in response.responses.items():
                if "error" not in kb_response:
                    law_name = "勞基法" if kb_name == "labor_law" else "食品安全法"
                    doc_count = len(kb_response.get('metadata', {}).get('retrieved_nodes', []))
                    answer_preview = kb_response.get('response', '')

                    total_docs += doc_count

                    # 檢查回答是否有意義（不是通用的"找不到"訊息）
                    is_meaningful = (
                        doc_count > 0 and
                        answer_preview and
                        not answer_preview.startswith("I'm sorry, but based on the provided context") and
                        not answer_preview.startswith("很抱歉，根據提供的") and
                        len(answer_preview.strip()) > 20
                    )

                    if is_meaningful:
                        meaningful_results.append({
                            'name': law_name,
                            'count': doc_count,
                            'preview': answer_preview[:100] + "..." if len(answer_preview) > 100 else answer_preview
                        })

            # 只有當有意義的結果時才顯示表格
            if meaningful_results:
                self.console.print("\n[bold cyan]📈 各法規查詢結果:[/bold cyan]")

                results_table = Table()
                results_table.add_column("法規", style="cyan")
                results_table.add_column("相關文檔數", style="yellow")
                results_table.add_column("答案預覽", style="white")

                for result in meaningful_results:
                    results_table.add_row(
                        result['name'],
                        str(result['count']),
                        result['preview']
                    )

                self.console.print(results_table)
            elif total_docs == 0:
                # 完全沒有找到相關文檔時顯示中文訊息
                self.console.print("\n[yellow]📋 查詢結果: 查詢不到任何相關法規結果[/yellow]")

    def batch_query_mode(self, questions_file: str):
        """批次查詢模式"""
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]

            self.console.print(f"[yellow]批次處理 {len(questions)} 個問題...[/yellow]")

            results = []
            with Progress(console=self.console) as progress:
                task = progress.add_task("處理問題...", total=len(questions))

                for question in questions:
                    result = self.rag_system.query(question)
                    results.append(result)
                    progress.advance(task)

            # 儲存結果
            output_file = f"batch_results_{Path(questions_file).stem}.json"
            self.save_batch_results(results, output_file)

            self.console.print(f"[OK] 批次查詢完成，結果已儲存至 {output_file}")

        except Exception as e:
            self.console.print(f"[FAIL] 批次查詢失敗: {e}")

    def save_batch_results(self, results: List, output_file: str):
        """儲存批次查詢結果"""
        output_data = []
        for result in results:
            output_data.append({
                "question": result.question,
                "answer": result.answer,
                "confidence_score": result.confidence_score,
                "query_type": result.query_type,
                "sources": result.sources
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def finish_monitoring_session(self):
        """完成監控會話"""
        if not self.monitor:
            return

        try:
            # 計算會話總時間
            total_session_time = time.time() - self.session_start_time

            # 建立最終摘要
            final_summary = {
                "total_session_time": total_session_time,
                "session_end_time": datetime.now().isoformat(),
                "monitoring_enabled": self.enable_monitoring
            }

            # 如果有 RAG 系統，加入其統計資訊
            if self.rag_system:
                rag_stats = self.rag_system.get_system_stats()
                final_summary.update({
                    "total_queries_processed": rag_stats.get("total_queries", 0),
                    "avg_query_time": rag_stats.get("avg_query_time", 0.0)
                })

            self.monitor.create_summary(final_summary)
            self.monitor.finish_run()
            self.console.print("[dim]W&B 監控會話已結束[/dim]")

        except Exception as e:
            self.console.print(f"[yellow]完成監控會話時發生錯誤: {e}[/yellow]")

    def show_all_domain_stats(self):
        """顯示所有法規索引統計資訊"""
        self.console.print("\n[bold blue]所有法規索引統計資訊[/bold blue]")

        # 食品安全法統計
        if self.food_safety_data_file.exists():
            try:
                index_builder = LegalIndexBuilder(
                    collection_name="food_safety_act",
                    enable_monitoring=False
                )
                if index_builder.load_existing_index():
                    stats = index_builder.get_index_stats()
                    self.show_index_stats(stats, "食品安全法")
                else:
                    self.console.print("[yellow]食品安全法索引不存在[/yellow]")
            except Exception as e:
                self.console.print(f"[yellow]無法載入食品安全法索引: {e}[/yellow]")
        else:
            self.console.print("[yellow]食品安全法資料不存在[/yellow]")

        # 勞基法統計
        if Path("data/labor_law_index_metadata.json").exists():
            try:
                with open('data/labor_law_index_metadata.json', 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                self.show_enhanced_stats(metadata.get('processing_stats', {}), "勞基法")
            except Exception as e:
                self.console.print(f"[yellow]無法載入勞基法統計: {e}[/yellow]")
        else:
            self.console.print("[yellow]勞基法索引不存在[/yellow]")

        # 民法統計
        if Path("data/civil_law_index_metadata.json").exists():
            try:
                with open('data/civil_law_index_metadata.json', 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                self.show_enhanced_stats(metadata.get('processing_stats', {}), "民法")
            except Exception as e:
                self.console.print(f"[yellow]無法載入民法統計: {e}[/yellow]")
        else:
            self.console.print("[yellow]民法索引不存在[/yellow]")

    def run(self, args):
        """主要執行邏輯"""
        self.console.print("[bold blue]台灣法規 RAG 知識檢索系統[/bold blue]\n")

        # 初始化監控
        if self.enable_monitoring:
            self.setup_monitoring()

        # 環境檢查
        if not self.check_environment():
            if Confirm.ask("是否要進行初始設置？"):
                self.console.print("\n[yellow]請先完成以下設置：[/yellow]")
                self.console.print("1. 複製 .env.template 為 .env")
                self.console.print("2. 在 .env 中設定您的 OPENAI_API_KEY")
                self.console.print("3. 重新執行程式")
            return

        # 處理所有統計資訊顯示
        if args.show_all_stats:
            self.show_all_domain_stats()
            return

        # 處理食品安全法資料及索引
        if args.fetch_food_data or (not self.food_safety_data_file.exists() and not args.multi_domain and not args.domain == 'labor'):
            if not self.setup_food_safety_data():
                return

        # 處理勞基法資料及索引
        if args.fetch_labor_data:
            if not self.setup_labor_law_data():
                return

        # 處理民法資料及索引
        if args.fetch_civil_data:
            if not self.setup_civil_law_data():
                return

        # 建立食品安全法索引
        if args.rebuild_food_index or (not Path("chroma_db").exists() and (not args.multi_domain and args.domain != 'labor')):
            if not self.build_food_safety_index(reset=args.rebuild_food_index):
                return

        # 建立勞基法索引
        if args.rebuild_labor_index:
            if not self.build_labor_law_index(reset=True):
                return

        # 建立民法索引
        if args.rebuild_civil_index:
            if not self.build_civil_law_index(reset=True):
                return

        # 選擇運行模式
        if args.multi_domain:
            # 多法規整合模式
            if not self.initialize_multi_domain_system():
                return

            # 執行對應模式
            if args.batch_file:
                self.multi_domain_batch_query_mode(args.batch_file)
            elif args.query:
                response = self.query_router.route_query(args.query, top_k=5)
                self.display_multi_domain_result(response)

                # 記錄單一查詢模式
                if self.monitor:
                    self.monitor.create_summary({
                        "session_type": "multi_domain_single_query",
                        "query_text": args.query[:100] + "..." if len(args.query) > 100 else args.query,
                        "primary_kb": response.route_decision.primary_kb.value if response.route_decision.primary_kb else "conversational",
                        "routing_confidence": response.route_decision.confidence_score
                    })
            else:
                self.multi_domain_query_interface()
        else:
            # 單一法規模式（向後相容）
            if not self.initialize_single_rag_system():
                return

            # 執行對應模式
            if args.batch_file:
                self.batch_query_mode(args.batch_file)
            elif args.query:
                result = self.rag_system.query(args.query)
                self.display_query_result(result)

                # 記錄單一查詢模式
                if self.monitor:
                    self.monitor.create_summary({
                        "session_type": "single_query",
                        "query_text": args.query[:100] + "..." if len(args.query) > 100 else args.query
                    })
            else:
                self.single_domain_query_interface()

    def multi_domain_batch_query_mode(self, questions_file: str):
        """多法規批次查詢模式"""
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]

            self.console.print(f"[yellow]多法規批次處理 {len(questions)} 個問題...[/yellow]")

            results = []
            with Progress(console=self.console) as progress:
                task = progress.add_task("處理問題...", total=len(questions))

                for question in questions:
                    response = self.query_router.route_query(question, top_k=5)
                    results.append(response)
                    progress.advance(task)

            # 儲存結果
            output_file = f"multi_domain_batch_results_{Path(questions_file).stem}.json"
            self.save_multi_domain_batch_results(results, output_file)

            self.console.print(f"[OK] 多法規批次查詢完成，結果已儲存至 {output_file}")

        except Exception as e:
            self.console.print(f"[FAIL] 多法規批次查詢失敗: {e}")

    def save_multi_domain_batch_results(self, results, output_file: str):
        """儲存多法規批次查詢結果"""
        output_data = []
        for response in results:
            output_data.append({
                "question": response.query,
                "primary_kb": response.route_decision.primary_kb.value if response.route_decision.primary_kb else "conversational",
                "confidence_score": response.route_decision.confidence_score,
                "reasoning": response.route_decision.reasoning,
                "fused_response": response.fused_response,
                "responses": {kb: resp for kb, resp in response.responses.items() if "error" not in resp},
                "metadata": response.metadata
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # 完成監控會話
        self.finish_monitoring_session()


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description="台灣法規 RAG 知識檢索系統 - 支持食品安全法、勞基法及多法規整合查詢 (含 W&B 監控)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本使用
  python main.py                                    # 單一法規互動式查詢
  python main.py --multi-domain                     # 多法規整合互動式查詢
  python main.py -q "食品添加物的限制？"            # 單一查詢
  python main.py --multi-domain -q "勞工食品安全規定"  # 多法規單一查詢

  # 資料管理
  python main.py --fetch-food-data                  # 下載食品安全法資料
  python main.py --fetch-labor-data                 # 下載勞基法資料
  python main.py --fetch-civil-data                 # 下載民法資料
  python main.py --rebuild-food-index               # 重建食品安全法索引
  python main.py --rebuild-labor-index              # 重建勞基法索引
  python main.py --rebuild-civil-index              # 重建民法索引

  # 批次處理和統計
  python main.py --batch questions.txt              # 單一法規批次查詢
  python main.py --multi-domain --batch questions.txt # 多法規批次查詢
  python main.py --all-stats                        # 顯示所有法規統計
  python main.py --no-monitoring                    # 停用 W&B 監控
        """
    )

    # 查詢相關參數
    parser.add_argument("-q", "--query",
                       help="執行單一查詢")
    parser.add_argument("--multi-domain", action="store_true",
                       help="啟用多法規整合模式，支持智能路由和跨法規查詢")
    parser.add_argument("--domain", choices=["food", "labor", "civil", "all"], default="food",
                       help="指定查詢的法規領域 (預設: food)")

    # 資料管理參數
    parser.add_argument("--fetch-food-data", action="store_true",
                       help="下載食品安全衛生管理法資料")
    parser.add_argument("--fetch-labor-data", action="store_true",
                       help="下載勞動基準法資料")
    parser.add_argument("--fetch-civil-data", action="store_true",
                       help="下載台灣民法資料")
    parser.add_argument("--rebuild-food-index", action="store_true",
                       help="重建食品安全法向量索引")
    parser.add_argument("--rebuild-labor-index", action="store_true",
                       help="重建勞基法向量索引")
    parser.add_argument("--rebuild-civil-index", action="store_true",
                       help="重建民法向量索引")

    # 批次處理和統計參數
    parser.add_argument("--batch", dest="batch_file",
                       help="批次查詢檔案路徑")
    parser.add_argument("--all-stats", dest="show_all_stats", action="store_true",
                       help="顯示所有法規索引統計資訊")

    # 系統參數
    parser.add_argument("--no-monitoring", action="store_true",
                       help="停用 W&B 監控")

    # 向後相容參數
    parser.add_argument("--fetch-data", action="store_true",
                       help="(已廢棄) 請使用 --fetch-food-data")
    parser.add_argument("--rebuild-index", action="store_true",
                       help="(已廢棄) 請使用 --rebuild-food-index")
    parser.add_argument("--stats", dest="show_stats", action="store_true",
                       help="(已廢棄) 請使用 --all-stats")

    args = parser.parse_args()

    # 處理向後相容參數
    if args.fetch_data:
        args.fetch_food_data = True
        print("[WARNING] --fetch-data 已廢棄，自動轉換為 --fetch-food-data")
    if args.rebuild_index:
        args.rebuild_food_index = True
        print("[WARNING] --rebuild-index 已廢棄，自動轉換為 --rebuild-food-index")
    if args.show_stats:
        args.show_all_stats = True
        print("[WARNING] --stats 已廢棄，自動轉換為 --all-stats")

    # 檢查Python版本
    if sys.version_info < (3, 8):
        print("[FAIL] 需要 Python 3.8 或更高版本")
        sys.exit(1)

    try:
        cli = LegalRAGCLI(enable_monitoring=not args.no_monitoring)
        cli.run(args)
    except KeyboardInterrupt:
        print("\n程式已中止")
    except Exception as e:
        print(f"[FAIL] 程式錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()