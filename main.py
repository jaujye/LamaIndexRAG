#!/usr/bin/env python3
"""
台灣法規 RAG 知識檢索系統
支持食品安全衛生管理法、勞動基準法、民法及多法規整合查詢
主要CLI介面程式 - 重構版本
"""

import os
import argparse
import sys
import time
from pathlib import Path
from typing import Optional
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
from rich.prompt import Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.cli.environment_validator import EnvironmentValidator
from src.cli.data_manager import DataManager, LawType
from src.cli.index_manager import IndexManager
from src.cli.cli_renderer import CLIRenderer
from src.cli.query_handler import QueryHandler
from src.monitoring import WandbMonitor, initialize_global_monitor, create_config_from_env


class LegalRAGCLI:
    """台灣法規RAG系統的CLI介面 - 重構版本，使用模組化組件"""

    def __init__(self, enable_monitoring: bool = True):
        """
        初始化 Legal RAG CLI.

        Args:
            enable_monitoring: 是否啟用 W&B 監控
        """
        # Initialize console
        try:
            self.console = Console(force_terminal=True, width=None)
            if sys.platform == "win32":
                self.console._file = sys.stdout
        except Exception:
            self.console = Console()

        # Initialize modular components
        self.renderer = CLIRenderer(self.console)
        self.env_validator = EnvironmentValidator()
        self.data_manager = DataManager()
        self.index_manager = IndexManager(
            self.data_manager,
            console=self.console,
            enable_monitoring=enable_monitoring
        )
        self.query_handler: Optional[QueryHandler] = None

        # Monitoring setup
        self.enable_monitoring = enable_monitoring
        self.monitor: Optional[WandbMonitor] = None
        self.session_start_time = time.time()

    def setup_monitoring(self):
        """設置 W&B 監控"""
        if not self.enable_monitoring:
            return

        try:
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv()

            # Check W&B settings
            wandb_mode = os.getenv("WANDB_MODE", "online")
            wandb_project = os.getenv("WANDB_PROJECT", "food-safety-rag")

            if wandb_mode == "disabled":
                self.renderer.display_warning("W&B 監控已停用")
                self.enable_monitoring = False
                return

            # Initialize monitor
            try:
                self.monitor = WandbMonitor(
                    project_name=wandb_project,
                    mode=wandb_mode,
                    tags=["cli-session", "legal-rag"]
                )
            except Exception as e:
                self.renderer.display_warning(f"監控器初始化失敗: {e}")
                self.enable_monitoring = False
                self.monitor = None
                return

            # Set as global monitor
            try:
                initialize_global_monitor(
                    project_name=wandb_project,
                    mode=wandb_mode,
                    tags=["cli-session", "legal-rag"]
                )
            except Exception as e:
                self.renderer.display_warning(f"全域監控器設定失敗: {e}")

            # Initialize W&B run
            if self.monitor:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                config = create_config_from_env()
                config["session_type"] = "cli"

                self.monitor.init_run(
                    run_name=f"cli_session_{timestamp}",
                    config=config
                )

                self.renderer.display_success(f"W&B 監控已啟用 - 專案: {wandb_project}")

        except Exception as e:
            self.renderer.display_warning(f"W&B 監控初始化失敗: {e}")
            self.enable_monitoring = False
            self.monitor = None

    def run(self, args):
        """
        主要執行邏輯 - 使用模組化組件簡化流程

        Args:
            args: Parsed command-line arguments
        """
        self.renderer.display_system_header()

        # 1. Initialize monitoring
        if self.enable_monitoring:
            self.setup_monitoring()

        # 2. Environment validation
        data_files = [
            self.data_manager.get_data_path(LawType.FOOD_SAFETY),
            self.data_manager.get_data_path(LawType.LABOR_LAW),
            self.data_manager.get_data_path(LawType.CIVIL_LAW)
        ]
        validation_result = self.env_validator.validate(data_files)

        if not validation_result.passed:
            self.renderer.display_validation_result(validation_result)

            # Log validation failure
            if self.monitor:
                self.monitor.log_metrics({
                    "environment_check_passed": False,
                    "environment_issues_count": validation_result.issues_count
                })

            if Confirm.ask("是否要進行初始設置？"):
                self.renderer.display_info("\n請先完成以下設置：")
                self.renderer.display_info("1. 複製 .env.template 為 .env")
                self.renderer.display_info("2. 在 .env 中設定您的 OPENAI_API_KEY")
                self.renderer.display_info("3. 重新執行程式")
            return

        self.renderer.display_success("環境檢查通過")

        # Log validation success
        if self.monitor:
            self.monitor.log_metrics({
                "environment_check_passed": True,
                "environment_issues_count": 0
            })

        # 3. Handle data fetching
        if args.fetch_food_data:
            self._fetch_data_with_progress(LawType.FOOD_SAFETY)

        if args.fetch_labor_data:
            self._fetch_data_with_progress(LawType.LABOR_LAW)

        if args.fetch_civil_data:
            self._fetch_data_with_progress(LawType.CIVIL_LAW)

        # 4. Handle index building
        if args.rebuild_food_index or (not Path("chroma_db").exists() and not args.multi_domain and args.domain != 'labor'):
            success, stats = self.index_manager.build_index(LawType.FOOD_SAFETY, reset=args.rebuild_food_index)
            if not success:
                return
            if stats:
                self.renderer.display_index_stats(stats, self.data_manager.get_law_name(LawType.FOOD_SAFETY))

        if args.rebuild_labor_index:
            success, stats = self.index_manager.build_index(LawType.LABOR_LAW, reset=True)
            if not success:
                return
            if stats:
                self.renderer.display_enhanced_stats(stats, self.data_manager.get_law_name(LawType.LABOR_LAW))

        if args.rebuild_civil_index:
            success, stats = self.index_manager.build_index(LawType.CIVIL_LAW, reset=True)
            if not success:
                return
            if stats:
                self.renderer.display_enhanced_stats(stats, self.data_manager.get_law_name(LawType.CIVIL_LAW))

        # 5. Handle statistics display
        if args.show_all_stats:
            self._show_all_stats()
            return

        # 6. Initialize query handler
        self.query_handler = QueryHandler(
            self.index_manager,
            self.renderer,
            enable_monitoring=self.enable_monitoring,
            monitor=self.monitor
        )

        # Update index_manager's monitor reference for consistency
        self.index_manager.monitor = self.monitor
        self.index_manager.enable_monitoring = self.enable_monitoring

        # 7. Execute query mode
        if args.multi_domain:
            self._run_multi_domain_mode(args)
        else:
            self._run_single_domain_mode(args)

        # 8. Finish monitoring session
        self._finish_monitoring_session()

    def _fetch_data_with_progress(self, law_type: LawType):
        """
        Fetch law data with progress indicator.

        Args:
            law_type: Type of law to fetch
        """
        law_name = self.data_manager.get_law_name(law_type)

        # Check if already exists
        if self.data_manager.data_exists(law_type):
            if not Confirm.ask(f"{law_name}資料檔案已存在，是否重新下載？"):
                return

        self.renderer.display_info(f"\n開始下載{law_name}...")

        # Special warning for civil law
        if law_type == LawType.CIVIL_LAW:
            self.renderer.display_warning("注意：民法有1229條，預計需要40分鐘以上時間")

        # Fetch with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task(f"下載{law_name}...", total=None)

            # Determine delay based on law type
            delay = 2.0 if law_type == LawType.CIVIL_LAW else 1.5 if law_type == LawType.LABOR_LAW else 1.0

            success = self.data_manager.fetch_and_save(law_type, delay=delay, overwrite=True)

            if success:
                article_count = self.data_manager.get_article_count(law_type)
                progress.update(task, description=f"完成！共 {article_count} 條")
                self.renderer.display_success(f"成功下載 {article_count} 條{law_name}")
            else:
                progress.update(task, description="失敗")
                self.renderer.display_error(f"{law_name}下載失敗")

    def _show_all_stats(self):
        """顯示所有法規索引統計資訊"""
        self.console.print("\n[bold blue]所有法規索引統計資訊[/bold blue]")

        for law_type in LawType:
            stats = self.index_manager.get_index_stats(law_type)
            law_name = self.data_manager.get_law_name(law_type)

            if stats:
                # Check if it's enhanced stats (has processing_stats key)
                if 'processing_stats' in stats and law_type != LawType.FOOD_SAFETY:
                    self.renderer.display_enhanced_stats(stats.get('processing_stats', {}), law_name)
                else:
                    self.renderer.display_index_stats(stats, law_name)
            else:
                self.renderer.display_warning(f"{law_name}索引不存在")

    def _run_single_domain_mode(self, args):
        """
        Execute single-domain query mode.

        Args:
            args: Command-line arguments
        """
        # Determine law type from args
        law_type = self._get_law_type_from_args(args)

        # Initialize single-domain system
        if not self.query_handler.initialize_single_domain_system(law_type):
            return

        # Execute appropriate query mode
        if args.batch_file:
            self.query_handler.batch_query(args.batch_file, multi_domain=False)
        elif args.query:
            self.query_handler.execute_single_query(args.query)
        else:
            self.query_handler.interactive_single_query()

    def _run_multi_domain_mode(self, args):
        """
        Execute multi-domain query mode.

        Args:
            args: Command-line arguments
        """
        # Initialize multi-domain system
        if not self.query_handler.initialize_multi_domain_system():
            return

        # Execute appropriate query mode
        if args.batch_file:
            self.query_handler.batch_query(args.batch_file, multi_domain=True)
        elif args.query:
            self.query_handler.execute_multi_domain_query(args.query)
        else:
            self.query_handler.interactive_multi_query()

    def _get_law_type_from_args(self, args) -> LawType:
        """
        Determine law type from command-line arguments.

        Args:
            args: Command-line arguments

        Returns:
            LawType enum value
        """
        if args.domain == 'labor':
            return LawType.LABOR_LAW
        elif args.domain == 'civil':
            return LawType.CIVIL_LAW
        else:
            return LawType.FOOD_SAFETY

    def _finish_monitoring_session(self):
        """完成監控會話並記錄最終統計"""
        if not self.monitor:
            return

        try:
            # Calculate total session time
            total_session_time = time.time() - self.session_start_time

            # Create final summary
            final_summary = {
                "total_session_time": total_session_time,
                "session_end_time": datetime.now().isoformat(),
                "monitoring_enabled": self.enable_monitoring
            }

            # Add query handler stats if available
            if self.query_handler:
                final_summary.update({
                    "session_queries": self.query_handler.session_queries,
                })

            self.monitor.create_summary(final_summary)
            self.monitor.finish_run()
            self.renderer.display_info("W&B 監控會話已結束")

        except Exception as e:
            self.renderer.display_warning(f"完成監控會話時發生錯誤: {e}")


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description="台灣法規 RAG 知識檢索系統 - 支持食品安全法、勞基法、民法及多法規整合查詢 (含 W&B 監控)",
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

    # Query parameters
    parser.add_argument("-q", "--query",
                       help="執行單一查詢")
    parser.add_argument("--multi-domain", action="store_true",
                       help="啟用多法規整合模式，支持智能路由和跨法規查詢")
    parser.add_argument("--domain", choices=["food", "labor", "civil", "all"], default="food",
                       help="指定查詢的法規領域 (預設: food)")

    # Data management parameters
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

    # Batch processing and statistics parameters
    parser.add_argument("--batch", dest="batch_file",
                       help="批次查詢檔案路徑")
    parser.add_argument("--all-stats", dest="show_all_stats", action="store_true",
                       help="顯示所有法規索引統計資訊")

    # System parameters
    parser.add_argument("--no-monitoring", action="store_true",
                       help="停用 W&B 監控")

    # Backward compatibility parameters
    parser.add_argument("--fetch-data", action="store_true",
                       help="(已廢棄) 請使用 --fetch-food-data")
    parser.add_argument("--rebuild-index", action="store_true",
                       help="(已廢棄) 請使用 --rebuild-food-index")
    parser.add_argument("--stats", dest="show_stats", action="store_true",
                       help="(已廢棄) 請使用 --all-stats")

    args = parser.parse_args()

    # Handle backward compatibility
    if args.fetch_data:
        args.fetch_food_data = True
        print("[WARNING] --fetch-data 已廢棄，自動轉換為 --fetch-food-data")
    if args.rebuild_index:
        args.rebuild_food_index = True
        print("[WARNING] --rebuild-index 已廢棄，自動轉換為 --rebuild-food-index")
    if args.show_stats:
        args.show_all_stats = True
        print("[WARNING] --stats 已廢棄，自動轉換為 --all-stats")

    # Check Python version
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