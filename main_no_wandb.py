#!/usr/bin/env python3
"""
台灣食品安全衛生管理法 RAG 知識檢索系統
主要CLI介面程式 (不含 W&B 監控)

[WARNING] DEPRECATION WARNING:
    此檔案已被棄用。請使用更強大的模組化版本:

    python main.py --no-monitoring

    新版本支援:
    - 食品安全法、勞基法、民法 (三種法規)
    - 單一法規與多法規整合查詢
    - 更好的架構與可維護性

    此檔案僅保留作為向後相容性。
"""

import os
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List
import json
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich import print as rprint

from src.legal_food_safety_fetcher import FoodSafetyActFetcher
from src.legal_basic_processor import LegalDocumentProcessor
from src.index_builder import LegalIndexBuilder
from src.legal_single_domain_rag import LegalRAGSystem


class FoodSafetyRAGCLI:
    """食品安全法RAG系統的CLI介面（簡化版，不含監控）"""

    def __init__(self):
        self.console = Console()
        self.data_file = Path("data/food_safety_act.json")
        self.env_file = Path(".env")

        # 系統組件
        self.rag_system: Optional[LegalRAGSystem] = None
        self.index_builder: Optional[LegalIndexBuilder] = None

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
        if not self.data_file.exists():
            issues.append("[FAIL] 未找到法規資料檔案，需要先下載法規內容")

        if issues:
            self.console.print("\n[red]環境設置問題：[/red]")
            for issue in issues:
                self.console.print(f"  {issue}")
            return False

        self.console.print("[OK] 環境檢查通過")
        return True

    def setup_data(self) -> bool:
        """設置資料（下載法規）"""
        if self.data_file.exists():
            if not Confirm.ask("資料檔案已存在，是否重新下載？"):
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
                fetcher.save_to_json(str(self.data_file))

                progress.update(task, description="完成！")

            self.console.print(f"[OK] 成功下載 {len(articles)} 條法規")
            return True

        except Exception as e:
            self.console.print(f"[FAIL] 下載失敗: {e}")
            return False

    def build_index(self, reset: bool = False) -> bool:
        """建立向量索引"""
        try:
            # 使用 LegalIndexBuilder（停用監控）
            self.index_builder = LegalIndexBuilder(enable_monitoring=False)

            # 檢查是否已有索引
            existing_index = self.index_builder.load_existing_index()

            if existing_index and not reset:
                self.console.print("[OK] 找到現有索引，跳過建立步驟")
                return True

            self.console.print("\n[yellow]建立向量索引...[/yellow]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:

                task = progress.add_task("處理文件並建立索引...", total=None)

                index = self.index_builder.build_index_from_json(
                    str(self.data_file),
                    reset=reset
                )

                progress.update(task, description="索引建立完成！")

            # 顯示統計資訊
            stats = self.index_builder.get_index_stats()
            self.show_index_stats(stats)

            return True

        except Exception as e:
            self.console.print(f"[FAIL] 索引建立失敗: {e}")
            return False

    def initialize_rag_system(self) -> bool:
        """初始化RAG系統"""
        try:
            self.console.print("\n[yellow]初始化RAG查詢系統...[/yellow]")

            # 使用原始的 LegalRAGSystem（不含監控）
            self.rag_system = LegalRAGSystem()
            self.rag_system.setup_query_engine(
                similarity_top_k=10,
                similarity_cutoff=0.3
            )

            self.console.print("[OK] RAG系統初始化完成")
            return True

        except Exception as e:
            self.console.print(f"[FAIL] RAG系統初始化失敗: {e}")
            return False

    def show_index_stats(self, stats: dict):
        """顯示索引統計資訊"""
        table = Table(title="索引統計資訊")
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

    def query_interface(self):
        """互動式查詢介面"""
        self.console.print("\n" + "="*60)
        self.console.print(Panel.fit(
            "[bold blue]台灣食品安全衛生管理法 RAG 知識檢索系統[/bold blue]\n"
            "輸入您的問題，系統將基於法規內容為您解答\n"
            "[dim]輸入 'quit' 或 'exit' 結束程式[/dim]\n"
            "[dim]💡 簡化版（不含 W&B 監控）[/dim]",
            border_style="blue"
        ))

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

                result = self.rag_system.query(question)

                # 顯示結果
                self.display_query_result(result)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[FAIL] 查詢錯誤: {e}")

        self.console.print("\n[green]感謝使用食品安全法規知識檢索系統！[/green]")

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

    def run(self, args):
        """主要執行邏輯"""
        self.console.print("[bold blue]食品安全衛生管理法 RAG 系統[/bold blue]")
        self.console.print("[dim]簡化版（不含 W&B 監控）[/dim]\n")

        # 環境檢查
        if not self.check_environment():
            if Confirm.ask("是否要進行初始設置？"):
                self.console.print("\n[yellow]請先完成以下設置：[/yellow]")
                self.console.print("1. 複製 .env.template 為 .env")
                self.console.print("2. 在 .env 中設定您的 OPENAI_API_KEY")
                self.console.print("3. 重新執行程式")
            return

        # 設置資料
        if args.fetch_data or not self.data_file.exists():
            if not self.setup_data():
                return

        # 建立索引
        if args.rebuild_index or not Path("chroma_db").exists():
            if not self.build_index(reset=args.rebuild_index):
                return
        else:
            # 檢查現有索引
            try:
                self.index_builder = LegalIndexBuilder(enable_monitoring=False)
                if self.index_builder.load_existing_index():
                    stats = self.index_builder.get_index_stats()
                    self.console.print("[OK] 載入現有索引")
                    if args.show_stats:
                        self.show_index_stats(stats)
                else:
                    if not self.build_index():
                        return
            except Exception as e:
                self.console.print(f"載入索引失敗: {e}")
                if not self.build_index(reset=True):
                    return

        # 初始化RAG系統
        if not self.initialize_rag_system():
            return

        # 執行對應模式
        if args.batch_file:
            self.batch_query_mode(args.batch_file)
        elif args.query:
            result = self.rag_system.query(args.query)
            self.display_query_result(result)
        else:
            self.query_interface()


def main():
    """主程式入口"""
    # Display deprecation warning
    print("\n" + "="*70)
    print("[WARNING] DEPRECATION WARNING")
    print("="*70)
    print("main_no_wandb.py 已被棄用，請改用:")
    print("  python main.py --no-monitoring")
    print("\n新版本功能更完整，支援三種法規與多領域整合查詢。")
    print("="*70 + "\n")

    parser = argparse.ArgumentParser(
        description="台灣食品安全衛生管理法 RAG 知識檢索系統 (簡化版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python main_no_wandb.py                          # 啟動互動式查詢介面
  python main_no_wandb.py -q "食品添加物的限制？"      # 單一查詢
  python main_no_wandb.py --fetch-data              # 重新下載法規資料
  python main_no_wandb.py --rebuild-index           # 重建向量索引
  python main_no_wandb.py --batch questions.txt     # 批次查詢
        """
    )

    parser.add_argument("-q", "--query",
                       help="執行單一查詢")
    parser.add_argument("--fetch-data", action="store_true",
                       help="重新下載法規資料")
    parser.add_argument("--rebuild-index", action="store_true",
                       help="重建向量索引")
    parser.add_argument("--batch", dest="batch_file",
                       help="批次查詢檔案路徑")
    parser.add_argument("--stats", dest="show_stats", action="store_true",
                       help="顯示索引統計資訊")

    args = parser.parse_args()

    # 檢查Python版本
    if sys.version_info < (3, 8):
        print("[FAIL] 需要 Python 3.8 或更高版本")
        sys.exit(1)

    try:
        cli = FoodSafetyRAGCLI()
        cli.run(args)
    except KeyboardInterrupt:
        print("\n程式已中止")
    except Exception as e:
        print(f"[FAIL] 程式錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()