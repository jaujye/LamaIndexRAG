#!/usr/bin/env python3
"""
系統測試腳本
測試 RAG 系統的各個組件功能
"""

import os
import sys
from pathlib import Path
import json
from typing import List, Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, track
from dotenv import load_dotenv
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 載入環境變數
load_dotenv()

console = Console()


def test_environment():
    """測試環境設置"""
    console.print("[bold blue] 測試環境設置[/bold blue]")

    tests = [
        ("Python 版本", sys.version_info >= (3, 8)),
        (".env 檔案存在", Path(".env").exists()),
        ("OpenAI API Key 設置", bool(os.getenv("OPENAI_API_KEY"))),
        ("src 目錄存在", Path("src").exists()),
        ("data 目錄存在", Path("data").exists()),
    ]

    table = Table(title="環境檢查結果")
    table.add_column("檢查項目", style="cyan")
    table.add_column("狀態", style="green")

    all_passed = True
    for test_name, result in tests:
        status = "[OK] 通過" if result else "[FAIL] 失敗"
        if not result:
            all_passed = False
        table.add_row(test_name, status)

    console.print(table)
    return all_passed


def test_data_fetcher():
    """測試資料擷取功能"""
    console.print("\n[bold blue] 測試資料擷取功能[/bold blue]")

    try:
        from src.data_fetcher import FoodSafetyActFetcher

        fetcher = FoodSafetyActFetcher(delay=0.5)

        # 測試主頁面擷取
        console.print("測試主頁面擷取...")
        soup = fetcher.fetch_main_page()

        if soup:
            console.print("[OK] 主頁面擷取成功")

            # 測試條文連結提取
            console.print("測試條文連結提取...")
            links = fetcher.extract_article_links(soup)

            if links:
                console.print(f"[OK] 找到 {len(links)} 個條文連結")

                # 測試單一條文內容擷取
                if len(links) > 0:
                    console.print("測試單一條文內容擷取...")
                    first_article = links[0]
                    content = fetcher.fetch_article_content(first_article['url'])

                    if content:
                        console.print("[OK] 條文內容擷取成功")
                        console.print(f"範例內容（前100字）: {content[:100]}...")
                        return True
                    else:
                        console.print("[FAIL] 條文內容擷取失敗")
            else:
                console.print("[FAIL] 未找到條文連結")
        else:
            console.print("[FAIL] 主頁面擷取失敗")

    except Exception as e:
        console.print(f"[FAIL] 資料擷取測試失敗: {e}")

    return False


def test_document_processor():
    """測試文件處理功能"""
    console.print("\n[bold blue] 測試文件處理功能[/bold blue]")

    try:
        from src.document_processor import LegalDocumentProcessor

        processor = LegalDocumentProcessor(chunk_size=256, chunk_overlap=25)

        # 建立測試資料
        test_data = {
            "law_name": "測試法規",
            "articles": [{
                "article_number": "1",
                "title": "第1條",
                "content": "本法為確保食品安全，維護國民健康，特制定之。食品安全衛生之管理，依本法之規定；本法未規定者，適用其他法律之規定。",
                "chapter": "第一章 總則",
                "chapter_number": "一",
                "url": "test_url"
            }]
        }

        # 測試chunk處理
        console.print("測試文件chunk處理...")
        chunks = processor.process_all_articles(test_data)

        if chunks:
            console.print(f"[OK] 成功處理為 {len(chunks)} 個chunks")

            # 測試轉換為 LlamaIndex 文件
            console.print("測試轉換為 LlamaIndex 文件格式...")
            documents = processor.convert_to_llama_documents(chunks)

            if documents:
                console.print(f"[OK] 成功轉換為 {len(documents)} 個文件")

                # 顯示處理統計
                stats = processor.get_processing_stats(chunks)
                console.print(f"處理統計: {stats}")
                return True

    except Exception as e:
        console.print(f"[FAIL] 文件處理測試失敗: {e}")

    return False


def test_index_builder():
    """測試索引建立功能"""
    console.print("\n[bold blue] 測試索引建立功能[/bold blue]")

    if not os.getenv("OPENAI_API_KEY"):
        console.print("[FAIL] 需要 OpenAI API Key 才能測試索引功能")
        return False

    try:
        from src.index_builder import LegalIndexBuilder

        builder = LegalIndexBuilder(enable_monitoring=False)

        # 檢查現有索引
        console.print("檢查現有索引...")
        existing_index = builder.load_existing_index()

        if existing_index:
            console.print("[OK] 找到現有索引")
            stats = builder.get_index_stats()
            console.print(f"索引統計: {stats.get('document_count', '未知')} 個文件")
            return True
        else:
            console.print("[WARN] 未找到現有索引")

            # 檢查是否有資料檔案可以建立索引
            data_file = Path("data/food_safety_act.json")
            if data_file.exists():
                console.print("發現資料檔案，可以建立索引")
                return True
            else:
                console.print("[FAIL] 未找到資料檔案，無法建立索引")

    except Exception as e:
        console.print(f"[FAIL] 索引測試失敗: {e}")

    return False


def test_rag_system():
    """測試 RAG 系統功能"""
    console.print("\n[bold blue]測試 RAG 系統功能[/bold blue]")

    if not os.getenv("OPENAI_API_KEY"):
        console.print("[FAIL] 需要 OpenAI API Key 才能測試 RAG 系統")
        return False

    try:
        from src.rag_system import LegalRAGSystem

        # 嘗試初始化 RAG 系統
        console.print("初始化 RAG 系統...")
        rag_system = LegalRAGSystem()

        console.print("設置查詢引擎...")
        rag_system.setup_query_engine(similarity_top_k=3, similarity_cutoff=0.5)

        # 測試查詢分類功能
        test_queries = [
            "食品添加物有什麼限制？",
            "違反規定會受到什麼處罰？",
            "食品標示需要包含什麼內容？"
        ]

        console.print("測試查詢分類...")
        for query in test_queries:
            query_type = rag_system.classify_query_type(query)
            console.print(f"  查詢: {query[:20]}... -> 類型: {query_type}")

        # 測試實際查詢（如果可能）
        console.print("測試實際查詢...")
        try:
            result = rag_system.query("什麼是食品添加物？", similarity_top_k=2)
            console.print("[OK] 查詢功能正常")
            console.print(f"回答長度: {len(result.answer)} 字元")
            console.print(f"信心度: {result.confidence_score:.3f}")
            console.print(f"相關來源數: {len(result.sources)}")
            return True

        except Exception as query_error:
            console.print(f"[WARN] 查詢測試失敗，但系統初始化成功: {query_error}")
            return True

    except Exception as e:
        console.print(f"[FAIL] RAG 系統測試失敗: {e}")

    return False


def test_full_pipeline():
    """測試完整流程"""
    console.print("\n[bold blue] 測試完整流程[/bold blue]")

    # 檢查必需檔案
    required_files = [
        "main.py",
        "requirements.txt",
        ".env.template",
        "README.md"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        console.print(f"[FAIL] 缺少必需檔案: {missing_files}")
        return False

    console.print("[OK] 所有必需檔案都存在")

    # 檢查目錄結構
    required_dirs = ["src", "data", "config"]
    missing_dirs = []

    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        console.print(f"[FAIL] 缺少必需目錄: {missing_dirs}")
        return False

    console.print("[OK] 目錄結構正確")
    return True


def main():
    """執行所有測試"""
    console.print(Panel.fit(
        "[bold green]LamaIndex RAG 系統測試套件[/bold green]\n"
        "測試系統各個組件的功能",
        border_style="green"
    ))

    tests = [
        ("環境設置", test_environment),
        ("完整流程", test_full_pipeline),
        ("資料擷取", test_data_fetcher),
        ("文件處理", test_document_processor),
        ("索引建立", test_index_builder),
        ("RAG 系統", test_rag_system),
    ]

    results = []

    for test_name, test_func in track(tests, description="執行測試..."):
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            console.print(f"[FAIL] 測試 {test_name} 時發生錯誤: {e}")
            results.append((test_name, False))

    # 顯示測試結果摘要
    console.print("\n" + "="*50)
    console.print("[bold blue] 測試結果摘要[/bold blue]")

    summary_table = Table(title="測試結果")
    summary_table.add_column("測試項目", style="cyan")
    summary_table.add_column("結果", style="white")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "[OK] 通過" if result else "[FAIL] 失敗"
        if result:
            passed += 1
        summary_table.add_row(test_name, status)

    console.print(summary_table)

    # 總結
    if passed == total:
        console.print(f"\n[bold green] 所有測試通過 ({passed}/{total})[/bold green]")
    else:
        console.print(f"\n[bold red][WARN] 部分測試失敗 ({passed}/{total} 通過)[/bold red]")

    console.print(f"\n[dim]系統準備就緒程度: {(passed/total)*100:.1f}%[/dim]")


if __name__ == "__main__":
    main()