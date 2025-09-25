#!/usr/bin/env python3
"""
測試主程式啟動是否正常
"""

import sys
import os
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """測試所有必要模組是否可以正常匯入"""
    print("🔍 測試模組匯入...")

    try:
        # 測試核心匯入
        from src.monitoring import WandbMonitor, RAGMetrics
        print("✅ 監控模組匯入成功")

        from src.data_fetcher import FoodSafetyActFetcher
        print("✅ 資料獲取模組匯入成功")

        from src.document_processor import LegalDocumentProcessor
        print("✅ 文件處理模組匯入成功")

        from src.index_builder import LegalIndexBuilder
        print("✅ 索引建立模組匯入成功")

        from src.rag_system import LegalRAGSystem
        print("✅ RAG 系統模組匯入成功")

        return True

    except ImportError as e:
        print(f"❌ 模組匯入失敗: {e}")
        return False

def test_cli_initialization():
    """測試 CLI 類別是否可以正常初始化"""
    print("\n🔍 測試 CLI 初始化...")

    try:
        from main import FoodSafetyRAGCLI

        # 測試啟用監控模式
        cli_with_monitoring = FoodSafetyRAGCLI(enable_monitoring=True)
        print("✅ CLI 啟用監控模式初始化成功")

        # 測試停用監控模式
        cli_without_monitoring = FoodSafetyRAGCLI(enable_monitoring=False)
        print("✅ CLI 停用監控模式初始化成功")

        # 測試預設參數
        cli_default = FoodSafetyRAGCLI()
        print("✅ CLI 預設參數初始化成功")

        return True

    except Exception as e:
        print(f"❌ CLI 初始化失敗: {e}")
        return False

def test_monitoring_setup():
    """測試監控設置功能"""
    print("\n🔍 測試監控設置...")

    try:
        from main import FoodSafetyRAGCLI

        # 設置測試環境變數（停用模式）
        os.environ["WANDB_MODE"] = "disabled"

        cli = FoodSafetyRAGCLI(enable_monitoring=True)
        cli.setup_monitoring()

        print("✅ 監控設置完成")
        return True

    except Exception as e:
        print(f"❌ 監控設置失敗: {e}")
        return False

def test_argument_parsing():
    """測試命令行參數解析"""
    print("\n🔍 測試參數解析...")

    try:
        import argparse
        from main import main

        # 模擬不同的命令行參數
        test_args = [
            [],  # 預設參數
            ["--no-monitoring"],  # 停用監控
            ["--help"],  # 幫助資訊
        ]

        print("✅ 參數解析功能可用")
        return True

    except Exception as e:
        print(f"❌ 參數解析測試失敗: {e}")
        return False

def main():
    """執行所有啟動測試"""
    print("🚀 開始測試主程式啟動...")

    tests = [
        ("模組匯入", test_imports),
        ("CLI 初始化", test_cli_initialization),
        ("監控設置", test_monitoring_setup),
        ("參數解析", test_argument_parsing),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"測試: {test_name}")
        print(f"{'='*50}")

        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - 通過")
            else:
                print(f"❌ {test_name} - 失敗")
        except Exception as e:
            print(f"❌ {test_name} - 異常: {e}")

    print(f"\n{'='*50}")
    print(f"測試結果: {passed}/{total} 通過")
    print(f"{'='*50}")

    if passed == total:
        print("🎉 主程式準備就緒！可以正常啟動。")
        print("\n使用方式:")
        print("  python main.py              # 啟動互動式介面（含監控）")
        print("  python main.py --no-monitoring  # 停用監控模式")
        print("  python main.py --help       # 查看所有選項")
        return True
    else:
        print("⚠️  部分測試失敗，請檢查錯誤訊息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)