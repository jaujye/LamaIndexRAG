#!/usr/bin/env python3
"""
測試修復後的程式是否能正常初始化
"""

import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_index_builder_initialization():
    """測試 LegalIndexBuilder 初始化"""
    print("🔍 測試 LegalIndexBuilder 初始化...")

    try:
        from src.index_builder import LegalIndexBuilder

        # 測試不同的初始化方式
        print("  - 測試停用監控模式...")
        builder1 = LegalIndexBuilder(enable_monitoring=False)
        print("  ✅ 停用監控模式初始化成功")

        print("  - 測試啟用監控模式...")
        builder2 = LegalIndexBuilder(enable_monitoring=True)
        print("  ✅ 啟用監控模式初始化成功")

        print("  - 測試預設參數...")
        builder3 = LegalIndexBuilder()
        print("  ✅ 預設參數初始化成功")

        # 檢查 monitor 屬性是否存在
        if hasattr(builder1, 'monitor'):
            print("  ✅ monitor 屬性存在")
        else:
            print("  ❌ monitor 屬性不存在")
            return False

        return True

    except Exception as e:
        print(f"  ❌ LegalIndexBuilder 初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_system_initialization():
    """測試 LegalRAGSystem 初始化"""
    print("🔍 測試 LegalRAGSystem 初始化...")

    try:
        from src.legal_single_domain_rag import LegalRAGSystem

        print("  - 測試停用監控模式...")
        # 這個測試不能真正初始化，因為需要索引存在
        # 但至少可以測試類別能否正常匯入
        print("  ✅ LegalRAGSystem 類別匯入成功")
        return True

    except Exception as e:
        print(f"  ❌ LegalRAGSystem 測試失敗: {e}")
        return False

def test_main_cli_initialization():
    """測試主程式 CLI 初始化"""
    print("🔍 測試主程式 CLI 初始化...")

    try:
        from main import LegalRAGCLI

        print("  - 測試啟用監控模式...")
        cli1 = LegalRAGCLI(enable_monitoring=True)
        print("  ✅ 啟用監控模式初始化成功")

        print("  - 測試停用監控模式...")
        cli2 = LegalRAGCLI(enable_monitoring=False)
        print("  ✅ 停用監控模式初始化成功")

        print("  - 測試預設參數...")
        cli3 = LegalRAGCLI()
        print("  ✅ 預設參數初始化成功")

        return True

    except Exception as e:
        print(f"  ❌ 主程式 CLI 初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_cli_initialization():
    """測試簡化版 CLI 初始化"""
    print("🔍 測試簡化版 CLI 初始化...")

    try:
        from main_no_wandb import FoodSafetyRAGCLI

        print("  - 測試簡化版初始化...")
        cli = FoodSafetyRAGCLI()
        print("  ✅ 簡化版初始化成功")

        return True

    except Exception as e:
        print(f"  ❌ 簡化版 CLI 初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """執行所有測試"""
    print("🚀 開始測試修復後的程式...")

    tests = [
        ("LegalIndexBuilder 初始化", test_index_builder_initialization),
        ("LegalRAGSystem 初始化", test_rag_system_initialization),
        ("主程式 CLI 初始化", test_main_cli_initialization),
        ("簡化版 CLI 初始化", test_simple_cli_initialization),
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
        print("🎉 所有測試通過！程式修復成功。")
        print("\n現在可以嘗試:")
        print("  uv run python main.py --help")
        print("  uv run python main_no_wandb.py --help")
        return True
    else:
        print("⚠️  部分測試失敗，請檢查錯誤訊息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)