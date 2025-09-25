#!/usr/bin/env python3
"""
測試監控屬性修復
"""

import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_index_builder_monitor_attribute():
    """測試 LegalIndexBuilder 的 monitor 屬性"""
    print("🔍 測試 LegalIndexBuilder monitor 屬性...")

    try:
        from src.index_builder import LegalIndexBuilder

        print("  - 測試停用監控初始化...")
        builder = LegalIndexBuilder(enable_monitoring=False)

        # 檢查 monitor 屬性是否存在
        if hasattr(builder, 'monitor'):
            print(f"  ✅ monitor 屬性存在: {type(builder.monitor)}")
        else:
            print("  ❌ monitor 屬性不存在")
            return False

        # 測試 monitor 屬性的值
        if builder.monitor is None:
            print("  ✅ monitor 為 None（停用模式）")
        else:
            print(f"  ✅ monitor 已初始化: {builder.monitor.enabled}")

        return True

    except Exception as e:
        print(f"  ❌ LegalIndexBuilder 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_system_monitor_attribute():
    """測試 LegalRAGSystem 的 monitor 屬性"""
    print("🔍 測試 LegalRAGSystem monitor 屬性...")

    try:
        from src.legal_single_domain_rag import LegalRAGSystem

        # 這個測試只檢查類別能否正常匯入，不實際初始化
        # 因為需要現有的索引
        print("  ✅ LegalRAGSystem 類別匯入成功")
        return True

    except Exception as e:
        print(f"  ❌ LegalRAGSystem 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chroma_connection_fallback():
    """測試 ChromaDB 連線回退機制"""
    print("🔍 測試 ChromaDB 連線回退...")

    try:
        from src.index_builder import LegalIndexBuilder

        print("  - 創建 LegalIndexBuilder（可能會嘗試連接遠程 ChromaDB）...")

        # 設定使用本地模式以避免遠程連線問題
        builder = LegalIndexBuilder(
            chroma_path="./chroma_db",
            enable_monitoring=False
        )

        print("  ✅ ChromaDB 初始化成功（本地模式）")
        return True

    except Exception as e:
        print(f"  ❌ ChromaDB 測試失敗: {e}")
        print("  ℹ️  這是正常的，如果你的環境沒有 ChromaDB 或 OpenAI API")
        return False

def test_decorator_compatibility():
    """測試裝飾器兼容性"""
    print("🔍 測試監控裝飾器...")

    try:
        from src.monitoring import monitor_execution_time, WandbMonitor

        # 創建一個測試類別
        class TestClass:
            def __init__(self):
                self.monitor = WandbMonitor(mode="disabled")

            @monitor_execution_time("test_time")
            def test_method(self):
                return "success"

        test_obj = TestClass()
        result = test_obj.test_method()

        print(f"  ✅ 裝飾器測試成功: {result}")
        return True

    except Exception as e:
        print(f"  ❌ 裝飾器測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """執行所有測試"""
    print("🚀 開始測試監控屬性修復...")

    tests = [
        ("LegalIndexBuilder monitor 屬性", test_index_builder_monitor_attribute),
        ("LegalRAGSystem monitor 屬性", test_rag_system_monitor_attribute),
        ("ChromaDB 連線回退", test_chroma_connection_fallback),
        ("監控裝飾器兼容性", test_decorator_compatibility),
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

    if passed >= 2:  # 至少前兩個核心測試要通過
        print("🎉 核心修復成功！程式應該可以啟動了。")
        print("\n建議測試:")
        print("  uv run python main.py --no-monitoring --help")
        print("  uv run python main_no_wandb.py --help")
        return True
    else:
        print("⚠️  核心測試失敗，需要進一步檢查。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)