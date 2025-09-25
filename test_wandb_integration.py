#!/usr/bin/env python3
"""
測試 W&B 監控整合功能
驗證監控模組是否正常工作
"""

import os
import sys
import time
from pathlib import Path

# 添加 src 模組到路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.monitoring import WandbMonitor, RAGMetrics, create_config_from_env
    print("✅ 成功匯入監控模組")
except ImportError as e:
    print(f"❌ 無法匯入監控模組: {e}")
    sys.exit(1)


def test_wandb_monitor_disabled():
    """測試停用模式的 W&B 監控器"""
    print("\n🧪 測試停用模式的監控器...")

    try:
        # 使用停用模式
        monitor = WandbMonitor(mode="disabled")
        print("✅ 成功建立停用模式監控器")

        # 測試基本操作
        monitor.log_metrics({"test_metric": 1.0})
        monitor.log_error("TestError", "這是測試錯誤")
        print("✅ 停用模式下的操作正常")

        return True

    except Exception as e:
        print(f"❌ 停用模式測試失敗: {e}")
        return False


def test_rag_metrics():
    """測試 RAG 指標資料類別"""
    print("\n🧪 測試 RAG 指標...")

    try:
        metrics = RAGMetrics(
            query_text="測試查詢",
            query_type="test",
            total_time=1.5,
            documents_retrieved=5,
            similarity_scores=[0.8, 0.7, 0.6, 0.5, 0.4],
            response_length=100,
            confidence_score=0.75
        )

        print(f"✅ RAG 指標建立成功:")
        print(f"  查詢: {metrics.query_text}")
        print(f"  平均相似度: {metrics.avg_similarity:.3f}")
        print(f"  信心度: {metrics.confidence_score:.3f}")

        return True

    except Exception as e:
        print(f"❌ RAG 指標測試失敗: {e}")
        return False


def test_config_creation():
    """測試配置建立"""
    print("\n🧪 測試配置建立...")

    try:
        config = create_config_from_env()
        print("✅ 成功建立配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ 配置建立測試失敗: {e}")
        return False


def test_monitor_with_init():
    """測試包含初始化的監控器"""
    print("\n🧪 測試監控器初始化...")

    try:
        # 設定測試環境變數
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_PROJECT"] = "test-project"

        monitor = WandbMonitor()

        # 測試初始化
        if monitor.enabled:
            config = create_config_from_env()
            monitor.init_run("test_run", config)
            print("✅ 監控器初始化成功")

            # 記錄一些測試指標
            test_metrics = RAGMetrics(
                query_text="食品添加物的規定",
                query_type="additives",
                total_time=2.1,
                documents_retrieved=3,
                similarity_scores=[0.85, 0.72, 0.68],
                response_length=150,
                confidence_score=0.82
            )

            monitor.log_metrics(test_metrics)
            monitor.log_query_result(
                "食品添加物的規定",
                "根據法規...",
                [{"article_number": "15", "similarity_score": 0.85, "text_preview": "預覽文字..."}],
                test_metrics
            )

            monitor.finish_run()
            print("✅ 監控資料記錄成功")
        else:
            print("ℹ️  監控器已停用，跳過初始化測試")

        return True

    except Exception as e:
        print(f"❌ 監控器初始化測試失敗: {e}")
        return False


def main():
    """執行所有測試"""
    print("🚀 開始測試 W&B 監控整合...")

    tests = [
        ("停用模式監控器", test_wandb_monitor_disabled),
        ("RAG 指標", test_rag_metrics),
        ("配置建立", test_config_creation),
        ("監控器初始化", test_monitor_with_init),
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
        print("🎉 所有測試都通過了！W&B 監控整合準備就緒。")
        return True
    else:
        print("⚠️  有些測試失敗。請檢查設定和依賴項。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)