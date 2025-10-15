#!/usr/bin/env python3
"""
配置驗證腳本
用於檢查系統配置是否正確
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from src.config_validator import ConfigValidator


def main():
    """驗證系統配置"""
    try:
        console = Console(force_terminal=True, legacy_windows=False)
    except:
        console = Console()

    console.print(Panel.fit(
        "[bold blue]RAG 系統配置驗證工具[/bold blue]",
        border_style="blue"
    ))

    # 執行驗證
    validator = ConfigValidator()
    result = validator.validate_all()

    # 顯示結果
    if result.passed:
        console.print("\n[green]✅ 配置驗證通過！[/green]\n")
    else:
        console.print("\n[red]❌ 配置驗證失敗！[/red]\n")

    # 顯示錯誤
    if result.errors:
        console.print("[bold red]錯誤：[/bold red]")
        for error in result.errors:
            console.print(f"  {error}")
        console.print()

    # 顯示警告
    if result.warnings:
        console.print("[bold yellow]警告：[/bold yellow]")
        for warning in result.warnings:
            console.print(f"  {warning}")
        console.print()

    # 顯示配置摘要
    if result.config_summary:
        console.print("[bold cyan]配置摘要：[/bold cyan]")
        important_keys = [
            ('llm_model', 'LLM 模型'),
            ('embedding_model', '嵌入模型'),
            ('temperature', '溫度參數'),
            ('WANDB_MODE', 'W&B 模式')
        ]

        for key, label in important_keys:
            if key in result.config_summary:
                value = result.config_summary[key]
                console.print(f"  [cyan]{label}:[/cyan] {value}")
        console.print()

    # 顯示建議
    if not result.passed:
        console.print("[bold yellow]💡 推薦配置：[/bold yellow]")
        recommendations = validator.get_config_recommendations()
        for key, value in recommendations.items():
            console.print(f"  [cyan]{key}:[/cyan] {value}")
        console.print()

        console.print("[yellow]請修改 .env 文件以修復配置問題[/yellow]")
        sys.exit(1)
    else:
        console.print("[green]系統已準備就緒，可以啟動 RAG 系統 ✨[/green]")
        sys.exit(0)


if __name__ == "__main__":
    main()
