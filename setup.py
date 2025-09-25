#!/usr/bin/env python3
"""
LamaIndex RAG 系統安裝腳本
自動化設置環境和初始配置
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def check_python_version():
    """檢查 Python 版本"""
    if sys.version_info < (3, 8):
        console.print("❌ 需要 Python 3.8 或更高版本")
        console.print(f"目前版本: {sys.version}")
        return False

    console.print(f"✅ Python 版本檢查通過: {sys.version.split()[0]}")
    return True


def install_requirements():
    """安裝 Python 套件"""
    console.print("\n[yellow]安裝 Python 套件...[/yellow]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            task = progress.add_task("安裝套件中...", total=None)

            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True, check=True)

            progress.update(task, description="套件安裝完成！")

        console.print("✅ Python 套件安裝成功")
        return True

    except subprocess.CalledProcessError as e:
        console.print(f"❌ 套件安裝失敗: {e}")
        console.print(f"錯誤輸出: {e.stderr}")
        return False


def setup_environment():
    """設置環境檔案"""
    console.print("\n[yellow]設置環境檔案...[/yellow]")

    env_template = Path(".env.template")
    env_file = Path(".env")

    if not env_template.exists():
        console.print("❌ 找不到 .env.template 檔案")
        return False

    if env_file.exists():
        if not Confirm.ask("發現現有的 .env 檔案，是否要覆蓋？"):
            console.print("✅ 保留現有 .env 檔案")
            return True

    # 複製範本檔案
    shutil.copy(env_template, env_file)

    # 詢問 OpenAI API Key
    api_key = Prompt.ask(
        "\n請輸入您的 OpenAI API Key",
        password=True
    )

    if api_key:
        # 讀取並更新 .env 檔案
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替換 API key
        content = content.replace(
            'OPENAI_API_KEY=your_openai_api_key_here',
            f'OPENAI_API_KEY={api_key}'
        )

        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(content)

        console.print("✅ OpenAI API Key 已設定")
    else:
        console.print("⚠️ 未設定 OpenAI API Key，請稍後手動設定")

    console.print("✅ 環境檔案設置完成")
    return True


def create_directories():
    """建立必要的目錄"""
    console.print("\n[yellow]建立目錄結構...[/yellow]")

    directories = [
        "data",
        "config",
        "logs"
    ]

    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            console.print(f"✅ 建立目錄: {dir_name}")
        else:
            console.print(f"📁 目錄已存在: {dir_name}")

    return True


def run_initial_test():
    """執行初始測試"""
    console.print("\n[yellow]執行系統測試...[/yellow]")

    try:
        result = subprocess.run([
            sys.executable, "test_system.py"
        ], capture_output=False, check=True)

        console.print("✅ 系統測試完成")
        return True

    except subprocess.CalledProcessError as e:
        console.print(f"⚠️ 系統測試發現問題: {e}")
        return False


def main():
    """主要安裝流程"""
    console.print(Panel.fit(
        "[bold green]🚀 LamaIndex RAG 系統安裝程式[/bold green]\n"
        "自動設置台灣食品安全衛生管理法 RAG 知識檢索系統",
        border_style="green"
    ))

    # 檢查 Python 版本
    if not check_python_version():
        sys.exit(1)

    # 安裝套件
    if not install_requirements():
        console.print("\n❌ 安裝失敗，請檢查錯誤訊息")
        sys.exit(1)

    # 設置環境
    if not setup_environment():
        console.print("\n❌ 環境設置失敗")
        sys.exit(1)

    # 建立目錄
    if not create_directories():
        console.print("\n❌ 目錄建立失敗")
        sys.exit(1)

    # 詢問是否執行測試
    if Confirm.ask("\n是否要執行系統測試？"):
        run_initial_test()

    # 完成訊息
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold green]🎉 安裝完成！[/bold green]\n\n"
        "接下來您可以：\n"
        "1. 執行 [cyan]python main.py[/cyan] 開始使用系統\n"
        "2. 執行 [cyan]python test_system.py[/cyan] 再次測試系統\n"
        "3. 查看 [cyan]README.md[/cyan] 了解更多使用方法\n\n"
        "[dim]如果遇到問題，請檢查 .env 檔案中的 API key 設定[/dim]",
        border_style="green"
    ))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n❌ 安裝已中斷")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n\n❌ 安裝過程發生錯誤: {e}")
        sys.exit(1)