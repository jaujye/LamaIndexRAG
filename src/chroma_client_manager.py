"""
ChromaDB 客戶端管理器
提供全局單例客戶端，避免多實例衝突
"""

from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings


class ChromaClientManager:
    """
    ChromaDB 客戶端管理器 - 單例模式
    確保整個應用程序只有一個 ChromaDB 客戶端實例
    """

    _instance: Optional['ChromaClientManager'] = None
    _client: Optional[chromadb.ClientAPI] = None
    _is_remote: bool = False

    def __new__(cls):
        """實現單例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_standard_settings(cls, allow_reset: bool = True) -> Settings:
        """
        獲取標準的 ChromaDB Settings 配置
        所有客戶端實例都應使用此配置以確保一致性

        Args:
            allow_reset: 是否允許重置數據庫

        Returns:
            統一的 Settings 配置
        """
        return Settings(
            anonymized_telemetry=False,
            allow_reset=allow_reset,
            chroma_server_ssl_enabled=False,
            chroma_server_cors_allow_origins=["*"]
        )

    @classmethod
    def get_standard_path(cls) -> str:
        """
        獲取標準的 ChromaDB 路徑
        確保所有地方使用相同的路徑字符串

        Returns:
            標準化的路徑字符串
        """
        return "chroma_db"

    @classmethod
    def get_client(
        cls,
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
        force_local: bool = False
    ) -> chromadb.ClientAPI:
        """
        獲取或創建 ChromaDB 客戶端（單例）

        Args:
            chroma_host: ChromaDB 服務器主機（遠程模式）
            chroma_port: ChromaDB 服務器端口（遠程模式）
            force_local: 強制使用本地模式

        Returns:
            ChromaDB 客戶端實例
        """
        # 如果已存在客戶端，直接返回
        if cls._client is not None:
            return cls._client

        # 確定使用遠程還是本地模式
        use_remote = (not force_local) and (chroma_host is not None) and (chroma_port is not None)

        try:
            if use_remote:
                # 嘗試連接遠程 ChromaDB
                print(f"[INFO] Attempting to connect to remote ChromaDB at {chroma_host}:{chroma_port}")

                cls._client = chromadb.HttpClient(
                    host=chroma_host,
                    port=chroma_port,
                    settings=cls.get_standard_settings()
                )

                # 測試連接
                cls._client.heartbeat()
                cls._is_remote = True
                print(f"[OK] Connected to remote ChromaDB at {chroma_host}:{chroma_port}")

            else:
                # 使用本地 ChromaDB
                db_path = cls.get_standard_path()
                Path(db_path).mkdir(parents=True, exist_ok=True)

                print(f"[INFO] Using local ChromaDB at: {db_path}")

                cls._client = chromadb.PersistentClient(
                    path=db_path,
                    settings=cls.get_standard_settings()
                )
                cls._is_remote = False
                print(f"[OK] Local ChromaDB client initialized at: {db_path}")

        except Exception as e:
            # 如果是遠程連接失敗，回退到本地模式
            if use_remote:
                print(f"[WARN] Remote ChromaDB connection failed: {e}")
                print("[INFO] Falling back to local ChromaDB...")

                db_path = cls.get_standard_path()
                Path(db_path).mkdir(parents=True, exist_ok=True)

                cls._client = chromadb.PersistentClient(
                    path=db_path,
                    settings=cls.get_standard_settings()
                )
                cls._is_remote = False
                print(f"[OK] Local ChromaDB client initialized at: {db_path}")
            else:
                # 本地連接失敗，拋出異常
                raise RuntimeError(f"Failed to initialize ChromaDB client: {e}")

        return cls._client

    @classmethod
    def is_remote(cls) -> bool:
        """
        檢查當前客戶端是否為遠程模式

        Returns:
            True 如果是遠程模式，False 如果是本地模式
        """
        return cls._is_remote

    @classmethod
    def reset_client(cls):
        """
        重置客戶端實例（主要用於測試）
        警告：生產環境不建議使用
        """
        cls._client = None
        cls._is_remote = False

    @classmethod
    def get_client_info(cls) -> dict:
        """
        獲取客戶端信息

        Returns:
            包含客戶端狀態的字典
        """
        return {
            "initialized": cls._client is not None,
            "is_remote": cls._is_remote,
            "path": cls.get_standard_path() if not cls._is_remote else None
        }
