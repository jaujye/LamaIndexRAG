"""
台灣食品安全衛生管理法 RAG 知識檢索系統

主要模組：
- data_fetcher: 法規資料擷取
- document_processor: 文件處理和分塊
- index_builder: 向量索引建立
- rag_system: RAG 查詢系統
"""

__version__ = "1.0.0"
__author__ = "LamaIndex RAG System"
__description__ = "台灣食品安全衛生管理法 RAG 知識檢索系統"

from .data_fetcher import FoodSafetyActFetcher
from .document_processor import LegalDocumentProcessor
from .index_builder import LegalIndexBuilder
from .rag_system import LegalRAGSystem

__all__ = [
    "FoodSafetyActFetcher",
    "LegalDocumentProcessor",
    "LegalIndexBuilder",
    "LegalRAGSystem"
]