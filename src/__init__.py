"""
台灣食品安全衛生管理法 RAG 知識檢索系統

主要模組：
- legal_food_safety_fetcher: 食品安全法規資料擷取
- legal_basic_processor: 基礎文件處理和分塊
- index_builder: 向量索引建立
- legal_single_domain_rag: 單一域RAG查詢系統
"""

__version__ = "1.0.0"
__author__ = "LamaIndex RAG System"
__description__ = "台灣食品安全衛生管理法 RAG 知識檢索系統"

from .legal_food_safety_fetcher import FoodSafetyActFetcher
from .legal_basic_processor import LegalDocumentProcessor
from .index_builder import LegalIndexBuilder
from .legal_single_domain_rag import LegalRAGSystem

__all__ = [
    "FoodSafetyActFetcher",
    "LegalDocumentProcessor",
    "LegalIndexBuilder",
    "LegalRAGSystem"
]