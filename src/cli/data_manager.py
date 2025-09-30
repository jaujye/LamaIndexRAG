"""
Data Manager
Manages legal document data fetching, loading, and file operations
"""

from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

from src.labor_law_fetcher import LaborLawFetcher
from src.civil_law_fetcher import CivilLawFetcher
from src.legal_food_safety_fetcher import FoodSafetyActFetcher


class LawType(Enum):
    """Supported law types"""
    FOOD_SAFETY = "food_safety"
    LABOR_LAW = "labor_law"
    CIVIL_LAW = "civil_law"


class DataManager:
    """
    Manages legal document data operations.

    Responsibilities:
    - Fetch legal documents from online sources
    - Save/load legal documents to/from JSON files
    - Provide unified interface for different law types
    """

    # Default data file paths
    DEFAULT_PATHS = {
        LawType.FOOD_SAFETY: Path("data/food_safety_act.json"),
        LawType.LABOR_LAW: Path("data/labor_standards_act.json"),
        LawType.CIVIL_LAW: Path("data/civil_code.json"),
    }

    # Fetcher classes for each law type
    FETCHERS = {
        LawType.FOOD_SAFETY: FoodSafetyActFetcher,
        LawType.LABOR_LAW: LaborLawFetcher,
        LawType.CIVIL_LAW: CivilLawFetcher,
    }

    def __init__(self, data_dir: Path = Path("data")):
        """
        Initialize data manager.

        Args:
            data_dir: Base directory for data files (default: "data/")
        """
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_data_path(self, law_type: LawType) -> Path:
        """Get the data file path for a specific law type"""
        return self.DEFAULT_PATHS[law_type]

    def data_exists(self, law_type: LawType) -> bool:
        """Check if data file exists for a law type"""
        return self.get_data_path(law_type).exists()

    def fetch_and_save(
        self,
        law_type: LawType,
        delay: float = 1.0,
        overwrite: bool = False
    ) -> bool:
        """
        Fetch legal documents and save to file.

        Args:
            law_type: Type of law to fetch
            delay: Delay between requests in seconds
            overwrite: Whether to overwrite existing file

        Returns:
            True if successful, False otherwise
        """
        data_path = self.get_data_path(law_type)

        # Check if file exists and should not overwrite
        if data_path.exists() and not overwrite:
            return True

        try:
            # Get appropriate fetcher class
            FetcherClass = self.FETCHERS[law_type]

            # Use context manager to ensure resource cleanup
            with FetcherClass(delay=delay) as fetcher:
                # Fetch all articles
                articles = fetcher.fetch_all_articles()

                if not articles:
                    return False

                # Save to JSON
                fetcher.save_to_json(str(data_path))

                return True

        except Exception as e:
            print(f"Error fetching {law_type.value}: {e}")
            return False

    def load_data(self, law_type: LawType) -> Optional[Dict[str, Any]]:
        """
        Load legal document data from file.

        Args:
            law_type: Type of law to load

        Returns:
            Dictionary containing law data, or None if file doesn't exist
        """
        import json

        data_path = self.get_data_path(law_type)

        if not data_path.exists():
            return None

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {law_type.value} data: {e}")
            return None

    def get_article_count(self, law_type: LawType) -> int:
        """
        Get the number of articles in a law.

        Args:
            law_type: Type of law

        Returns:
            Number of articles, or 0 if data not found
        """
        data = self.load_data(law_type)
        if not data:
            return 0

        return data.get('total_articles', len(data.get('articles', [])))

    def get_law_name(self, law_type: LawType) -> str:
        """
        Get the full name of a law.

        Args:
            law_type: Type of law

        Returns:
            Full law name in Chinese
        """
        names = {
            LawType.FOOD_SAFETY: "食品安全衛生管理法",
            LawType.LABOR_LAW: "勞動基準法",
            LawType.CIVIL_LAW: "民法",
        }
        return names[law_type]

    def get_all_existing_data(self) -> Dict[LawType, Dict[str, Any]]:
        """
        Load all existing law data files.

        Returns:
            Dictionary mapping LawType to law data
        """
        result = {}
        for law_type in LawType:
            data = self.load_data(law_type)
            if data:
                result[law_type] = data
        return result

    def check_all_data_ready(self) -> bool:
        """
        Check if data files exist for all law types.

        Returns:
            True if all data files exist
        """
        return all(self.data_exists(law_type) for law_type in LawType)