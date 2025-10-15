"""
Environment Validator
Validates system environment including .env files, API keys, and data files
"""

import os
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

from src.config_validator import ConfigValidator


@dataclass
class ValidationResult:
    """Result of environment validation"""
    passed: bool
    issues: List[str]
    warnings: Optional[List[str]] = None
    config_summary: Optional[Dict[str, str]] = None

    @property
    def issues_count(self) -> int:
        return len(self.issues)

    @property
    def warnings_count(self) -> int:
        return len(self.warnings) if self.warnings else 0


class EnvironmentValidator:
    """
    Validates the environment setup for the Legal RAG system.

    Responsibilities:
    - Check .env file existence
    - Validate required API keys
    - Verify data file availability
    """

    def __init__(
        self,
        env_file: Path = Path(".env"),
        required_keys: Optional[List[str]] = None
    ):
        """
        Initialize environment validator.

        Args:
            env_file: Path to .env file (default: ".env")
            required_keys: List of required environment variables (default: ["OPENAI_API_KEY"])
        """
        self.env_file = env_file
        self.required_keys = required_keys or ["OPENAI_API_KEY"]

    def validate(self, data_files: Optional[List[Path]] = None, validate_models: bool = True) -> ValidationResult:
        """
        Perform complete environment validation.

        Args:
            data_files: Optional list of data files to check
            validate_models: Whether to validate LLM model configuration (default: True)

        Returns:
            ValidationResult with passed status and any issues found
        """
        issues = []
        warnings = []
        config_summary = {}

        # Check .env file
        env_issues = self._check_env_file()
        issues.extend(env_issues)

        # Validate OpenAI configuration and models
        if validate_models:
            config_validator = ConfigValidator()
            config_result = config_validator.validate_all()

            issues.extend(config_result.errors)
            warnings.extend(config_result.warnings)
            config_summary.update(config_result.config_summary)

        # Check data files if provided
        if data_files:
            data_issues = self._check_data_files(data_files)
            issues.extend(data_issues)

        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            warnings=warnings if warnings else None,
            config_summary=config_summary if config_summary else None
        )

    def _check_env_file(self) -> List[str]:
        """Check .env file and required API keys"""
        issues = []

        if not self.env_file.exists():
            issues.append(
                f"[FAIL] 未找到 {self.env_file} 檔案。"
                "請複製 .env.template 並設定您的 API keys"
            )
            return issues

        # Load and check environment variables
        from dotenv import load_dotenv
        load_dotenv(self.env_file)

        for key in self.required_keys:
            value = os.getenv(key)
            if not value:
                issues.append(f"[FAIL] .env 檔案中未設定 {key}")
            elif len(value.strip()) == 0:
                issues.append(f"[FAIL] {key} 的值為空")

        return issues

    def _check_data_files(self, data_files: List[Path]) -> List[str]:
        """Check if required data files exist"""
        issues = []

        # Check if at least one data file exists
        existing_files = [f for f in data_files if f.exists()]

        if not existing_files:
            file_names = ", ".join(f.name for f in data_files)
            issues.append(
                f"[FAIL] 未找到任何法規資料檔案 ({file_names})。"
                "需要先下載法規內容"
            )

        return issues

    def check_single_api_key(self, key_name: str) -> bool:
        """
        Check if a specific API key is set.

        Args:
            key_name: Name of the environment variable

        Returns:
            True if key exists and is not empty
        """
        value = os.getenv(key_name)
        return bool(value and value.strip())

    def get_api_key(self, key_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get an API key value.

        Args:
            key_name: Name of the environment variable
            default: Default value if key not found

        Returns:
            API key value or default
        """
        return os.getenv(key_name, default)