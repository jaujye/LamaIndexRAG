"""
Configuration Validator
驗證系統配置，確保在啟動時就發現問題，遵循 Fail-Fast 原則
"""

import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class ValidationResult:
    """驗證結果"""
    passed: bool
    errors: List[str]
    warnings: List[str]
    config_summary: Dict[str, str]


class ConfigValidator:
    """系統配置驗證器"""

    # OpenAI 有效的模型列表
    VALID_LLM_MODELS = {
        # GPT-4 系列
        'gpt-4', 'gpt-4-32k', 'gpt-4-turbo', 'gpt-4-turbo-preview',
        'gpt-4-0125-preview', 'gpt-4-1106-preview',
        'gpt-4-vision-preview', 'gpt-4-1106-vision-preview',
        'gpt-4-turbo-2024-04-09',
        'gpt-4-0613', 'gpt-4-32k-0613', 'gpt-4-0314', 'gpt-4-32k-0314',

        # GPT-4o 系列
        'gpt-4o', 'gpt-4o-mini',
        'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06',
        'gpt-4o-mini-2024-07-18',

        # GPT-3.5 系列
        'gpt-3.5-turbo', 'gpt-3.5-turbo-16k',
        'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106',
        'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613',
        'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-instruct',

        # Azure 版本
        'gpt-35-turbo', 'gpt-35-turbo-16k',
        'gpt-35-turbo-0125', 'gpt-35-turbo-1106',
        'gpt-35-turbo-0613', 'gpt-35-turbo-16k-0613',

        # Legacy 模型
        'text-davinci-003', 'text-davinci-002',
        'text-ada-001', 'text-babbage-001', 'text-curie-001',
        'ada', 'babbage', 'curie', 'davinci'
    }

    VALID_EMBEDDING_MODELS = {
        'text-embedding-3-small',
        'text-embedding-3-large',
        'text-embedding-ada-002'
    }

    REQUIRED_ENV_VARS = [
        'OPENAI_API_KEY'
    ]

    OPTIONAL_ENV_VARS = {
        'OPENAI_LLM_MODEL': 'gpt-4o-mini',
        'OPENAI_EMBEDDING_MODEL': 'text-embedding-3-small',
        'OPENAI_TEMPERATURE': '0.1',
        'CHROMA_DB_PATH': './chroma_db',
        'WANDB_MODE': 'disabled'
    }

    def __init__(self):
        """初始化驗證器"""
        load_dotenv()

    def validate_all(self) -> ValidationResult:
        """
        執行完整的配置驗證

        Returns:
            ValidationResult 包含所有驗證結果
        """
        errors = []
        warnings = []
        config_summary = {}

        # 1. 驗證必需的環境變數
        env_errors, env_warnings, env_config = self._validate_env_vars()
        errors.extend(env_errors)
        warnings.extend(env_warnings)
        config_summary.update(env_config)

        # 2. 驗證 OpenAI 配置
        openai_errors, openai_warnings, openai_config = self._validate_openai_config()
        errors.extend(openai_errors)
        warnings.extend(openai_warnings)
        config_summary.update(openai_config)

        # 3. 驗證模型名稱
        model_errors, model_warnings, model_config = self._validate_models()
        errors.extend(model_errors)
        warnings.extend(model_warnings)
        config_summary.update(model_config)

        passed = len(errors) == 0

        return ValidationResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            config_summary=config_summary
        )

    def _validate_env_vars(self) -> Tuple[List[str], List[str], Dict[str, str]]:
        """驗證環境變數"""
        errors = []
        warnings = []
        config = {}

        # 檢查必需的環境變數
        for var in self.REQUIRED_ENV_VARS:
            value = os.getenv(var)
            if not value:
                errors.append(f"❌ 缺少必需的環境變數: {var}")
            else:
                # 只顯示 API key 的前幾個字符
                if 'KEY' in var or 'TOKEN' in var:
                    display_value = value[:8] + "..." if len(value) > 8 else "***"
                else:
                    display_value = value
                config[var] = display_value

        # 檢查可選的環境變數並使用默認值
        for var, default in self.OPTIONAL_ENV_VARS.items():
            value = os.getenv(var, default)
            config[var] = value

            if not os.getenv(var):
                warnings.append(f"⚠️  未設置 {var}，使用默認值: {default}")

        return errors, warnings, config

    def _validate_openai_config(self) -> Tuple[List[str], List[str], Dict[str, str]]:
        """驗證 OpenAI 配置"""
        errors = []
        warnings = []
        config = {}

        api_key = os.getenv('OPENAI_API_KEY')

        if api_key:
            # 驗證 API key 格式
            if not api_key.startswith('sk-'):
                errors.append("❌ OPENAI_API_KEY 格式不正確（應該以 'sk-' 開頭）")

            if len(api_key) < 20:
                errors.append("❌ OPENAI_API_KEY 長度不足（可能不是有效的 API key）")

        # 驗證溫度參數
        temperature_str = os.getenv('OPENAI_TEMPERATURE', '0.1')
        try:
            temperature = float(temperature_str)
            if not (0.0 <= temperature <= 2.0):
                warnings.append(f"⚠️  OPENAI_TEMPERATURE={temperature} 超出建議範圍 [0.0, 2.0]")
            config['temperature'] = str(temperature)
        except ValueError:
            errors.append(f"❌ OPENAI_TEMPERATURE 不是有效的數字: {temperature_str}")

        return errors, warnings, config

    def _validate_models(self) -> Tuple[List[str], List[str], Dict[str, str]]:
        """驗證模型名稱"""
        errors = []
        warnings = []
        config = {}

        # 驗證 LLM 模型
        llm_model = os.getenv('OPENAI_LLM_MODEL', 'gpt-4o-mini')
        config['llm_model'] = llm_model

        if llm_model not in self.VALID_LLM_MODELS:
            errors.append(
                f"❌ 無效的 LLM 模型: '{llm_model}'\n"
                f"   建議使用: gpt-4o-mini (經濟), gpt-3.5-turbo (快速), gpt-4o (強大)"
            )

            # 嘗試提供建議
            suggestions = self._suggest_similar_models(llm_model, self.VALID_LLM_MODELS)
            if suggestions:
                errors.append(f"   您是否想使用: {', '.join(suggestions[:3])}")

        # 驗證嵌入模型
        embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        config['embedding_model'] = embedding_model

        if embedding_model not in self.VALID_EMBEDDING_MODELS:
            errors.append(
                f"❌ 無效的嵌入模型: '{embedding_model}'\n"
                f"   有效選項: {', '.join(self.VALID_EMBEDDING_MODELS)}"
            )

        return errors, warnings, config

    def _suggest_similar_models(self, invalid_model: str, valid_models: set) -> List[str]:
        """
        提供相似的模型建議

        Args:
            invalid_model: 無效的模型名稱
            valid_models: 有效模型集合

        Returns:
            相似模型的列表
        """
        invalid_lower = invalid_model.lower()
        suggestions = []

        # 簡單的字符串匹配
        for model in valid_models:
            model_lower = model.lower()

            # 檢查是否包含相似的部分
            if any(part in model_lower for part in invalid_lower.split('-')):
                suggestions.append(model)

        # 按相似度排序（簡單的長度相似度）
        suggestions.sort(key=lambda m: abs(len(m) - len(invalid_model)))

        return suggestions[:5]

    def get_config_recommendations(self) -> Dict[str, str]:
        """
        獲取配置建議

        Returns:
            包含推薦配置的字典
        """
        return {
            "經濟型配置": "OPENAI_LLM_MODEL=gpt-4o-mini",
            "平衡型配置": "OPENAI_LLM_MODEL=gpt-3.5-turbo",
            "性能型配置": "OPENAI_LLM_MODEL=gpt-4o",
            "嵌入模型": "OPENAI_EMBEDDING_MODEL=text-embedding-3-small",
            "溫度設置": "OPENAI_TEMPERATURE=0.1 (更確定) 或 0.7 (更創意)"
        }

    @staticmethod
    def validate_api_key_live(api_key: str) -> Tuple[bool, str]:
        """
        驗證 API key 是否有效（可選的實時驗證）

        Args:
            api_key: OpenAI API key

        Returns:
            (是否有效, 錯誤訊息或成功訊息)
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            # 嘗試一個輕量級的API調用
            models = client.models.list()
            return True, "✅ API key 驗證成功"

        except Exception as e:
            return False, f"❌ API key 驗證失敗: {str(e)}"


def validate_config() -> ValidationResult:
    """
    便捷函數：驗證配置

    Returns:
        ValidationResult
    """
    validator = ConfigValidator()
    return validator.validate_all()
