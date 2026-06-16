"""
config.py — 集中管理 Settings。

設定優先順序：
  1. 環境變數
  2. .env（本地覆蓋，不進 git）
  3. config.env（預設值範本）
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("config.env", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OCR 工具選擇 ──────────────────────────────────────────────────────
    # 預設 extractor：azure_cu | docling
    default_extractor: str = "azure_cu"

    # ── Azure Content Understanding API ───────────────────────────────────
    azure_cu_endpoint: str = "https://aif-futago-dev-eus2-01.services.ai.azure.com/"
    azure_cu_api_key: str = "unused"  # 不填則用 DefaultAzureCredential

    # ── Azure Document Intelligence API（照片 path）────────────────────────
    # 若與 CU 使用同一資源，endpoint 相同；但 SDK 不同（azure-ai-formrecognizer / azure-ai-documentintelligence）
    azure_di_endpoint: str = "https://aif-futago-dev-eus2-01.services.ai.azure.com/"
    azure_di_api_key: str = "unused"  # 不填則用 DefaultAzureCredential

    # ── LLM Extractor（Beta）— 支援三種模型 ──────────────────────────────
    # gemma3（預設）
    gemma3_endpoint:   str = "http://172.31.6.3:8080/gemma3/v1"
    gemma3_model_name: str = "/model"
    gemma3_api_key:    str = "unused"
    # gemma4
    gemma4_endpoint:   str = "http://172.31.6.3:8080/gemma4/v1"
    gemma4_model_name: str = "/model"
    gemma4_api_key:    str = "unused"
    # GPT-4.1（Azure OpenAI）
    gpt41_endpoint:    str = "https://aif-futago-dev-eus2-01.services.ai.azure.com/openai/v1"
    gpt41_deployment:  str = "gpt-4.1"
    gpt41_api_key:     str = "unused"

    # ── 圖片輸出 ──────────────────────────────────────────────────────────
    # 轉換結果的圖片預設存放目錄（相對於 server.py 所在目錄）
    # 可透過 API ?output_dir= 參數覆蓋
    images_output_dir: str = "output/images"

    # ── 服務設定 ──────────────────────────────────────────────────────────
    api_port: int = 8765
    api_host: str = "0.0.0.0"


# 全域單例
settings = Settings()
