from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Gemini
    google_api_key: str = ""

    # Google Drive
    google_drive_drawings_folder_id: str = ""
    google_service_account_json: str = ""  # file path or inline JSON string

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.google_api_key)

    @property
    def drive_enabled(self) -> bool:
        return bool(self.google_drive_drawings_folder_id and self.google_service_account_json)


@lru_cache
def get_settings() -> Settings:
    return Settings()
