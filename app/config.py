from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Segmentation API"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    project_root: Path = Path(__file__).resolve().parent.parent
    weights_dir: Path = project_root / "weights"
    outputs_dir: Path = project_root / "outputs"
    votes_file: Path = outputs_dir / "votes.json"

    base_model_path: Path = weights_dir / "base.pth"
    felix_model_path: Path = weights_dir / "felix.pth"

    allowed_origins: list[str] = [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://lightbabanks.github.io",
    ]

    input_size: int = 256
    threshold: float = 0.5
    device: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
settings.outputs_dir.mkdir(parents=True, exist_ok=True)
