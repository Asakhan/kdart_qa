"""Shared utilities: config loading, logging, paths."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return config.yaml as a plain dict."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    """Resolve a config path (relative to project root) to an absolute Path."""
    return PROJECT_ROOT / relative


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Get a configured logger. Writes to both stdout and a file under logs/."""
    logger = logging.getLogger(name)
    if logger.handlers:  # idempotent
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    logs_dir = ensure_dir(PROJECT_ROOT / "logs")
    fname = log_file or f"{name}_{datetime.now():%Y%m%d}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / fname, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def require_env(var: str) -> str:
    """Read a required env var or raise a clear error (no silent fail)."""
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(
            f"환경변수 {var}가 설정되지 않았습니다. "
            f"export {var}='...' 후 재실행하세요."
        )
    return val


@dataclass(frozen=True)
class Company:
    name: str
    corp_code: str
    stock: str
    sector: str

    @classmethod
    def from_dict(cls, d: dict) -> "Company":
        return cls(
            name=d["name"],
            corp_code=str(d["corp_code"]).zfill(8),
            stock=str(d["stock"]).zfill(6),
            sector=d["sector"],
        )


def companies_from_config(cfg: dict[str, Any]) -> list[Company]:
    return [Company.from_dict(d) for d in cfg["dart"]["companies"]]
