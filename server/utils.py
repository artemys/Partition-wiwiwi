import json
import logging
import os
import shutil
import subprocess
from typing import Iterable, List, Optional

from .config import SETTINGS


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def read_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_job_logger(job_id: str, logs_path: str) -> logging.Logger:
    logger = logging.getLogger(f"job.{job_id}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(logs_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def command_exists(path: str) -> bool:
    return shutil.which(path) is not None


def run_cmd(
    args: List[str],
    logger: logging.Logger,
    timeout_seconds: Optional[int] = None,
    cwd: Optional[str] = None,
) -> None:
    logger.info("Command: %s", " ".join(args))
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds or SETTINGS.subprocess_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("Timeout: %s", exc)
        raise RuntimeError(f"Timeout subprocess: {' '.join(args)}") from exc
    if result.stdout:
        logger.info("stdout: %s", result.stdout.strip())
    if result.stderr:
        logger.info("stderr: %s", result.stderr.strip())
    if result.returncode != 0:
        details = result.stderr.strip() if result.stderr else "aucun détail"
        raise RuntimeError(
            f"Commande échouée: {' '.join(args)} (code {result.returncode}). Détails: {details}"
        )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
