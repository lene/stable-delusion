"""
Token usage tracking service for image generation APIs.
Provides persistent storage and retrieval of API token usage statistics.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from stable_delusion.models.token_usage_entry import TokenUsageEntry
from stable_delusion.models.token_usage_stats import TokenUsageStats


class TokenUsageTracker:
    """Tracks and persists API token usage from various image generation services."""

    def __init__(self, storage_file: Optional[Path] = None):
        if storage_file is None:
            storage_file = self._get_default_storage_path()
        self.storage_file = storage_file
        self._ensure_storage_directory()

    def _get_default_storage_path(self) -> Path:
        home_dir = Path.home()
        storage_dir = home_dir / ".stable-delusion"
        return storage_dir / "token_usage.json"

    def _ensure_storage_directory(self) -> None:
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_usage_data(self) -> List[Dict[str, Any]]:
        if not self.storage_file.exists():
            return []
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning("Failed to load token usage data: %s", e)
            return []

    def _save_usage_data(self, data: List[Dict[str, Any]]) -> None:
        try:
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logging.error("Failed to save token usage data: %s", e)

    def _calculate_prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    def _extract_gemini_token_count(self, response: Any) -> Optional[int]:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            if hasattr(response.usage_metadata, "total_token_count"):
                return response.usage_metadata.total_token_count
        return None

    def _extract_seedream_token_count(self, response: Any) -> Optional[int]:
        if hasattr(response, "usage") and response.usage:
            if hasattr(response.usage, "total_tokens"):
                return response.usage.total_tokens
        return None

    def record_from_gemini_response(
        self, response: Any, prompt: str, operation: str = "generate"
    ) -> None:
        tokens = self._extract_gemini_token_count(response)
        if tokens is None:
            logging.warning("No token usage information found in Gemini response")
            return

        self._record_usage("gemini-2.5-flash", tokens, operation, prompt)

    def record_from_seedream_response(
        self, response: Any, prompt: str, operation: str = "generate"
    ) -> None:
        tokens = self._extract_seedream_token_count(response)
        if tokens is None:
            logging.warning("No token usage information found in Seedream response")
            return

        model = getattr(response, "model", "seedream-4-0")
        self._record_usage(model, tokens, operation, prompt)

    def _record_usage(
        self, model: str, tokens: int, operation: str, prompt: Optional[str] = None
    ) -> None:
        prompt_hash = self._calculate_prompt_hash(prompt) if prompt else None

        entry = TokenUsageEntry(
            timestamp=datetime.utcnow().isoformat(),
            model=model,
            tokens=tokens,
            operation=operation,
            prompt_hash=prompt_hash,
        )

        usage_data = self._load_usage_data()
        usage_data.append(entry.to_dict())
        self._save_usage_data(usage_data)

        logging.info("Recorded token usage: %d tokens for %s (%s)", tokens, model, operation)

    def get_usage_history(self, limit: Optional[int] = None) -> List[TokenUsageEntry]:
        usage_data = self._load_usage_data()
        if limit:
            usage_data = usage_data[-limit:]
        return [TokenUsageEntry(**entry) for entry in usage_data]

    def _aggregate_by_field(self, usage_data: List[Dict[str, Any]], field: str) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for entry in usage_data:
            key = entry.get(field, "unknown")
            result[key] = result.get(key, 0) + entry.get("tokens", 0)
        return result

    def _count_requests_by_field(
        self, usage_data: List[Dict[str, Any]], field: str
    ) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for entry in usage_data:
            key = entry.get(field, "unknown")
            result[key] = result.get(key, 0) + 1
        return result

    def get_statistics(self) -> TokenUsageStats:
        usage_data = self._load_usage_data()

        total_tokens = sum(entry.get("tokens", 0) for entry in usage_data)
        total_requests = len(usage_data)
        tokens_by_model = self._aggregate_by_field(usage_data, "model")
        requests_by_model = self._count_requests_by_field(usage_data, "model")
        tokens_by_operation = self._aggregate_by_field(usage_data, "operation")

        return TokenUsageStats(
            total_tokens=total_tokens,
            total_requests=total_requests,
            tokens_by_model=tokens_by_model,
            requests_by_model=requests_by_model,
            tokens_by_operation=tokens_by_operation,
        )

    def clear_history(self) -> None:
        self._save_usage_data([])
        logging.info("Cleared token usage history")
