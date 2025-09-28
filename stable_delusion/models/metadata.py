"""
Data models for metadata storage and management.
Provides structured formats for storing generation metadata and supporting deduplication.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass
class GenerationMetadata:
    """Metadata for image generation operations."""

    prompt: str
    images: List[str]  # S3 URLs or local paths
    generated_image: str  # S3 URL or local path
    gcp_project_id: Optional[str] = None
    gcp_location: Optional[str] = None
    scale: Optional[int] = None
    model: Optional[str] = None
    timestamp: Optional[str] = None
    content_hash: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.content_hash is None:
            self.content_hash = self._compute_content_hash()

    def _compute_content_hash(self) -> str:
        # Create deterministic string from inputs
        hash_input = {
            "prompt": self.prompt,
            "images": sorted(self.images),  # Sort for consistency
            "gcp_project_id": self.gcp_project_id,
            "gcp_location": self.gcp_location,
            "scale": self.scale,
            "model": self.model,
        }

        # Convert to JSON string with sorted keys for consistency
        json_str = json.dumps(hash_input, sort_keys=True, separators=(",", ":"))

        # Compute SHA256 hash
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationMetadata":
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "GenerationMetadata":
        return cls.from_dict(json.loads(json_str))

    def get_metadata_filename(self) -> str:
        # Extract date from timestamp for readability
        if self.timestamp:
            try:
                dt = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y%m%d_%H%M%S")
            except ValueError:
                date_str = "unknown"
        else:
            date_str = "unknown"

        # Use first 8 characters of hash for brevity
        hash_prefix = self.content_hash[:8] if self.content_hash else "nohash"

        return f"metadata_{hash_prefix}_{date_str}.json"
