"""EDL (edit decision list) save/load: JSON on disk, pydantic in memory."""

from __future__ import annotations

import logging
from pathlib import Path

from ytpgen.models import EDL

logger = logging.getLogger(__name__)


def save_edl(edl: EDL, path: Path) -> Path:
    """Serialize an EDL to JSON (pretty-printed) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(edl.model_dump_json(indent=2), encoding="utf-8")
    logger.info("saved EDL: %s (%d events)", path, len(edl.events))
    return path


def load_edl(path: Path) -> EDL:
    """Load an EDL from JSON. Raises pydantic ValidationError on bad shape."""
    path = Path(path)
    edl = EDL.model_validate_json(path.read_text(encoding="utf-8"))
    logger.info("loaded EDL: %s (%d events)", path, len(edl.events))
    return edl
