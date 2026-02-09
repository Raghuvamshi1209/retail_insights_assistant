from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def friendly_error(e: Exception):
    logger.exception("Unhandled error")
    return f"⚠️ {type(e).__name__}: {e}"
