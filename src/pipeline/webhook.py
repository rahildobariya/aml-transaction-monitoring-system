import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 5  # seconds


def route_alerts(
    df_tiered: pd.DataFrame,
    webhook_url: str,
    *,
    tiers: tuple = ("CRITICAL", "HIGH"),
    timeout: int = _TIMEOUT,
) -> int:
    """POST CRITICAL/HIGH alerts to the n8n webhook. Returns the number of alerts sent, or 0 on failure."""
    actionable = df_tiered[df_tiered["tier"].isin(tiers)]
    if actionable.empty:
        logger.info("[webhook] No actionable alerts to route.")
        return 0

    cols = [
        c for c in
        ["rank", "tier", "fraud_score", "amount", "tx_type",
         "structuring_flag", "is_new_beneficiary", "is_fraud"]
        if c in actionable.columns
    ]
    payload = {
        "source": "aml-monitoring",
        "alert_count": len(actionable),
        "alerts": actionable[cols].to_dict(orient="records"),
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        logger.info(
            "[webhook] Routed %d alerts → n8n (HTTP %d).",
            len(actionable), resp.status_code,
        )
        return len(actionable)
    except requests.exceptions.ConnectionError:
        logger.warning("[webhook] n8n not reachable at %s — skipping alert routing.", webhook_url)
    except requests.exceptions.Timeout:
        logger.warning("[webhook] n8n webhook timed out after %ds — skipping.", timeout)
    except requests.exceptions.HTTPError as exc:
        logger.warning("[webhook] n8n returned error: %s — skipping.", exc)
    return 0
