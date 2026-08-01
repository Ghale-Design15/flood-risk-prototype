"""
Tamper-evident alert audit log (hash chain) + SendGrid dispatch.
======================================================================
The dashboard's alert-authorisation card POSTs an authorised flood alert to
``/alerts``. This module:

  1. Appends the alert to a Supabase table as a link in a hash chain. Each row
     stores ``prev_hash`` (the previous row's hash) and ``row_hash`` (the SHA-256
     of this row's canonical fields including ``prev_hash``). Any later edit to a
     row changes its hash and breaks every link after it, so tampering is
     detectable via ``verify_chain``. The table is also append-only at the DB
     level (see db/supabase_audit.sql), so rows cannot be silently rewritten.
  2. Sends an email via SendGrid when configured, and degrades gracefully to a
     "skipped" status when no API key/recipients are set, so the demo still works.

Env vars
    SUPABASE_URL, SUPABASE_SERVICE_KEY   required for the audit store
    SENDGRID_API_KEY                     optional; without it, email is skipped
    ALERT_EMAIL_FROM, ALERT_EMAIL_TO     sender + comma-separated recipients
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

TABLE = "flood_alert_audit"
GENESIS_HASH = "0" * 64

# Fields that make up a chained record, in a fixed order. row_hash is computed
# over exactly these (prev_hash included), so verification is deterministic.
CHAIN_FIELDS = [
    "alert_id", "created_at", "station_id", "station_name", "risk_band",
    "flood_probability", "horizon", "operator", "message", "recipients", "prev_hash",
]


# ---- Pure hashing helpers (no I/O — unit-testable without Supabase) --------
def _normalize(record: Dict) -> Dict:
    """Stable representation for hashing. Floats are formatted to a fixed
    precision so a Postgres round-trip can't drift the hash."""
    out = {}
    for k in CHAIN_FIELDS:
        v = record.get(k)
        if k == "flood_probability" and v is not None:
            v = f"{float(v):.4f}"
        out[k] = "" if v is None else str(v)
    return out


def canonical(record: Dict) -> str:
    return json.dumps(_normalize(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_hash(record: Dict) -> str:
    return hashlib.sha256(canonical(record).encode("utf-8")).hexdigest()


def _recipients_str(payload: Dict) -> str:
    """Normalise recipients (list or string) to a stable comma-separated string."""
    r = payload.get("recipients")
    if isinstance(r, (list, tuple)):
        return ", ".join(str(x).strip() for x in r if str(x).strip())
    return (r or "").strip()


def build_record(payload: Dict, prev_hash: str, *, alert_id: str, created_at: str) -> Dict:
    """Assemble a full chain row (business fields + prev_hash + row_hash)."""
    record = {
        "alert_id": alert_id,
        "created_at": created_at,
        "station_id": payload.get("station_id"),
        "station_name": payload.get("station_name"),
        "risk_band": payload.get("risk_band"),
        "flood_probability": payload.get("flood_probability"),
        "horizon": payload.get("horizon"),
        "operator": payload.get("operator"),
        "message": payload.get("message"),
        "recipients": _recipients_str(payload),
        "prev_hash": prev_hash,
    }
    record["row_hash"] = compute_hash(record)
    return record


def verify_chain(rows: List[Dict]) -> Dict:
    """Recompute every row's hash and check the prev_hash linkage.
    ``rows`` must be ordered oldest-first. Returns integrity summary."""
    expected_prev = GENESIS_HASH
    for i, row in enumerate(rows):
        if row.get("prev_hash") != expected_prev:
            return {"ok": False, "checked": len(rows), "broken_at": i,
                    "reason": "prev_hash does not match the previous row's hash"}
        if compute_hash(row) != row.get("row_hash"):
            return {"ok": False, "checked": len(rows), "broken_at": i,
                    "reason": "row_hash does not match recomputed hash (row altered)"}
        expected_prev = row["row_hash"]
    return {"ok": True, "checked": len(rows), "broken_at": None}


# ---- Supabase store -------------------------------------------------------
_client = None
_client_ready = False


def _supabase():
    """Lazily create the Supabase client, or None if not configured."""
    global _client, _client_ready
    if _client_ready:
        return _client
    _client_ready = True
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
    if not (url and key):
        _client = None
        return None
    from supabase import create_client  # lazy import so the API runs without it
    _client = create_client(url, key)
    return _client


def store_configured() -> bool:
    """True if the Supabase env vars are set. Only checks env — never imports the
    client or raises — so callers can gate on it without risking a 500."""
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"))


def _last_hash(client) -> str:
    res = client.table(TABLE).select("row_hash").order("id", desc=True).limit(1).execute()
    data = res.data or []
    return data[0]["row_hash"] if data else GENESIS_HASH


def _new_alert_id() -> str:
    return "A-" + uuid.uuid4().hex[:8].upper()


def record_alert(payload: Dict) -> Dict:
    """Append an alert to the chain and (optionally) email it.
    Raises RuntimeError if the audit store is not configured."""
    client = _supabase()
    if client is None:
        raise RuntimeError("Audit store not configured (set SUPABASE_URL and SUPABASE_SERVICE_KEY).")

    prev_hash = _last_hash(client)
    created_at = datetime.now(timezone.utc).isoformat()
    record = build_record(payload, prev_hash, alert_id=_new_alert_id(), created_at=created_at)
    client.table(TABLE).insert(record).execute()

    email_status = _send_email(record)
    return {
        "alert_id": record["alert_id"],
        "created_at": record["created_at"],
        "recipients": record["recipients"],
        "prev_hash": prev_hash,
        "row_hash": record["row_hash"],
        "chained": True,
        "email_status": email_status,
    }


def fetch_chain(limit: int = 1000) -> List[Dict]:
    client = _supabase()
    if client is None:
        raise RuntimeError("Audit store not configured (set SUPABASE_URL and SUPABASE_SERVICE_KEY).")
    res = client.table(TABLE).select("*").order("id", desc=False).limit(limit).execute()
    return res.data or []


# ---- Email (graceful) -----------------------------------------------------
def _send_email(record: Dict) -> str:
    key = os.getenv("SENDGRID_API_KEY")
    frm = os.getenv("ALERT_EMAIL_FROM")
    # Prefer the recipients chosen in the dashboard; fall back to ALERT_EMAIL_TO.
    to = record.get("recipients") or os.getenv("ALERT_EMAIL_TO")
    if not (key and to and frm):
        return "skipped (need SENDGRID_API_KEY, ALERT_EMAIL_FROM, and recipients or ALERT_EMAIL_TO)"
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        subject = f"[Flood Alert] {record['risk_band']} risk — {record.get('station_name') or record.get('station_id')}"
        body = (
            f"{record.get('message', '')}\n\n"
            f"Station: {record.get('station_name')} ({record.get('station_id')})\n"
            f"Risk band: {record['risk_band']}\n"
            f"Probability: {record.get('flood_probability')}\n"
            f"Horizon: {record.get('horizon')}\n"
            f"Authorised by: {record.get('operator')}\n"
            f"Alert id: {record['alert_id']}  (audit hash {record['row_hash'][:12]}…)\n"
            f"Time (UTC): {record['created_at']}\n"
        )
        msg = Mail(from_email=frm, to_emails=[t.strip() for t in to.split(",") if t.strip()],
                   subject=subject, plain_text_content=body)
        resp = SendGridAPIClient(key).send(msg)
        return f"sent ({resp.status_code})"
    except Exception as exc:  # never fail the alert because email failed
        return f"error: {exc}"
