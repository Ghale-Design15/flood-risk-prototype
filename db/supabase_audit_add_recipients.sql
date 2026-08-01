-- Migration: add the recipients column to the alert audit log.
-- Run this once in the Supabase SQL editor after the initial supabase_audit.sql.
--
-- recipients is now part of the hash chain (see backend/alerts.py CHAIN_FIELDS),
-- so rows created before this change will no longer verify. The audit table so
-- far only holds mock test rows, so we clear it once and let the chain restart
-- clean. TRUNCATE bypasses the append-only row trigger; going forward, inserts
-- are still the only allowed operation.

alter table flood_alert_audit add column if not exists recipients text;

truncate table flood_alert_audit restart identity;
