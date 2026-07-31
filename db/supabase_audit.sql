-- Flood alert audit log — tamper-evident hash chain.
-- Run this once in the Supabase SQL editor (or psql) to create the table the
-- backend /alerts endpoint writes to. See backend/alerts.py for the chain logic.
--
-- Each row stores prev_hash (the previous row's row_hash) and its own row_hash
-- (SHA-256 of the row's canonical fields, prev_hash included). Editing any row
-- changes its hash and breaks every link after it. To make that guarantee real,
-- the table is append-only: updates and deletes are blocked at the DB level.

create table if not exists flood_alert_audit (
    id                bigint generated always as identity primary key,
    alert_id          text        not null unique,
    created_at        timestamptz not null default now(),
    station_id        text,
    station_name      text,
    risk_band         text        not null,
    flood_probability double precision,
    horizon           text,
    operator          text        not null,
    message           text,
    prev_hash         text        not null,
    row_hash          text        not null unique
);

create index if not exists idx_faa_created_at on flood_alert_audit (created_at);

-- ---- Append-only enforcement --------------------------------------------
-- Block UPDATE and DELETE so the chain cannot be silently rewritten. Inserts
-- (new links) are still allowed.
create or replace function faa_block_mutations()
returns trigger
language plpgsql
as $$
begin
    raise exception 'flood_alert_audit is append-only; % is not permitted', tg_op;
end;
$$;

drop trigger if exists trg_faa_no_update on flood_alert_audit;
create trigger trg_faa_no_update
    before update or delete on flood_alert_audit
    for each row execute function faa_block_mutations();

-- Row Level Security: the backend uses the service-role key (bypasses RLS) to
-- insert. Enable RLS so nothing else can read/write without an explicit policy.
alter table flood_alert_audit enable row level security;
