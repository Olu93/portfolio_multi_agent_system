-- +goose Up
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TYPE content_kind AS ENUM ('html','text','markdown','screenshot');

CREATE TABLE pages (
  id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  status       TEXT NOT NULL CHECK (status IN ('OK','ERROR','PENDING')),
  url          TEXT NOT NULL,
  title        TEXT,
  error        TEXT,
  collected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE page_contents (
  id            BIGSERIAL PRIMARY KEY,
  page_id       UUID NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  kind          content_kind NOT NULL,
  content_type  TEXT NOT NULL,
  content_text  TEXT,
  content_bytes BYTEA,
  content_ts    TIMESTAMPTZ,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT content_presence_chk CHECK (
    (kind IN ('html','text','markdown') AND content_text IS NOT NULL AND content_bytes IS NULL)
    OR (kind = 'screenshot' AND content_bytes IS NOT NULL AND content_text IS NULL)
  ),
  CONSTRAINT content_type_kind_consistent CHECK (
    (kind = 'html' AND content_type ILIKE 'html%')
    OR (kind = 'text' AND content_type ILIKE 'text%')
    OR (kind = 'markdown' AND content_type ILIKE 'markdown%')
    OR (kind = 'screenshot' AND content_type ILIKE 'screenshot%')
  )
);

CREATE INDEX IF NOT EXISTS idx_pages_url ON pages (url);
CREATE INDEX IF NOT EXISTS idx_page_contents_fts
  ON page_contents USING GIN (to_tsvector('simple', coalesce(content_text, '')));
CREATE INDEX IF NOT EXISTS idx_page_contents_page_kind ON page_contents (page_id, kind);

-- +goose Down
DROP INDEX IF EXISTS idx_page_contents_page_kind;
DROP INDEX IF EXISTS idx_page_contents_fts;
DROP INDEX IF EXISTS idx_pages_url;
DROP TABLE IF EXISTS page_contents;
DROP TABLE IF EXISTS pages;
DROP TYPE IF EXISTS content_kind;
