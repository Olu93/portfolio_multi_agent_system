-- +goose Up
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TYPE web_content.content_kind AS ENUM ('html','text','markdown','screenshot');

CREATE TABLE web_content.pages (
  id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  status       TEXT NOT NULL CHECK (status IN ('OK','ERROR','PENDING')),
  url          TEXT NOT NULL,
  title        TEXT,
  error        TEXT,
  collected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE web_content.page_contents (
  id            BIGSERIAL PRIMARY KEY,
  page_id       UUID NOT NULL REFERENCES web_content.pages(id) ON DELETE CASCADE,
  kind          web_content.content_kind NOT NULL,
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

CREATE INDEX IF NOT EXISTS idx_pages_url ON web_content.pages (url);
CREATE INDEX IF NOT EXISTS idx_page_contents_fts
  ON web_content.page_contents USING GIN (to_tsvector('simple', coalesce(content_text, '')));
CREATE INDEX IF NOT EXISTS idx_page_contents_page_kind
  ON web_content.page_contents (page_id, kind);

-- +goose Down
DROP INDEX IF EXISTS web_content.idx_page_contents_page_kind;
DROP INDEX IF EXISTS web_content.idx_page_contents_fts;
DROP INDEX IF EXISTS web_content.idx_pages_url;
DROP TABLE IF EXISTS web_content.page_contents;
DROP TABLE IF EXISTS web_content.pages;
DROP TYPE IF EXISTS web_content.content_kind;
