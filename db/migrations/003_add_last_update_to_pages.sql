-- +goose Up
ALTER TABLE web_content.pages ADD COLUMN last_update TIMESTAMPTZ DEFAULT now();

-- +goose Down
ALTER TABLE web_content.pages DROP COLUMN IF EXISTS last_update;