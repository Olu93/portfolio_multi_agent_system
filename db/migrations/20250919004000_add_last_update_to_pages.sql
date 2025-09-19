-- +goose Up
ALTER TABLE pages ADD COLUMN last_update TIMESTAMPTZ DEFAULT now();

-- +goose Down
ALTER TABLE pages DROP COLUMN IF EXISTS last_update;
