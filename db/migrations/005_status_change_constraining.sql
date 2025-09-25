-- +goose Up
ALTER TABLE web_content.pages
  ALTER COLUMN status TYPE text,
  DROP CONSTRAINT IF EXISTS pages_status_check,
  ADD CONSTRAINT pages_status_check
    CHECK (status IN ('OK','ERROR','PENDING'));

-- +goose Down
ALTER TABLE web_content.pages
  DROP CONSTRAINT IF EXISTS pages_status_check,
  ADD CHECK (status IN ('OK','ERROR','PENDING'));
