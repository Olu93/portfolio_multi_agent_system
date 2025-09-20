-- +goose Up
CREATE SCHEMA IF NOT EXISTS web_content AUTHORIZATION app;

-- +goose Down
DROP SCHEMA IF EXISTS web_content CASCADE;
