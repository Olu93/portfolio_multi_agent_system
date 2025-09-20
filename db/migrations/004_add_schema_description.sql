-- +goose Up
COMMENT ON SCHEMA web_content IS 'This schema holds all web content tables';
COMMENT ON TABLE web_content.pages IS 'This table holds all web pages scraped';
COMMENT ON TABLE web_content.page_contents IS 'This table holds all web page contents scraped';


-- +goose Down
COMMENT ON SCHEMA web_content IS NULL;
COMMENT ON TABLE web_content.pages IS NULL;
COMMENT ON TABLE web_content.page_contents IS NULL;