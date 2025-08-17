npx @playwright/mcp@latest --port 8001 --isolated --headless --browser chrome 
npx @browsermcp/mcp@latest --port 8005 
npx -y fetcher-mcp --log --transport=http --host=0.0.0.0 --port=8005 
python -m lingua_scraper.mcp.file_storage
python -m lingua_scraper.mcp.duckduckgo
python -m lingua_scraper.mcp.playwright_extension