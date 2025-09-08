# TODOS:

## Manual Changes
- Update langgraph dependency to ">0.6"
- Introduce ruff https://github.com/astral-sh/ruff
- Maybe introduce test runner https://github.com/nat-n/poethepoet
- Introduce pre-prepared MCP servers from python-a2a 
    - Initial discovery: https://python-a2a.readthedocs.io/en/latest/guides/mcp.html
    - Available servers: https://github.com/themanojdesai/python-a2a?tab=readme-ov-file#%EF%B8%8F-provider-architecture-overview
    - How to run: https://github.com/themanojdesai/python-a2a/wiki/MCP-Integration#2-fastmcp
- Introduce general agent
- Switch to python-a2a
- Introduce configuration validation

## Next up
[] User Interface
[] Add logging to mcp servers 
[x] Change to python SDK agents
[x] Fix double sends
[] Put registry background tasks into its seperate module
[] Make sure the supervisor continuously updates its skills
[] Make sure supervisor corrects missing info itself
[] Make sure conversation remains intact throughout context


## Gather TODO's
- Go through each file in a2a_servers and look for TODO comments and add them below in this TODO.md.
- Go through each file in mcp_servers and look for TODO comments and add them below in this TODO.md.
- Do NOT go ahead and implement anything yet.

## Automated Refactorings

### Refactor environment MCP variables
- Got into all ./backend/src/mcp_servers/ files and replace the environment vars MCP_HOST & MCP_PORT to HOST and PORT
- Apply the change to ./docker-compose.mcp.yml accordingly
- Apply the change to ./.vscode/launch.json accordingly

### Refactor Docker
- Put all dockerfiles into ./docker folder, but keep the docker-compose files where they are
- Update the references to docker files in the docker
 
### Add log messages
- Add a top level src/mcp_servers/log.py configuration which does src\a2a_servers\__init__.py
- In every code file add logger = get_logger(__file__)
- Make sure that all the mpc_servers import LOG_CONFIGURATION_COMPLETED from log.py and log.info that. This is to ensure that the log.py was definetely imported.
- Add reasonable log.info, log.warning and log.error statements in each mcp_servers file.   

### Multiline strings
- Go through the code and check where there are multiline strings which mess up the indentation.
- Replace these strings using "String1"\n"String2"\n"String3" pattern

### Add information to readme.md
- How to run in debug mode
    - Set env var IS_PROD=false
    - 

## Weird Ideas Backlog (LLM: Don't do anything with that)
### Create a ui interface to manage multiple debugpy servers
- Would listen to multiple services
- Features:
    - Toggle break-points in files 
    - Display stack-trace 
    - Display local variables
### Create lingua showcase
- Make lingua an MCP server
- Rename lingua to portfolio_smart_data_extraction

## Gathered TODO's
LIST OF TODOS

