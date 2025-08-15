# TODOS:

## Manual Changes
- Update langgraph dependency to ">0.6"
- Introduce ruff https://github.com/astral-sh/ruff
- Maybe introduce test runner https://github.com/nat-n/poethepoet


## Gather TODO's
- Go through each file in a2a_servers and look for TODO comments and add them below in this TODO.md.
- Go through each file in mcp_servers and look for TODO comments and add them below in this TODO.md.
- Do NOT go ahead and implement anything yet.

## Automated Refactorings

### Refactor environment MCP variables
- Got into all ./src/mcp_servers/ files and replace the environment vars MCP_HOST & MCP_PORT to HOST and PORT
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

## Gathered TODO's
LIST OF TODOS