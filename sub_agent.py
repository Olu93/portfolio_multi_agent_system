import logging
import click
from dotenv import load_dotenv, find_dotenv



load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=5001)
def main(host, port):
    """Starts the Research Agent server."""
    run_agent_server(
        host=host, port=port, log_level="info"
    )