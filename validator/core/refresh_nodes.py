"""
Gets the latest nodes from the network and stores them in the database,
migrating the old nodes to history in the process
"""

import asyncio
from datetime import datetime
from datetime import timedelta

from fiber.chain import fetch_nodes
from fiber.chain.models import Node
from fiber.logging_utils import get_logger

from validator.core.config import Config
from validator.db.sql.nodes import get_all_nodes
from validator.db.sql.nodes import get_last_updated_time_for_nodes
from validator.db.sql.nodes import insert_nodes
from validator.db.sql.nodes import migrate_nodes_to_history


logger = get_logger(__name__)


async def _fetch_nodes_from_substrate(config: Config) -> list[Node]:
    # It won't cause issues with substrate, as it creates a new connection.
    return await asyncio.to_thread(fetch_nodes.get_nodes_for_netuid, config.substrate, config.netuid)


async def _is_recent_update(config: Config) -> bool:
    async with await config.psql_db.connection() as connection:
        last_updated_time = await get_last_updated_time_for_nodes(connection)
        if last_updated_time is not None and datetime.now() - last_updated_time.replace(tzinfo=None) < timedelta(minutes=30):
            logger.info(
                f"Last update for nodes table was at {last_updated_time}, which is less than 30 minutes ago - skipping refresh"
            )
            return True
        return False


async def get_and_store_nodes(config: Config) -> list[Node]:
    if await _is_recent_update(config):
        nodes = await get_all_nodes(config.psql_db)

    raw_nodes = await _fetch_nodes_from_substrate(config)
    nodes = [Node(**node.model_dump(mode="json")) for node in raw_nodes]

    async with await config.psql_db.connection() as connection:
        await migrate_nodes_to_history(connection)
        await insert_nodes(connection, nodes)

    logger.info(f"Stored {len(nodes)} nodes.")
    return nodes
