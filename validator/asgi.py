import asyncio
import os

import uvicorn
from dotenv import load_dotenv

from validator.utils.call_endpoint import sign_up_cron_job
from validator.utils.call_endpoint import sign_up_to_gradients


load_dotenv(os.getenv("ENV_FILE", ".env"))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fiber.logging_utils import get_logger

from validator.core.config import load_config
from validator.core.cycle import init_validator_cycles
from validator.endpoints.health import factory_router as health_router
from validator.endpoints.tasks import factory_router as tasks_router


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug("Entering lifespan context manager")
    config = load_config()

    logger.debug("Attempting to connect to PostgreSQL")
    await config.psql_db.connect()
    logger.debug("PostgreSQL connected successfully")

    logger.debug("Attempting to connect to Redis")
    await config.redis_db.ping()
    logger.debug("Redis connected successfully")

    logger.info("Signing up to Gradients API...")
    await sign_up_to_gradients(config.keypair)

    asyncio.create_task(sign_up_cron_job(config.keypair))

    logger.info("Starting up...")
    app.state.config = config

    # Start the validation cycles
    try:
        logger.debug("Initializing validator cycle")
        init_validator_cycles(config)
        logger.debug("Validator cycle initialized")
    except Exception as e:
        logger.error(f"Failed to initialize validator cycle: {e}")

    yield

    logger.info("Shutting down...")
    await config.psql_db.close()
    await config.redis_db.close()


def factory() -> FastAPI:
    logger.debug("Entering factory function")
    app = FastAPI(lifespan=lifespan)

    app.include_router(health_router())
    app.include_router(tasks_router())

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.debug(f"App created with {len(app.routes)} routes")
    return app


if __name__ == "__main__":
    logger.info("Starting main validator")

    uvicorn.run(factory(), host="0.0.0.0", port=8010)
