import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from lims_agent.agent.graph import build_agent_graph
from lims_agent.api.routes import chat, health
from lims_agent.core.config import Settings, get_settings
from lims_agent.core.logging import configure_logging
from lims_agent.middleware.validation import ValidationMiddleware
from lims_agent.rag.retriever import build_retriever
from lims_agent.tools.samples import get_sample_data
from lims_agent.tools.sops import make_get_sop_rules

logger = logging.getLogger(__name__)


def configure_langsmith(settings: Settings) -> None:
    tracing_enabled = settings.langsmith_tracing and bool(settings.langsmith_api_key)
    if settings.langsmith_tracing and not tracing_enabled:
        logger.warning("langsmith.tracing_disabled reason=missing_api_key")
    os.environ["LANGSMITH_TRACING"] = str(tracing_enabled).lower()
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_langsmith(settings)
    logger.info("app.startup title=%s version=%s", settings.api_title, settings.api_version)

    database_url = settings.database_url
    conn_string = database_url[10:] if database_url.startswith("sqlite:///") else database_url

    async with AsyncSqliteSaver.from_conn_string(conn_string) as checkpointer:
        logger.info("checkpointer.initialized db=%s", database_url)

        retriever = build_retriever(settings)
        tools = [get_sample_data, make_get_sop_rules(retriever)]

        agent_graph = build_agent_graph(tools, checkpointer, settings)

        middleware = ValidationMiddleware(settings)
        app.state.agent_invoke = middleware.wrap_agent(agent_graph)
        app.state.agent_graph = agent_graph

        logger.info("validation.middleware_enabled strict=%s", settings.block_uncertain_responses)

        yield

    logger.info("app.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="Agentic SOP and AI assistant for LIMS.",
        lifespan=lifespan,
    )
    app.include_router(chat.router)
    app.include_router(health.router)
    return app


app = create_app()


def run() -> None:
    uvicorn.run(
        "lims_agent.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()
