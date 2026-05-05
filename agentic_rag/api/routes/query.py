"""query.py — agentic query endpoint + persistent WebSocket streaming."""

import json
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..dependencies import get_agent
from ..models import QueryRequest, QueryResponse

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Run the agentic RAG loop and return a final answer.
    The agent selects which collection to search based on the query.
    """
    try:
        agent = await get_agent()

        if request.max_iterations is not None:
            agent.config.max_iterations = request.max_iterations

        query_text = request.query
        if request.collection_hint:
            query_text = (
                f"[Search in collection: {request.collection_hint}] {request.query}"
            )

        answer = await agent.run(query_text)
        sources = agent.get_sources()

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
        )

    except TimeoutError:
        raise HTTPException(status_code=504, detail="Agent processing timed out")
    except Exception as e:
        log.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    Persistent WebSocket — stays open across multiple queries.

    Client sends:
        {"type": "query", "payload": {"query": "...", "collection_hint": "..."}}

    Server emits AgentEvent objects as they occur, then loops back waiting
    for the next message. Connection only closes on client disconnect.
    """
    await websocket.accept()
    log.info("WebSocket client connected")

    async def send(data: dict):
        """Send JSON, ignore errors if client already disconnected."""
        try:
            await websocket.send_json(data)
        except Exception:
            pass

    try:
        while True:
            # ── Receive next message ──────────────────────────────────────────
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                log.info("WebSocket client disconnected")
                return

            # ── Parse ─────────────────────────────────────────────────────────
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send({"type": "error", "content": "Invalid JSON"})
                continue

            if msg.get("type") != "query":
                await send({"type": "error", "content": "Expected type=query"})
                continue

            payload = msg.get("payload", {})
            query_text = payload.get("query", "").strip()
            collection_hint = payload.get("collection_hint")

            if not query_text:
                await send({"type": "error", "content": "Missing query"})
                continue

            # ── Run agent ─────────────────────────────────────────────────────
            try:
                agent = await get_agent()

                if collection_hint:
                    query_text = (
                        f"[Search in collection: {collection_hint}] {query_text}"
                    )

                async def on_event(event):
                    await send(
                        {
                            "type": event.type
                            if isinstance(event.type, str)
                            else event.type.value,
                            "content": event.content,
                            "data": event.data,
                            "timestamp": event.timestamp,
                        }
                    )

                await agent.run(query_text, on_event=on_event)

            except Exception as e:
                # Agent error — send error event but keep WebSocket open
                log.error("Agent error during WebSocket query: %s", e, exc_info=True)
                await send({"type": "error", "content": str(e)})
                # Continue loop — client can send another query

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as e:
        # Unexpected error on the receive loop itself — log and close
        log.error("WebSocket receive loop error: %s", e, exc_info=True)
        await send({"type": "error", "content": f"Server error: {e}"})
