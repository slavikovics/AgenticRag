import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..dependencies import get_agent, get_llm_client
from ..models import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def agentic_query(request: QueryRequest):
    try:
        agent = await get_agent(model=request.model, temperature=request.temperature)
        agent.config.max_iterations = request.max_iterations
        answer = await agent.run(request.query)
        sources = agent.get_sources()
        llm = await get_llm_client()
        cost = getattr(llm, "total_cost", None)
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            iterations=agent.config.max_iterations,
            cost_usd=cost,
        )
    except asyncio.TimeoutError:
        logger.error("Agent query timeout")
        raise HTTPException(status_code=504, detail="Agent processing timeout")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get("type") != "query":
                await websocket.send_json(
                    {"type": "error", "content": "Invalid message type"}
                )
                continue
            query = data.get("payload", {}).get("query")
            if not query:
                await websocket.send_json({"type": "error", "content": "Missing query"})
                continue
            agent = await get_agent()

            async def handle_event(event):
                event_type = (
                    event.type.value if hasattr(event.type, "value") else event.type
                )
                await websocket.send_json(
                    {
                        "type": event_type,
                        "content": event.content,
                        "data": event.data,
                        "timestamp": event.timestamp,
                    }
                )

            await websocket.send_json(
                {"type": "thinking", "content": f"Processing: {query}"}
            )
            answer = await agent.run(query, on_event=handle_event)
            await websocket.send_json({"type": "complete", "content": "Done"})
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except:
            pass
