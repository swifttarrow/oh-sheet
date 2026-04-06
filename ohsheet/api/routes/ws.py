"""WebSocket endpoint for live job updates.

Connect to ``/v1/jobs/{job_id}/ws`` to stream JobEvents as JSON. Late
subscribers receive a replay of all events that have already happened so
they don't miss the start of the pipeline. The connection closes after the
terminal event (``job_succeeded`` or ``job_failed``).
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from ohsheet.api.deps import get_job_manager
from ohsheet.jobs.manager import JobManager

router = APIRouter()


@router.websocket("/jobs/{job_id}/ws")
async def job_events_ws(
    websocket: WebSocket,
    job_id: str,
    manager: Annotated[JobManager, Depends(get_job_manager)],
) -> None:
    await websocket.accept()

    queue = await manager.subscribe(job_id)
    if queue is None:
        await websocket.send_json({"error": f"job not found: {job_id}"})
        await websocket.close(code=1008)
        return

    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event.model_dump(mode="json"))
            if event.type in ("job_succeeded", "job_failed"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(job_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass
