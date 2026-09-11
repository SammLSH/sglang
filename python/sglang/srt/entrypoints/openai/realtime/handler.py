"""WebSocket entry for realtime transcription.

Assembles the transport and session, and owns connection task cleanup.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from openai.types.realtime import RealtimeErrorEvent
from openai.types.realtime.realtime_error import RealtimeError

from sglang.srt.entrypoints.openai.realtime.encoder_window_policy import (
    ResolvedEncoderWindowPolicy,
)
from sglang.srt.entrypoints.openai.realtime.session import RealtimeASRSession
from sglang.srt.entrypoints.openai.realtime.transport import WebSocketRealtimeTransport
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_serving
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import random_uuid

logger = logging.getLogger(__name__)

# Preserve the original default queue budget: 60 s * 48,000 samples/s *
# 2 bytes/sample (PCM16) * 2 for base64/JSON overhead = 11,520,000 bytes.
# Keep this fixed so raising the item duration limit does not grow the backlog.
_MAX_PENDING_INPUT_BYTES = 11_520_000


async def _safe_send(websocket: WebSocket, text: str) -> None:
    try:
        await websocket.send_text(text)
    except (WebSocketDisconnect, RuntimeError) as e:
        logger.debug("[realtime] send failed (peer gone): %s", e)


async def _safe_close(websocket: WebSocket) -> None:
    try:
        await websocket.close()
    except (WebSocketDisconnect, RuntimeError) as e:
        logger.debug("[realtime] close failed (already closed): %s", e)


async def _reject_before_session(
    websocket: WebSocket,
    code: str,
    message: str,
    *,
    error_type: str = "invalid_request_error",
) -> None:
    """Reject path that runs before acquiring the session semaphore, so
    unsupported / over-capacity peers don't hold a session slot."""
    try:
        await websocket.accept()
    except (WebSocketDisconnect, RuntimeError) as e:
        logger.debug("[realtime] reject: accept failed: %s", e)
        return
    logger.info("[realtime] rejected (%s)", code)
    envelope = RealtimeErrorEvent(
        event_id=f"event_{random_uuid()}",
        type="error",
        error=RealtimeError(type=error_type, code=code, message=message),
    )
    await _safe_send(websocket, envelope.model_dump_json())
    await _safe_close(websocket)


async def _run_connection_tasks(
    transport: WebSocketRealtimeTransport, session: RealtimeASRSession
) -> None:
    """Stop and await both tasks when the WebSocket or the session exits."""
    receiver_task = asyncio.create_task(transport.receive_messages())
    consumer_task = asyncio.create_task(session.run())
    overflow = False
    try:
        done, _ = await asyncio.wait(
            (receiver_task, consumer_task), return_when=asyncio.FIRST_COMPLETED
        )
        if receiver_task in done:
            overflow = receiver_task.result()
        else:
            consumer_task.result()
    finally:
        for task in (receiver_task, consumer_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(receiver_task, consumer_task, return_exceptions=True)
        transport.clear_pending_messages()

    if overflow:
        await session.send_input_overflow_error()


async def handle_realtime_transcription(
    websocket: WebSocket,
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    server_args: ServerArgs,
    session_semaphore: asyncio.Semaphore,
    encoder_window: Optional[ResolvedEncoderWindowPolicy] = None,
) -> None:
    """WS endpoint for /v1/realtime. Pre-session validation runs before
    the semaphore so rejects don't consume a session slot; the
    ``async with`` then guarantees the slot is released even if
    RealtimeASRSession raises."""
    if not adapter.supports_chunked_streaming:
        await _reject_before_session(
            websocket,
            "not_supported",
            "Model does not support streaming ASR",
        )
        return

    if session_semaphore.locked():
        await _reject_before_session(
            websocket,
            "too_many_sessions",
            f"Maximum concurrent sessions reached "
            f"({get_serving().asr_max_concurrent_sessions}).",
            error_type="rate_limit_exceeded",
        )
        return

    async with session_semaphore:
        session = None
        try:
            try:
                await websocket.accept()
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.debug("[realtime] accept failed: %s", e)
                return
            transport = WebSocketRealtimeTransport(
                websocket, max_pending_bytes=_MAX_PENDING_INPUT_BYTES
            )
            session = RealtimeASRSession(
                transport,
                tokenizer_manager,
                adapter,
                server_args,
                encoder_window=encoder_window,
            )
            await session.send_session_created()
            await _run_connection_tasks(transport, session)
        except (WebSocketDisconnect, ConnectionError):
            logger.info("[realtime] client disconnected (normal)")
        except Exception:
            logger.exception("[realtime] unexpected error in session")
            if session is not None:
                try:
                    await session.send_error(
                        "inference_failed",
                        "Internal server error",
                        error_type="server_error",
                    )
                except ConnectionError as e:
                    logger.debug("[realtime] failed to notify client: %s", e)
                return
            envelope = RealtimeErrorEvent(
                event_id=f"event_{random_uuid()}",
                type="error",
                error=RealtimeError(
                    type="server_error",
                    code="inference_failed",
                    message="Internal server error",
                ),
            )
            await _safe_send(websocket, envelope.model_dump_json())
        finally:
            await _safe_close(websocket)
