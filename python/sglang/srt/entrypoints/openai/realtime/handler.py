"""Connection lifecycle for realtime transcription.

Validates admission, creates the session, and owns connection task cleanup.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from openai.types.realtime import RealtimeErrorEvent
from openai.types.realtime.realtime_error import RealtimeError

from sglang.srt.entrypoints.openai.realtime.audio_transcription.windowed_transcription import (
    ResolvedEncoderWindowPolicy,
)
from sglang.srt.entrypoints.openai.realtime.session import RealtimeASRSession
from sglang.srt.entrypoints.openai.realtime.transport import RealtimeTransport
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_serving
from sglang.srt.utils import random_uuid

logger = logging.getLogger(__name__)


async def _safe_send(transport: RealtimeTransport, event: RealtimeErrorEvent) -> None:
    try:
        await transport.send(event)
    except ConnectionError as e:
        logger.debug("[realtime] send failed (peer gone): %s", e)


async def _reject_before_session(
    transport: RealtimeTransport,
    code: str,
    message: str,
    *,
    error_type: str = "invalid_request_error",
) -> None:
    """Reject path that runs before acquiring the session semaphore, so
    unsupported / over-capacity peers don't hold a session slot."""
    try:
        await transport.open()
    except ConnectionError as e:
        logger.debug("[realtime] reject: open failed: %s", e)
        return
    try:
        logger.info("[realtime] rejected (%s)", code)
        envelope = RealtimeErrorEvent(
            event_id=f"event_{random_uuid()}",
            type="error",
            error=RealtimeError(type=error_type, code=code, message=message),
        )
        await _safe_send(transport, envelope)
    finally:
        await transport.finish()


async def _run_connection_tasks(
    transport: RealtimeTransport, session: RealtimeASRSession
) -> None:
    """Drain normal input completion; cancel on disconnection or session exit."""
    receiver_task = asyncio.create_task(transport.receive_messages())
    consumer_task = asyncio.create_task(session.run())
    overflow = False
    try:
        done, _ = await asyncio.wait(
            (receiver_task, consumer_task), return_when=asyncio.FIRST_COMPLETED
        )
        if receiver_task in done:
            overflow = receiver_task.result()
            if not overflow:
                await consumer_task
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
    transport: RealtimeTransport,
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    session_semaphore: asyncio.Semaphore,
    encoder_window: Optional[ResolvedEncoderWindowPolicy] = None,
) -> None:
    """Run one realtime session over the supplied transport.

    Pre-session validation runs before the semaphore so rejects don't consume a
    session slot. The ``async with`` releases the slot even if session creation
    or processing raises.
    """
    if not adapter.supports_chunked_streaming:
        await _reject_before_session(
            transport,
            "not_supported",
            "Model does not support streaming ASR",
        )
        return

    if session_semaphore.locked():
        await _reject_before_session(
            transport,
            "too_many_sessions",
            f"Maximum concurrent sessions reached "
            f"({get_serving().asr_max_concurrent_sessions}).",
            error_type="rate_limit_exceeded",
        )
        return

    async with session_semaphore:
        session = None
        try:
            await transport.open()
            session = RealtimeASRSession(
                transport,
                tokenizer_manager,
                adapter,
                encoder_window=encoder_window,
            )
            await session.send_session_created()
            await _run_connection_tasks(transport, session)
        except ConnectionError:
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
            await _safe_send(transport, envelope)
        finally:
            await transport.finish()
