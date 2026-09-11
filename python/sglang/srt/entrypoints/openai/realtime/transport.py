"""Realtime event transport and its WebSocket implementation."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal, Protocol

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

FinishReason = Literal["normal", "buffer_overflow", "server_error"]


class RealtimeTransport(Protocol):
    """One realtime event stream, independent of the underlying connection."""

    async def receive(self) -> dict[str, Any] | None:
        """Receive an event object; raise ValueError for an invalid envelope.

        None means input ended normally, after all queued events were read.
        Disconnection or cancellation must instead interrupt session processing.
        """
        ...

    async def send(self, event: BaseModel) -> None:
        """Send a server event; raise ConnectionError if the stream is closed."""
        ...

    async def finish(self, reason: FinishReason = "normal") -> None:
        """Finish this stream, including when sending the last event failed."""
        ...


class WebSocketRealtimeTransport:
    """Queue WebSocket input while the session awaits inference.

    The endpoint runs receive_messages concurrently with the session and stops
    both on disconnect or overflow. Only the session consumes and sends events.
    """

    def __init__(self, websocket: WebSocket, max_pending_bytes: int) -> None:
        self.websocket = websocket
        self._max_pending_bytes = max_pending_bytes
        self._incoming_messages: asyncio.Queue[tuple[dict, int]] = asyncio.Queue()
        self._pending_input_bytes = 0

    async def receive_messages(self) -> bool:
        """Queue raw messages; return True on overflow, False on disconnect."""
        while True:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                return False
            # Include JSON/base64 payloads and overhead even for empty frames.
            size = (
                len((message.get("text") or "").encode("utf-8"))
                + len(message.get("bytes") or b"")
                + 256
            )
            if self._pending_input_bytes + size > self._max_pending_bytes:
                # Waiting for queue space would hide disconnects behind audio.
                return True
            self._pending_input_bytes += size
            self._incoming_messages.put_nowait((message, size))
            await asyncio.sleep(0)

    async def receive(self) -> dict[str, Any] | None:
        while True:
            # A nonempty queue and short handlers may otherwise never yield.
            await asyncio.sleep(0)
            message, size = await self._incoming_messages.get()
            self._pending_input_bytes -= size

            text = message.get("text")
            if not text:
                if message.get("bytes") is not None:
                    raise ValueError(
                        "Binary frames are not supported on /v1/realtime; "
                        "use input_audio_buffer.append with base64 audio."
                    )
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON") from e
            if not isinstance(raw, dict):
                raise ValueError("Top-level event must be a JSON object")
            return raw

    async def send(self, event: BaseModel) -> None:
        text = event.model_dump_json()
        try:
            await self.websocket.send_text(text)
        except (WebSocketDisconnect, RuntimeError) as e:
            raise ConnectionError("Realtime WebSocket is closed") from e

    async def finish(self, reason: FinishReason = "normal") -> None:
        code = {"normal": 1000, "buffer_overflow": 1009, "server_error": 1011}[reason]
        try:
            await self.websocket.close(code=code)
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.debug("[realtime] close %d failed: %s", code, e)

    def clear_pending_messages(self) -> None:
        """Release queued payloads after the receiver and session have stopped."""
        self._incoming_messages = asyncio.Queue()
        self._pending_input_bytes = 0
