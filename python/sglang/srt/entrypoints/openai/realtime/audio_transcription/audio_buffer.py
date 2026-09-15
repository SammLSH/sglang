import asyncio

import msgspec
import numpy as np

# Realtime input is validated as PCM16; keep all byte offsets sample-aligned.
PCM_SAMPLE_WIDTH_BYTES = 2


class AudioBuffer(msgspec.Struct):
    """Resident PCM16 range [base_offset_bytes, received_bytes), with absolute cursors."""

    data: bytearray = msgspec.field(default_factory=bytearray)
    # Resident bytes may start after offset zero once compaction drops them.
    base_offset_bytes: int = 0
    # A deferred decode advances only `last_attempted`, so pacing waits for new
    # input while `last_processed` keeps that audio in the next request.
    last_attempted_offset_bytes: int = 0
    last_processed_offset_bytes: int = 0

    @property
    def received_bytes(self) -> int:
        return self.base_offset_bytes + len(self.data)

    def append_pcm(self, pcm: bytes) -> None:
        self.data.extend(pcm)

    def snapshot(self, start_offset_bytes: int, end_offset_bytes: int) -> bytes:
        """Copy the audio in absolute range [start, end); the copy stays valid
        after discard_before() drops the underlying bytes."""
        start = start_offset_bytes - self.base_offset_bytes
        end = end_offset_bytes - self.base_offset_bytes
        if not (0 <= start <= end <= len(self.data)):
            raise ValueError(
                "audio range "
                f"[{start_offset_bytes}, {end_offset_bytes}) is outside resident "
                f"[{self.base_offset_bytes}, {self.received_bytes})"
            )
        else:
            return bytes(memoryview(self.data)[start:end])

    def discard_before(self, offset_bytes: int) -> None:
        """Free memory by dropping all audio before an absolute offset."""
        if offset_bytes % PCM_SAMPLE_WIDTH_BYTES:
            raise ValueError("discard offset must be PCM16 sample-aligned")
        elif (
            offset_bytes < self.base_offset_bytes or offset_bytes > self.received_bytes
        ):
            raise ValueError(
                f"discard offset {offset_bytes} is outside resident range "
                f"[{self.base_offset_bytes}, {self.received_bytes}]"
            )
        else:
            del self.data[: offset_bytes - self.base_offset_bytes]
            self.base_offset_bytes = offset_bytes


async def snapshot_samples(
    audio: AudioBuffer, start_offset_bytes: int, end_offset_bytes: int
) -> np.ndarray:
    """Copy PCM before converting it off the event loop."""
    pcm = audio.snapshot(start_offset_bytes, end_offset_bytes)

    def _convert() -> np.ndarray:
        # Match soundfile's PCM16 normalization used by the former WAV path.
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    return await asyncio.to_thread(_convert)
