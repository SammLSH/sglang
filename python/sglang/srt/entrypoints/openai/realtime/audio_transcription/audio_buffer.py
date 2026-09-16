import asyncio

import msgspec
import numpy as np

# PCM16 uses two bytes per sample, so trimming must keep both bytes together.
PCM_SAMPLE_WIDTH_BYTES = 2


class AudioBuffer(msgspec.Struct):
    """Buffered PCM16 audio with byte offsets measured from the recording's start."""

    data: bytearray = msgspec.field(default_factory=bytearray)
    # Track removed bytes so offsets still refer to the original recording.
    base_offset_bytes: int = 0
    # Track each attempt to decide when enough new audio is ready.
    last_attempted_offset_bytes: int = 0
    # Keep this unchanged when a retry is needed so that audio is included again.
    last_processed_offset_bytes: int = 0

    @property
    def received_bytes(self) -> int:
        return self.base_offset_bytes + len(self.data)

    def trim_buffer(self, new_base_offset_bytes: int) -> None:
        """Free unneeded audio to avoid keeping the entire recording in memory."""
        if (
            new_base_offset_bytes < self.base_offset_bytes
            or new_base_offset_bytes > self.received_bytes
        ):
            raise ValueError(
                f"trim offset {new_base_offset_bytes} is outside buffered audio range "
                f"[{self.base_offset_bytes}, {self.received_bytes}]"
            )
        else:
            del self.data[: new_base_offset_bytes - self.base_offset_bytes]
            self.base_offset_bytes = new_base_offset_bytes


async def extract_audio_samples(
    audio: AudioBuffer, start_offset_bytes: int, end_offset_bytes: int
) -> np.ndarray:
    """Copy audio so buffer changes cannot affect the worker thread's input."""
    start = start_offset_bytes - audio.base_offset_bytes
    end = end_offset_bytes - audio.base_offset_bytes
    if not (0 <= start <= end <= len(audio.data)):
        raise ValueError(
            "audio range "
            f"[{start_offset_bytes}, {end_offset_bytes}) is outside buffered audio range "
            f"[{audio.base_offset_bytes}, {audio.received_bytes})"
        )
    else:
        pcm = bytes(memoryview(audio.data)[start:end])

    # Scale signed PCM16 samples to [-1, 1), as soundfile does when reading WAV.
    return await asyncio.to_thread(
        lambda: np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    )
