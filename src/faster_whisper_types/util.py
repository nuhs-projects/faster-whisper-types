from collections.abc import Iterable
from dataclasses import asdict

from faster_whisper.transcribe import Segment as FwSegment
from faster_whisper.transcribe import TranscriptionInfo as FwTranscriptionInfo
from pydantic import TypeAdapter

from faster_whisper_types.types import Segment, TranscriptionInfo


def fw_transcribe_output_to_pydantic(
    o: tuple[Iterable[FwSegment], FwTranscriptionInfo],
) -> tuple[list[Segment], TranscriptionInfo]:
    """Convert the output of a faster-whisper model's `.transcribe()` to the corresponding Pydantic models."""
    ta = TypeAdapter(tuple[list[Segment], TranscriptionInfo])
    return ta.validate_python(
        ([segment_to_pydantic(s) for s in o[0]], transcription_info_to_pydantic(o[1]))
    )


def transcription_info_to_pydantic(info: FwTranscriptionInfo) -> TranscriptionInfo:
    """Convert faster-whisper's TranscriptionInfo to the corresponding Pydantic model."""
    return TranscriptionInfo.model_validate(asdict(info))


def segment_to_pydantic(s: FwSegment) -> Segment:
    """Convert faster-whisper' Segment to the corresponding Pydantic model."""
    return Segment.model_validate(asdict(s))
