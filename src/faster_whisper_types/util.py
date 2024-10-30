from collections.abc import Iterable

from faster_whisper.transcribe import Segment as FwSegment
from faster_whisper.transcribe import TranscriptionInfo as FwTranscriptionInfo
from faster_whisper.transcribe import VadOptions as FwVadOptions
from pydantic import TypeAdapter

from faster_whisper_types.types import Segment, TranscriptionInfo, Word


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
    info_dict = info._asdict()
    info_dict["transcription_options"] = info_dict["transcription_options"]._asdict()
    if isinstance(info_dict["vad_options"], FwVadOptions):
        info_dict["vad_options"] = info_dict["vad_options"]._asdict()
    return TranscriptionInfo(**info_dict)


def segment_to_pydantic(s: FwSegment) -> Segment:
    """Convert faster-whisper' Segment to the corresponding Pydantic model."""
    s_dict = s._asdict()
    if isinstance(s_dict["words"], list):
        s_dict["words"] = [Word(**w._asdict()) for w in s_dict["words"]]
    return Segment(**s_dict)
