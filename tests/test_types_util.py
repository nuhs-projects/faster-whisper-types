"""
Tests Pydantic conversion of various faster-whisper types.
"""

import pytest
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.transcribe import Segment as FwSegment
from faster_whisper.transcribe import TranscriptionInfo as FwTranscriptionInfo
from faster_whisper.transcribe import TranscriptionOptions as FwTranscriptionOptions
from faster_whisper.transcribe import VadOptions as FwVadOptions
from faster_whisper.transcribe import Word as FwWord

from faster_whisper_types.types import (
    Segment,
    TranscriptionInfo,
    WhisperBatchOptions,
    WhisperOptions,
)
from faster_whisper_types.util import (
    fw_transcribe_output_to_pydantic,
    segment_to_pydantic,
    transcription_info_to_pydantic,
)


@pytest.fixture
def audio_file() -> str:
    return "tests/audio/short.flac"


@pytest.fixture(
    scope="module",
    params=[
        [
            FwWord(start=10.48, end=10.88, word=" We", probability=0.91064453125),
            FwWord(start=10.88, end=11.06, word=" are", probability=0.80322265625),
        ],
        None,
    ],
)
def segment(request) -> FwSegment:
    return FwSegment(
        id=3,
        seek=2490,
        start=10.700000000000001,
        end=18.08,
        text=" We are exploring the risks, the ways in which deliberate online falsehoods are spread,",
        tokens=[
            50899,
            492,
        ],
        avg_logprob=-0.26447609329924865,
        compression_ratio=1.597883597883598,
        no_speech_prob=0.0135040283203125,
        words=request.param,
        temperature=0.0,
    )


@pytest.fixture(
    scope="module",
    params=[
        FwVadOptions(onset=10, min_silence_duration_ms=100, max_speech_duration_s=100),
        None,
    ],
)
def transcription_info(request) -> FwTranscriptionInfo:
    return FwTranscriptionInfo(
        language="en",
        language_probability=0.98291015625,
        duration=60.0,
        duration_after_vad=60.0,
        all_language_probs=[
            ("en", 0.98291015625),
            ("ms", 0.0034351348876953125),
        ],
        transcription_options=FwTranscriptionOptions(
            beam_size=5,
            best_of=5,
            patience=1,
            length_penalty=1,
            repetition_penalty=1,
            no_repeat_ngram_size=0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=True,
            prompt_reset_on_temperature=0.5,
            temperatures=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            initial_prompt=None,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=(
                1,
                2,
            ),  # pyright: ignore[reportArgumentType] This is the actual output from faster-whisper
            without_timestamps=False,
            max_initial_timestamp=1.0,
            word_timestamps=True,
            prepend_punctuations="\"'“¿([{-",
            append_punctuations="\"'.。,，!！?？:：”)]}、",  # noqa: RUF001
            multilingual=False,
            max_new_tokens=None,
            clip_timestamps="0",
            hallucination_silence_threshold=None,
            hotwords=None,
        ),
        vad_options=request.param,
    )


def test_segment_to_pydantic(segment):
    output = segment_to_pydantic(segment)
    assert isinstance(output, Segment)


def test_transcription_info_to_pydantic(transcription_info):
    output = transcription_info_to_pydantic(transcription_info)
    assert isinstance(output, TranscriptionInfo)


def test_fw_transcribe_output_to_pydantic(transcription_info, segment):
    output = fw_transcribe_output_to_pydantic(([segment], transcription_info))
    assert isinstance(output[0], list)
    assert isinstance(output[0][0], Segment)
    assert isinstance(output[1], TranscriptionInfo)


def test_whisper_options(audio_file):
    model = WhisperModel("tiny.en")
    segments, info = fw_transcribe_output_to_pydantic(
        model.transcribe(audio_file, **WhisperOptions().model_dump())
    )
    assert isinstance(segments[0], Segment)
    assert isinstance(info, TranscriptionInfo)


def test_whisper_batch_options(audio_file):
    model = BatchedInferencePipeline(WhisperModel("tiny.en"))
    segments, info = fw_transcribe_output_to_pydantic(
        model.transcribe(audio_file, **WhisperBatchOptions(batch_size=2).model_dump())
    )
    assert isinstance(segments[0], Segment)
    assert isinstance(info, TranscriptionInfo)


def test_dict_diff():
    option1 = WhisperOptions(language="en")
    option2 = WhisperOptions(language="zh")
    assert option1.dict_diff(option2) == {"language": "zh"}
