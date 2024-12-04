from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel
from python_utils.dict_diff import dict_diff


class VadOptions(BaseModel):
    onset: float = 0.5
    offset: float = onset - 0.15
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400


class _Base(BaseModel):
    task: Literal["transcribe", "translate"] = "transcribe"
    language: str | None = None
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1
    length_penalty: float = 1
    repetition_penalty: float = 1
    no_repeat_ngram_size: int = 0
    temperature: float | list[float] | tuple[float, ...] = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]
    compression_ratio_threshold: float | None = 2.4
    log_prob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6
    initial_prompt: str | Iterable[int] | None = None
    prefix: str | None = None
    suppress_blank: bool = True
    suppress_tokens: list[int] | None = [-1]
    word_timestamps: bool = False
    prepend_punctuations: str = "\"'“¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、"  # noqa: RUF001
    vad_parameters: dict | VadOptions | None = None
    max_new_tokens: int | None = None
    chunk_length: int | None = None
    hotwords: str | None = None

    def dict_diff(self, other: BaseModel) -> dict[str, Any]:
        return dict_diff(self.model_dump(), other.model_dump())


class TranscriptionOptions(_Base):
    condition_on_previous_text: bool
    prompt_reset_on_temperature: float
    temperatures: Sequence[float]
    without_timestamps: bool
    max_initial_timestamp: float
    multilingual: bool
    clip_timestamps: str | list[dict] | list[float] | None = None
    hallucination_silence_threshold: float | None


class WhisperOptions(_Base):
    condition_on_previous_text: bool = True
    prompt_reset_on_temperature: float = 0.5
    without_timestamps: bool = False
    max_initial_timestamp: float = 1.0
    multilingual: bool = False
    vad_filter: bool = False
    clip_timestamps: str | list[float] = "0"
    hallucination_silence_threshold: float | None = None
    language_detection_threshold: float | None = None
    language_detection_segments: int = 1


class WhisperBatchOptions(_Base):
    log_progress: bool = False
    without_timestamps: bool = True
    vad_filter: bool = True
    clip_timestamps: list[dict] | None = None
    batch_size: int = 16


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: float


class Segment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[Word] | None
    temperature: float | None = 1.0


class TranscriptionInfo(BaseModel):
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: list[tuple[str, float]] | None
    transcription_options: TranscriptionOptions
    vad_options: VadOptions | None
