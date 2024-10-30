# Pydantic Types for faster-whisper

Pydantic types to make serializing to/from JSON easier for [faster-whisper].

This repo is locked to a specific commit of the [faster-whisper] repository.

## Usage

Parse JSON into model parameters:

```python
from faster_whisper_types.types import WhisperOptions

model = WhisperModel("tiny.en")
options = WhisperOptions.model_validate_json(some_json_data)
segments, info = model.transcribe(audio_file, **options.model_dump())
```

Convert faster-whisper output to JSON:

```python
from faster_whisper import WhisperModel
from faster_whisper_types.util import fw_transcribe_output_to_pydantic

model = WhisperModel("tiny.en", "cuda", compute_type="float16")

segments, info = fw_transcribe_output_to_pydantic(model.transcribe("tests/audio/short.flac"))

info.model_dump_json(indent=2)
# {
#   "language": "en",
#   "language_probability": 1.0,
#   "duration": 10.008,
#   "duration_after_vad": 10.008,
#   "all_language_probs": null,
# ...
```

## Testing

`coverage run --branch -m pytest && coverage html`

[faster-whisper]: https://github.com/SYSTRAN/faster-whisper
