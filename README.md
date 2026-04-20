# Speech Understanding Assignment 2

This repository contains a modular speech processing pipeline for code-switched Hinglish lecture audio. The project is organized around a single driver script, `pipeline.py`, plus supporting modules for preprocessing, LID, STT, phonetics, translation, TTS, prosody transfer, spoof detection, and evaluation.

## What The Main Scripts Do

### `pipeline.py`
Runs the end-to-end pipeline:

1. preprocesses the lecture audio
2. performs frame-level language identification
3. transcribes the lecture with optional constrained decoding
4. converts the transcript to IPA
5. translates the text to the target language
6. extracts a speaker embedding from the student reference voice
7. synthesizes translated speech
8. extracts and transfers prosody using DTW
9. trains and evaluates an anti-spoof classifier
10. computes evaluation metrics and saves outputs

### `find_lid_fgsm_threshold.py`
Searches for the minimum FGSM perturbation `epsilon` that flips a 5-second Hindi segment to English while respecting an SNR constraint. It is used for the adversarial robustness part of the assignment.

## Project Structure

- `preprocessing/`
  Short purpose: denoising, resampling, mono conversion, and normalization for lecture audio.

- `lid/`
  Short purpose: frame-level Hindi-English language identification, smoothing, segment refinement, utterance-level prediction, and FGSM attack utilities.

- `stt/`
  Short purpose: speech transcription and constrained decoding with an `N`-gram language model bias.

- `phonetics/`
  Short purpose: Hinglish grapheme-to-phoneme conversion and IPA-like representation generation.

- `translation/`
  Short purpose: dictionary-based and rule-based translation plus transliteration fallback for out-of-vocabulary tokens.

- `tts/`
  Short purpose: speaker embedding extraction and generative text-to-speech synthesis.

- `prosody/`
  Short purpose: F0 and energy extraction, DTW alignment, and prosody warping.

- `spoofing/`
  Short purpose: LFCC/CQCC-style feature extraction and bona fide vs spoof classification.

- `evaluation/`
  Short purpose: WER, MCD, EER, and LID switching accuracy computation.

- `outputs/`
  Short purpose: generated transcripts, translated text, synthesized audio, warped audio, metrics, and intermediate artifacts.

- `report/`
  Short purpose: IEEE-style report source and report notes.

- `tests/`
  Short purpose: unit tests for the main modules.

- `pretrained_models/`
  Short purpose: cached model files used by LID and speaker embedding modules.

## Requirements

Recommended environment:

- Python 3.11 for the most stable TTS compatibility
- Python 3.13 can run much of the repo, but Coqui XTTS is not reliable there
- PyTorch and Torchaudio installed

Main Python dependencies are listed in `requirements.txt`.

Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Inputs Needed

You need these inputs before running the pipeline:

- `original_segement.wav`
  The lecture audio segment to process.

- `student_voice_ref.wav`
  A 60-second student voice reference used for speaker embedding extraction and voice conditioning when the TTS backend supports it.

## How To Run `pipeline.py`

Basic command:

```powershell
python pipeline.py --lecture original_segement.wav --speaker student_voice_ref.wav --output_dir outputs
```

Useful options:

```powershell
python pipeline.py `
  --lecture original_segement.wav `
  --speaker student_voice_ref.wav `
  --output_dir outputs `
  --tts_backend auto `
  --tts_language en
```

### Important arguments

- `--lecture`
  Path to the input lecture audio.

- `--speaker`
  Path to the 60-second reference voice audio.

- `--output_dir`
  Directory where all outputs are saved.

- `--tts_backend`
  TTS backend to use: `auto`, `coqui`, or `mms`.

- `--tts_language`
  Language code used by the TTS backend.

- `--tts_hf_model`
  Optional Hugging Face MMS/VITS model override.

- `--resume_from_transcript`
  Skip STT and load a transcript from file. Transcripting can take a lot of time itself. So if you already have a transcript you can continue after transcripting.

- `--transcript_file`
  Transcript path used together with `--resume_from_transcript`.

- `--lid_file`
  Optional saved LID JSON for reuse.

### Example: resume from transcript

```powershell
python pipeline.py `
  --lecture original_segement.wav `
  --speaker student_voice_ref.wav `
  --output_dir outputs `
  --resume_from_transcript `
  --transcript_file outputs\transcript.txt `
  --lid_file outputs\lid_segments.json `
  --tts_backend mms `
  --tts_language en
```

## What `pipeline.py` Writes

Typical outputs inside `outputs/`:

- `lecture_clean.wav`
- `lid_segments.json`
- `transcript.txt`
- `ipa.txt`
- `translated.txt`
- `synthesized.wav`
- `output_LRL_cloned.wav`
- `prosody_mapping.npz`
- `spoof_classifier.pt`
- `metrics.json`

## How To Run `find_lid_fgsm_threshold.py`

Basic command:

```powershell
python find_lid_fgsm_threshold.py --lecture original_segement.wav --start 0 --duration 5
```

Example with saved outputs:

```powershell
python find_lid_fgsm_threshold.py `
  --lecture original_segement.wav `
  --start 0 `
  --duration 5 `
  --max-epsilon 0.05 `
  --min-snr-db 40 `
  --binary-search-steps 12 `
  --fgsm-search-steps 20 `
  --save-audio outputs\fgsm_segment.wav `
  --save-json outputs\fgsm_threshold.json
```

### Important arguments

- `--lecture`
  Input lecture audio path.

- `--start`
  Start time in seconds for the attack segment.

- `--duration`
  Segment duration in seconds. Default is `5.0`.

- `--max-epsilon`
  Upper epsilon search limit.

- `--min-snr-db`
  Minimum allowed SNR in dB.

- `--binary-search-steps`
  Number of outer binary search iterations.

- `--fgsm-search-steps`
  Number of epsilon grid steps inside each FGSM trial.

- `--save-audio`
  Optional path to save the perturbed segment.

- `--save-json`
  Optional path to save the result summary.

## Notes And Caveats

- The FGSM script requires the differentiable SpeechBrain LID backend. If the LID module falls back to Whisper, FGSM threshold search will fail.
- `mms` TTS may require a locally cached Hugging Face model if network access is blocked.
- `coqui` TTS is best used in Python 3.11 environments.
- The repository currently contains prototype outputs, so some metrics may not yet satisfy the assignment thresholds.

## Quick Submission Checklist

- Run `pipeline.py`
- Save final outputs in `outputs/`
- Run `find_lid_fgsm_threshold.py` if the SpeechBrain backend is available
- Update the report in `report/`
- Include this repository link, report, code, and readme in the final submission
