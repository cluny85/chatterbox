import random
import numpy as np
import torch
import os
import re
import datetime
import torchaudio
import gradio as gr
import spaces
import subprocess
from pydub import AudioSegment
import ffmpeg
import librosa
import string
import difflib
import time
import gc
from chatterbox.src.chatterbox.tts import ChatterboxTTS
from concurrent.futures import ThreadPoolExecutor, as_completed
import whisper
import nltk
from nltk.tokenize import sent_tokenize
from faster_whisper import WhisperModel as FasterWhisperModel
import json
import csv
import soundfile as sf
import inspect, traceback
from chatterbox.src.chatterbox.vc import ChatterboxVC
try:
    import pyrnnoise
    _PYRNNOISE_AVAILABLE = True
except Exception:
    _PYRNNOISE_AVAILABLE = False


SETTINGS_PATH = "settings.json"
#THIS IS THE START
def load_settings():
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                d = default_settings()
                d.update(data)
                return d
            except Exception:
                return default_settings()
    else:
        return default_settings()

def save_settings(mapping):
    # Ensure "whisper_model_dropdown" is always saved as the label, not code
    whisper_model_map = {
        "tiny (~1 GB VRAM OpenAI / ~0.5 GB faster-whisper)": "tiny",
        "base (~1.2–2 GB OpenAI / ~0.7–1 GB faster-whisper)": "base",
        "small (~2–3 GB OpenAI / ~1.2–1.7 GB faster-whisper)": "small",
        "medium (~5–8 GB OpenAI / ~2.5–4.5 GB faster-whisper)": "medium",
        "large (~10–13 GB OpenAI / ~4.5–6.5 GB faster-whisper)": "large"
    }
    v = mapping.get("whisper_model_dropdown", "")
    if v not in whisper_model_map:
        label = next((k for k, code in whisper_model_map.items() if code == v), v)
        mapping["whisper_model_dropdown"] = label

    # --- Add the extra "per-generation" fields for full compatibility ---
    if "input_basename" not in mapping:
        mapping["input_basename"] = "text_input_"
    if "audio_prompt_path_input" not in mapping:
        mapping["audio_prompt_path_input"] = None
    if "generation_time" not in mapping:
        import datetime
        mapping["generation_time"] = datetime.datetime.now().isoformat()
    if "output_audio_files" not in mapping:
        mapping["output_audio_files"] = []

    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
        
def save_settings_csv(settings_dict, output_audio_files, csv_path):
    """
    Save a dict of settings and a list of output audio files to a one-row CSV.
    """
    # Prepare a flattened settings dict for CSV
    flat_settings = {}
    for k, v in settings_dict.items():
        if isinstance(v, (list, tuple)):
            flat_settings[k] = '|'.join(map(str, v))
        else:
            flat_settings[k] = v
    flat_settings['output_audio_files'] = '|'.join(output_audio_files)
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_settings.keys()))
        writer.writeheader()
        writer.writerow(flat_settings)

def save_settings_json(settings_dict, json_path):
    """
    Save the settings dict as a JSON file.
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=2, ensure_ascii=False)
        
        
# === VC TAB (NEW) ===

VC_MODEL = None  # Reuse the global DEVICE defined earlier

def get_or_load_vc_model(quiet=False):
    global VC_MODEL
    if VC_MODEL is None:
        VC_MODEL = ChatterboxVC.from_pretrained(DEVICE)
    return VC_MODEL



def voice_conversion(input_audio_path, target_voice_audio_path, chunk_sec=60, overlap_sec=0.1, disable_watermark=True, pitch_shift=0, quiet=False):
    vc_model = get_or_load_vc_model(quiet=quiet)
    model_sr = vc_model.sr

    wav, sr = sf.read(input_audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != model_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=model_sr)
        sr = model_sr

    total_sec = len(wav) / model_sr

    if total_sec <= chunk_sec:
        wav_out = vc_model.generate(
            input_audio_path,
            target_voice_path=target_voice_audio_path,
            apply_watermark=not disable_watermark,
            pitch_shift=pitch_shift
        )
        out_wav = wav_out.squeeze(0).numpy()
        return model_sr, out_wav

    # chunking logic for long files
    chunk_samples = int(chunk_sec * model_sr)
    overlap_samples = int(overlap_sec * model_sr)
    step_samples = chunk_samples - overlap_samples

    out_chunks = []
    for start in range(0, len(wav), step_samples):
        end = min(start + chunk_samples, len(wav))
        chunk = wav[start:end]
        temp_chunk_path = f"temp_vc_chunk_{start}_{end}.wav"
        sf.write(temp_chunk_path, chunk, model_sr)
        out_chunk = vc_model.generate(
            temp_chunk_path,
            target_voice_path=target_voice_audio_path,
            apply_watermark=not disable_watermark,
            pitch_shift=pitch_shift
        )
        out_chunk_np = out_chunk.squeeze(0).numpy()
        out_chunks.append(out_chunk_np)
        os.remove(temp_chunk_path)

    # Crossfade join as before...
    result = out_chunks[0]
    for i in range(1, len(out_chunks)):
        overlap = min(overlap_samples, len(out_chunks[i]), len(result))
        if overlap > 0:
            fade_out = np.linspace(1, 0, overlap)
            fade_in = np.linspace(0, 1, overlap)
            result[-overlap:] = result[-overlap:] * fade_out + out_chunks[i][:overlap] * fade_in
            result = np.concatenate([result, out_chunks[i][overlap:]])
        else:
            result = np.concatenate([result, out_chunks[i]])
    return model_sr, result

def default_settings():
    return {
        "text_input": """Three Rings for the Elven-kings under the sky,

Seven for the Dwarf-lords in their halls of stone,

Nine for Mortal Men doomed to die,

One for the Dark Lord on his dark throne

In the Land of Mordor where the Shadows lie.

One Ring to rule them all, One Ring to find them,

One Ring to bring them all and in the darkness bind them

In the Land of Mordor where the Shadows lie.""",
        "separate_files_checkbox": False,
        "export_format_checkboxes": ["flac", "mp3"],
        "disable_watermark_checkbox": True,
        "num_generations_input": 1,
        "num_candidates_slider": 3,
        "max_attempts_slider": 3,
        "bypass_whisper_checkbox": False,
        "whisper_model_dropdown": "medium (~5–8 GB OpenAI / ~2.5–4.5 GB faster-whisper)",
        "use_faster_whisper_checkbox": True,
        "enable_parallel_checkbox": True,
        "use_longest_transcript_on_fail_checkbox": True,
        "num_parallel_workers_slider": 4,
        "exaggeration_slider": 0.5,
        "cfg_weight_slider": 1.0,
        "temp_slider": 0.75,
        "seed_input": 0,
        "enable_batching_checkbox": False,
        "smart_batch_short_sentences_checkbox": True,
        "to_lowercase_checkbox": True,
        "normalize_spacing_checkbox": True,
        "fix_dot_letters_checkbox": True,
        "remove_reference_numbers_checkbox": True,
        "use_auto_editor_checkbox": False,
        "keep_original_checkbox": False,
        "threshold_slider": 0.06,
        "margin_slider": 0.2,
        "normalize_audio_checkbox": False,
        "normalize_method_dropdown": "ebu",
        "normalize_level_slider": -24,
        "normalize_tp_slider": -2,
        "normalize_lra_slider": 7,
        "sound_words_field": "",
        "use_pyrnnoise_checkbox": False,
    }
        
settings = load_settings()        
# Download both punkt and punkt_tab if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
#try:
#    nltk.data.find('tokenizers/punkt_tab')
#except LookupError:
#    nltk.download('punkt_tab')

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Select device: Apple Silicon GPU (MPS) if available, else fallback to CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# print(f"🚀 Running on device: {DEVICE}")
# ---- Determinism (CUDA / PyTorch) ----
import os as _os, torch as _torch
_torch.backends.cudnn.benchmark = False
if hasattr(_torch.backends.cudnn, "deterministic"):
    _torch.backends.cudnn.deterministic = True
try:
    _torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
if DEVICE == "cuda":
    _torch.backends.cuda.matmul.allow_tf32 = False
    _torch.backends.cudnn.allow_tf32 = False
# --------------------------------------

MODEL = None

def load_whisper_backend(model_name, use_faster_whisper, device, quiet=False):
    if use_faster_whisper:
        if not quiet: print(f"[DEBUG] Loading faster-whisper model: {model_name}")
        return FasterWhisperModel(model_name, device=device, compute_type="float16" if device=="cuda" else "float32")
    else:
        if not quiet: print(f"[DEBUG] Loading openai-whisper model: {model_name}")
        return whisper.load_model(model_name, device=device)

def get_or_load_model(quiet=False):
    global MODEL
    if MODEL is None:
        if not quiet: print("Model not loaded, initializing...")
        MODEL = ChatterboxTTS.from_pretrained(DEVICE)
        if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
            MODEL.to(DEVICE)
        if hasattr(MODEL, "eval"):
            MODEL.eval()
        if not quiet: print(f"Model loaded on device: {getattr(MODEL, 'device', 'unknown')}")
    return MODEL

# try:
#     get_or_load_model()
# except Exception as e:
#     print(f"CRITICAL: Failed to load model. Error: {e}")

def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def derive_seed(base_seed: int, chunk_idx: int, cand_idx: int, attempt_idx: int) -> int:
    """
    Deterministically derive a 32-bit seed for each (chunk, candidate, attempt)
    from the user-supplied base seed. This avoids any use of global random().
    """
    # use 64-bit mixing then clamp to 32-bit
    mix = (np.uint64(base_seed) * np.uint64(1000003)
           + np.uint64(chunk_idx) * np.uint64(10007)
           + np.uint64(cand_idx) * np.uint64(10009)
           + np.uint64(attempt_idx) * np.uint64(101))
    s = int(mix & np.uint64(0xFFFFFFFF))
    return s if s != 0 else 1


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s{2,}', ' ', text.strip())

def replace_letter_period_sequences(text: str) -> str:
    def replacer(match):
        cleaned = match.group(0).rstrip('.')
        letters = cleaned.split('.')
        return ' '.join(letters)
    return re.sub(r'\b(?:[A-Za-z]\.){2,}', replacer, text)
    
def remove_inline_reference_numbers(text):
    # Remove reference numbers after sentence-ending punctuation, but keep the punctuation
    pattern = r'([.!?,\"\'”’)\]])(\d+)(?=\s|$)'
    return re.sub(pattern, r'\1', text)


def split_into_sentences(text):
    # NLTK's Punkt tokenizer handles abbreviations and common English quirks
    return sent_tokenize(text)

def split_long_sentence(sentence, max_len=300, seps=None):
    """
    Recursively split a sentence into chunks of <= max_len using a sequence of separators.
    Tries each separator in order, splitting further as needed.
    """
    if seps is None:
        seps = [';', ':', '-', ',', ' ']

    sentence = sentence.strip()
    if len(sentence) <= max_len:
        return [sentence]

    if not seps:
        # Fallback: force split every max_len chars
        return [sentence[i:i+max_len].strip() for i in range(0, len(sentence), max_len)]

    sep = seps[0]
    parts = sentence.split(sep)

    if len(parts) == 1:
        # Separator not found, try next separator
        return split_long_sentence(sentence, max_len, seps=seps[1:])

    # Now recursively process each part, joining separator back except for the first
    chunks = []
    current = parts[0].strip()
    for part in parts[1:]:
        candidate = (current + sep + part).strip()
        if len(candidate) > max_len:
            # Split current chunk further with the next separator
            chunks.extend(split_long_sentence(current.strip(), max_len, seps=seps[1:]))
            current = part.strip()
        else:
            current = candidate
    # Process the last current
    if current:
        if len(current) > max_len:
            chunks.extend(split_long_sentence(current.strip(), max_len, seps=seps[1:]))
        else:
            chunks.append(current.strip())

    return chunks

    # Fallback: force split every max_len chars
    #return [sentence[i:i+max_len].strip() for i in range(0, len(sentence), max_len)]

def group_sentences(sentences, max_chars=300, quiet=False):
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if not sentence:
            if not quiet: print(f"\033[32m[DEBUG] Skipping empty sentence\033[0m")
            continue
        sentence = sentence.strip()
        sentence_len = len(sentence)

        if not quiet: print(f"\033[32m[DEBUG] Processing sentence: len={sentence_len}, content='\033[33m{sentence}...'\033[0m")

        if sentence_len > 300:
            if not quiet: print(f"\033[32m[DEBUG] Splitting overlong sentence of {sentence_len} chars\033[0m")
            for chunk in split_long_sentence(sentence, 300):
                if len(chunk) > max_chars:
                    # For extremely long non-breakable segments, just chunk them
                    for i in range(0, len(chunk), max_chars):
                        chunks.append(chunk[i:i+max_chars])
                else:
                    chunks.append(chunk)
            current_chunk = []
            current_length = 0
            continue  # Skip the rest of the loop for this sentence

        if sentence_len > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                if not quiet: print(f"\033[32m[DEBUG] Finalized chunk: {' '.join(current_chunk)}...\033[0m")
            chunks.append(sentence)
            if not quiet: print(f"\032m[DEBUG] Added long sentence as chunk: {sentence}...\033[0m")
            current_chunk = []
            current_length = 0
        elif current_length + sentence_len + (1 if current_chunk else 0) <= max_chars:
            current_chunk.append(sentence)
            current_length += sentence_len + (1 if current_chunk else 0)
            if not quiet: print(f"\033[32m[DEBUG] Adding sentence to chunk: {sentence}...\033[0m")
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                if not quiet: print(f"\033[32m[DEBUG] Finalized chunk: {' '.join(current_chunk)}...\033[0m")
            current_chunk = [sentence]
            current_length = sentence_len
            if not quiet: print(f"\033[32m[DEBUG] Starting new chunk with: {sentence}...\033[0m")

    if current_chunk:
        chunks.append(" ".join(current_chunk))
        if not quiet: print(f"\033[32m[DEBUG] Finalized final chunk: {' '.join(current_chunk)}...\033[0m")

    if not quiet:
        print(f"\033[32m[DEBUG] Total chunks created: {len(chunks)}\033[0m")
        for i, chunk in enumerate(chunks):
            print(f"\033[32m[DEBUG] Chunk {i}: len={len(chunk)}, content='\033[33m{chunk}...'\033[0m")

    return chunks

def smart_append_short_sentences(sentences, max_chars=300):
    new_groups = []
    i = 0
    while i < len(sentences):
        current = sentences[i].strip()
        if len(current) >= 20:
            new_groups.append(current)
            i += 1
        else:
            appended = False
            if i + 1 < len(sentences):
                next_sentence = sentences[i + 1].strip()
                if len(current + " " + next_sentence) <= max_chars:
                    new_groups.append(current + " " + next_sentence)
                    i += 2
                    appended = True
            if not appended and new_groups:
                if len(new_groups[-1] + " " + current) <= max_chars:
                    new_groups[-1] += " " + current
                    i += 1
                    appended = True
            if not appended:
                new_groups.append(current)
                i += 1
    return new_groups

def normalize_with_ffmpeg(input_wav, output_wav, method="ebu", i=-24, tp=-2, lra=7):
    if method == "ebu":
        loudnorm = f"loudnorm=I={i}:TP={tp}:LRA={lra}"
        (
            ffmpeg
            .input(input_wav)
            .output(output_wav, af=loudnorm)
            .overwrite_output()
            .run(quiet=True)
        )
    elif method == "peak":
        (
            ffmpeg
            .input(input_wav)
            .output(output_wav, af="alimiter=limit=-2dB")
            .overwrite_output()
            .run(quiet=True)
        )

    else:
        raise ValueError("Unknown normalization method.")
    os.replace(output_wav, input_wav)

def _convert_to_pcm48k_mono(input_wav, output_wav, sr=48000):
    """
    Convert to 48kHz, mono, s16 PCM for RNNoise (pyrnnoise) best compatibility.
    """
    subprocess.run([
        "ffmpeg", "-y", "-i", input_wav,
        "-ac", "2", "-ar", str(sr), "-sample_fmt", "s16", output_wav
    ], check=True)


def _run_pyrnnoise(input_wav, output_wav, quiet=False):
    """
    Try the pyrnnoise CLI ('denoise') first; if missing or fails, fall back to Python API.
    """
    if not _PYRNNOISE_AVAILABLE:
        if not quiet: print("[DENOISE] pyrnnoise not available; skipping.")
        return False

    if not quiet: print("[DENOISE] Running pyrnnoise (RNNoise)…")
    # Prefer CLI if present (often faster and lighter on Python mem)
    try:
        result = subprocess.run(["denoise", input_wav, output_wav], capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 1024:
            if not quiet: print(f"[DENOISE] Saved: {output_wav}")
            return True
        else:
            if not quiet: print("[DENOISE] pyrnnoise CLI failed, falling back to Python API…")
    except FileNotFoundError:
        if not quiet: print("[DENOISE] pyrnnoise CLI not found, using Python API…")

    # Python API fallback
    rate, data = sf.read(input_wav)
    denoiser = pyrnnoise.RNNoise(rate)
    denoised = denoiser.process_buffer(data)
    sf.write(output_wav, denoised, rate)
    if not quiet: print(f"[DENOISE] Saved: {output_wav}")
    return True


def _apply_pyrnnoise_in_place(wav_output_path, quiet=False):
    """
    Denoise wav_output_path with RNNoise, preserving the original path.
    Converts to 48k mono s16 for processing, then converts back to the original sample rate.
    """
    try:
        original_sr = librosa.get_samplerate(wav_output_path)
    except Exception:
        # Fallback if librosa can't read it
        original_sr = None

    tmp_48kmono = wav_output_path.replace(".wav", "_48kmono.wav")
    tmp_dn = wav_output_path.replace(".wav", "_dn.wav")
    tmp_back = wav_output_path.replace(".wav", "_dn_resamp.wav")

    try:
        _convert_to_pcm48k_mono(wav_output_path, tmp_48kmono)
        ok = _run_pyrnnoise(tmp_48kmono, tmp_dn, quiet=quiet)
        if not ok:
            return False

        # Convert back to original sample rate (if known), keep mono
        if original_sr:
            subprocess.run([
                "ffmpeg", "-y", "-i", tmp_dn, "-ar", str(original_sr), "-ac", "1", tmp_back
            ], check=True)
            os.replace(tmp_back, wav_output_path)
        else:
            # If we don't know SR, just adopt the denoised file
            os.replace(tmp_dn, wav_output_path)

        if not quiet: print(f"[DENOISE] Denoised in-place: {wav_output_path}")
        return True
    except Exception as e:
        if not quiet: print(f"[DENOISE] RNNoise failed: {e}")
        return False
    finally:
        for p in [tmp_48kmono, tmp_dn, tmp_back]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def get_wav_duration(path, quiet=False):
    try:
        return librosa.get_duration(filename=path)
    except Exception as e:
        if not quiet: print(f"[ERROR] librosa.get_duration failed: {e}")
        return float('inf')

def normalize_for_compare_all_punct(text):
    text = re.sub(r'[–—-]', ' ', text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def fuzzy_match(text1, text2, threshold=0.85):
    t1 = normalize_for_compare_all_punct(text1)
    t2 = normalize_for_compare_all_punct(text2)
    seq = difflib.SequenceMatcher(None, t1, t2)
    return seq.ratio() >= threshold

def parse_sound_word_field(user_input):
    # Accepts comma or newline separated, allows 'sound=>replacement'
    lines = [l.strip() for l in user_input.split('\n') if l.strip()]
    result = []
    for line in lines:
        if '=>' in line:
            pattern, replacement = line.split('=>', 1)
            result.append((pattern.strip(), replacement.strip()))
        else:
            result.append((line, ''))  # Remove (replace with empty string)
    return result

def smart_remove_sound_words(text, sound_words):
    for pattern, replacement in sound_words:
        if replacement:
            # 1. Handle possessive: "Baggins’" or "Baggins'" (optionally with s or S after apostrophe)
            text = re.sub(
                r'(?i)(%s)([’\']s?)' % re.escape(pattern),
                lambda m: replacement + "'s" if m.group(2) else replacement,
                text
            )
            # 2. Replace word in quotes
            text = re.sub(
                r'(["\'])%s(["\'])' % re.escape(pattern),
                lambda m: f"{m.group(1)}{replacement}{m.group(2)}",
                text,
                flags=re.IGNORECASE
            )
            # If pattern is a punctuation character (like dash), replace all
            if all(char in "-–—" for char in pattern.strip()):
                text = re.sub(re.escape(pattern), replacement, text)
            else:
                # 3. Replace as whole word (not in quotes)
                text = re.sub(
                    r'\b%s\b' % re.escape(pattern),
                    replacement,
                    text,
                    flags=re.IGNORECASE
                )
        else:
            # Remove only the pattern itself, not adjacent spaces
            text = re.sub(
                r'%s' % re.escape(pattern),
                '',
                text,
                flags=re.IGNORECASE
            )

    # --- Fix accidental joining of words caused by quote removal ---
    # Add a space if a letter is next to a letter and was separated by removed quote
    #text = re.sub(r'(\w)([’\'"“”‘’])(\w)', r'\1 \3', text)
    # Add a space between lowercase and uppercase, likely joined words (e.g., rainbowPride)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # --- Clean up doubled-up commas and extra spaces ---
    text = re.sub(r'([,\s]+,)+', ',', text)
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'(\s+,|,\s+)', ', ', text)
    text = re.sub(r'(^|[\.!\?]\s*),+', r'\1', text)
    text = re.sub(r',+\s*([\.!\?])', r'\1', text)
    return text.strip()


def whisper_check_mp(candidate_path, target_text, whisper_model, use_faster_whisper=False, quiet=False):
    import difflib
    import re
    import string
    import os

    try:
        if not quiet: print(f"\033[32m[DEBUG] Whisper checking: {candidate_path}\033[0m")
        if use_faster_whisper:
            segments, info = whisper_model.transcribe(candidate_path)
            transcribed = "".join([seg.text for seg in segments]).strip().lower()
        else:
            result = whisper_model.transcribe(candidate_path)
            transcribed = result['text'].strip().lower()
        if not quiet: print(f"\033[32m[DEBUG] Whisper transcription: '\033[33m{transcribed}' for candidate '{os.path.basename(candidate_path)}'\033[0m")
        score = difflib.SequenceMatcher(
            None,
            normalize_for_compare_all_punct(transcribed),
            normalize_for_compare_all_punct(target_text.strip().lower())
        ).ratio()
        if not quiet: print(f"\033[32m[DEBUG] Score: {score:.3f} (target: '\033[33m{target_text}')\033[0m")
        return (candidate_path, score, transcribed)
    except Exception as e:
        if not quiet: print(f"[ERROR] Whisper transcription failed for {candidate_path}: {e}")
        return (candidate_path, 0.0, f"ERROR: {e}")
        
        
def process_one_chunk(
    model, sentence_group, idx, gen_index, this_seed,
    audio_prompt_path_input, exaggeration_input, temperature_input, cfgw_input,
    disable_watermark, num_candidates_per_chunk, max_attempts_per_candidate,
    bypass_whisper_checking,
    retry_attempt_number=1,
    quiet=False
):
    candidates = []
    try:
        if not sentence_group.strip():
            if not quiet: print(f"\033[32m[DEBUG] Skipping empty sentence group at index {idx}\033[0m")
            return (idx, candidates)
        if len(sentence_group) > 300:
            if not quiet: print(f"\033[33m[WARNING] Very long sentence group at index {idx} (len={len(sentence_group)}); proceeding anyway.\033[0m")

        if not quiet: print(f"\032m[DEBUG] Processing group {idx}: len={len(sentence_group)}:\033[33m {sentence_group}\033[0m")

        for cand_idx in range(num_candidates_per_chunk):
            for attempt in range(max_attempts_per_candidate):
                candidate_seed = derive_seed(this_seed, idx, cand_idx, attempt)
                set_seed(candidate_seed)
                try:
                    if not quiet: print(f"\033[32m[DEBUG] Generating candidate {cand_idx+1} attempt {attempt+1} for chunk {idx}...\033[0m")
#                    if not quiet: print(f"[TTS DEBUG] audio_prompt_path passed: {audio_prompt_path_input!r}")
                    wav = model.generate(
                        sentence_group,
                        audio_prompt_path=audio_prompt_path_input,
                        exaggeration=min(exaggeration_input, 1.0),
                        temperature=temperature_input,
                        cfg_weight=cfgw_input,
                        apply_watermark=not disable_watermark
                    )
                    

                    candidate_path = f"temp/gen{gen_index+1}_chunk_{idx:03d}_cand_{cand_idx+1}_try{retry_attempt_number}_seed{candidate_seed}.wav"
                    torchaudio.save(candidate_path, wav, model.sr)
                    for _ in range(10):
                        if os.path.exists(candidate_path) and os.path.getsize(candidate_path) > 1024:
                            break
                        time.sleep(0.05)
                    duration = get_wav_duration(candidate_path, quiet=quiet)
                    if not quiet: print(f"\033[32m[DEBUG] Saved candidate {cand_idx+1}, attempt {attempt+1}, duration={duration:.3f}s: {candidate_path}\033[0m")
                    candidates.append({
                        'path': candidate_path,
                        'duration': duration,
                        'sentence_group': sentence_group,
                        'cand_idx': cand_idx,
                        'attempt': attempt,
                        'seed': candidate_seed,
                    })
                    break
                except Exception as e:
                    if not quiet: print(f"[ERROR] Candidate {cand_idx+1} generation attempt {attempt+1} failed: {e}")
    except Exception as exc:
        if not quiet: print(f"[ERROR] Exception in chunk {idx}: {exc}")
    return (idx, candidates)

def process_one_chunk_deterministic(
    model, sentence_group, idx, gen_index, this_seed,
    audio_prompt_path_input, exaggeration_input, temperature_input, cfgw_input,
    disable_watermark, num_candidates_per_chunk, max_attempts_per_candidate,
    bypass_whisper_checking,
    retry_attempt_number=1,
    quiet=False
):
    """
    Deterministic per-chunk generation that does NOT mutate global RNG.
    - If model.generate supports `generator`, use a per-call torch.Generator.
    - Else, fallback to a forked RNG scope + manual seeds (still thread-local).
    Also logs full tracebacks on failure so we can see the exact cause.
    """
    import inspect, traceback

    candidates = []
    try:
        if not sentence_group.strip():
            if not quiet: print(f"\033[32m[DEBUG] Skipping empty sentence group at index {idx}\033[0m")
            return (idx, candidates)
        if len(sentence_group) > 300:
            if not quiet: print(f"\033[33m[WARNING] Very long sentence group at index {idx} (len={len(sentence_group)}); proceeding anyway.\033[0m")

        if not quiet: print(f"\033[32m[DEBUG] [DET] Processing group {idx}: len={len(sentence_group)}:\033[33m {sentence_group}\033[0m")

        # Detect whether model.generate accepts a `generator` argument
        supports_generator = False
        try:
            sig = inspect.signature(model.generate)
            supports_generator = ("generator" in sig.parameters)
        except Exception:
            supports_generator = False

        model_device = str(getattr(model, "device", "cpu"))
        on_cuda = torch.cuda.is_available() and (model_device == "cuda")
        devices = [torch.cuda.current_device()] if on_cuda else []

        for cand_idx in range(num_candidates_per_chunk):
            for attempt in range(max_attempts_per_candidate):
                candidate_seed = derive_seed(this_seed, idx, cand_idx, attempt)
                if not quiet: print(f"\033[32m[DEBUG] [DET] Generating cand {cand_idx+1} attempt {attempt+1} for chunk {idx} (seed={candidate_seed}).\033[0m")

                try:
                    if supports_generator and (model_device != "mps"):
                        # Use a per-call generator on the matching device (CUDA→cuda, otherwise CPU)
                        gen_device = "cuda" if on_cuda else "cpu"
                        gen = torch.Generator(device=gen_device)
                        gen.manual_seed(int(candidate_seed) & 0xFFFFFFFFFFFFFFFF)

                        wav = model.generate(
                            sentence_group,
                            audio_prompt_path=audio_prompt_path_input,
                            exaggeration=min(exaggeration_input, 1.0),
                            temperature=temperature_input,
                            cfg_weight=cfgw_input,
                            apply_watermark=not disable_watermark,
                            generator=gen,  # isolated RNG
                        )
                    else:
                        # Fallback: fork RNG state locally and seed inside the scope
                        with torch.random.fork_rng(devices=devices, enabled=True):
                            torch.manual_seed(int(candidate_seed))
                            if on_cuda:
                                torch.cuda.manual_seed_all(int(candidate_seed))
                            wav = model.generate(
                                sentence_group,
                                audio_prompt_path=audio_prompt_path_input,
                                exaggeration=min(exaggeration_input, 1.0),
                                temperature=temperature_input,
                                cfg_weight=cfgw_input,
                                apply_watermark=not disable_watermark,
                            )

                    candidate_path = f"temp/gen{gen_index+1}_chunk_{idx:03d}_cand_{cand_idx+1}_try{retry_attempt_number}_seed{candidate_seed}.wav"
                    torchaudio.save(candidate_path, wav, model.sr)

                    # Wait briefly for filesystem consistency
                    for _ in range(10):
                        if os.path.exists(candidate_path) and os.path.getsize(candidate_path) > 1024:
                            break
                        time.sleep(0.05)

                    duration = get_wav_duration(candidate_path, quiet=quiet)
                    if not quiet: print(f"\033[32m[DEBUG] [DET] Saved cand {cand_idx+1}, attempt {attempt+1}, duration={duration:.3f}s: {candidate_path}\033[0m")
                    candidates.append({
                        'path': candidate_path,
                        'duration': duration,
                        'sentence_group': sentence_group,
                        'cand_idx': cand_idx,
                        'attempt': attempt,
                        'seed': candidate_seed,
                    })

                    # If bypass is ON we can short-circuit after first successful candidate
                    if bypass_whisper_checking:
                        break

                except Exception as e:
                    tb = traceback.format_exc()
                    if not quiet: print(f"[ERROR] Deterministic generation failed for chunk {idx}, cand {cand_idx+1}, attempt {attempt+1}: {e}\n{tb}")
                    # Continue to next attempt/candidate

    except Exception as e:
        tb = traceback.format_exc()
        if not quiet: print(f"[ERROR] process_one_chunk_deterministic failed for index {idx}: {e}\n{tb}")

    return (idx, candidates)





def generate_and_preview(*args):

    output_paths = generate_batch_tts(*args)
    audio_files = [p for p in output_paths if os.path.splitext(p)[1].lower() in [".wav", ".mp3", ".flac"]]
    dropdown_value = audio_files[0] if audio_files else None
    return output_paths, gr.update(choices=audio_files, value=dropdown_value), dropdown_value
    

def update_audio_preview(selected_path):
    return selected_path

@spaces.GPU
def generate_batch_tts(
    text: str,
    text_file,
    audio_prompt_path_input,
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float,
    use_pyrnnoise: bool,
    use_auto_editor: bool,
    ae_threshold: float,
    ae_margin: float,
    export_formats: list,
    enable_batching: bool,
    to_lowercase: bool,
    normalize_spacing: bool,
    fix_dot_letters: bool,
    remove_reference_numbers: bool,
    keep_original_wav: bool,
    smart_batch_short_sentences: bool,
    disable_watermark: bool,
    num_generations: int,
    normalize_audio: bool,
    normalize_method: str,
    normalize_level: float,
    normalize_tp: float,
    normalize_lra: float,
    num_candidates_per_chunk: int,
    max_attempts_per_candidate: int,
    bypass_whisper_checking: bool,
    whisper_model_name: str,
    enable_parallel: bool = True,
    num_parallel_workers: int = 4,
    use_longest_transcript_on_fail: bool = False,
    sound_words_field: str = "",
    use_faster_whisper: bool = False,
    generate_separate_audio_files: bool = False,
    output_dir: str = "output",
    add_filename_suffix: bool = True,
    quiet: bool = False,
) -> list[str]:
    if not quiet: print(f"[DEBUG] Received audio_prompt_path_input: {audio_prompt_path_input!r}")

    if not audio_prompt_path_input or (isinstance(audio_prompt_path_input, str) and not os.path.isfile(audio_prompt_path_input)):
        audio_prompt_path_input = None
    model = get_or_load_model(quiet=quiet)

    # PATCH: Get file basename (to prepend) if a text file was uploaded
    # Support for multiple file uploads
    # PATCH: Get file basename (to prepend) if a text file was uploaded
    # Support for multiple file uploads
    input_basename = ""

    # Robust handling for Gradio's file input (can be None, False, or list containing such)
    files = []
    if text_file:
        files = text_file if isinstance(text_file, list) else [text_file]
        # Remove any entry that's not a file-like object with a .name attribute (filters out None, False, bool)
        files = [f for f in files if hasattr(f, "name") and isinstance(getattr(f, "name", None), str)]

    if files:
        # If generating separate audio files per text file:
        if generate_separate_audio_files:
            all_jobs = []
            for fobj in files:
                try:
                    fname = os.path.basename(fobj.name)
                    base = os.path.splitext(fname)[0]
                    base = re.sub(r'[^a-zA-Z0-9_\-]', '_', base)
                    with open(fobj.name, "r", encoding="utf-8") as f:
                        file_text = f.read()
                    all_jobs.append((file_text, base))
                except Exception as e:
                    if not quiet: print(f"[ERROR] Failed to read file: {getattr(fobj, 'name', repr(fobj))} | {e}")
            # Now process each file separately and collect outputs
            all_outputs = []
            for job_text, base in all_jobs:
                output_paths = process_text_for_tts(
                    job_text, base,
                    audio_prompt_path_input,
                    exaggeration_input, temperature_input, seed_num_input, cfgw_input,
                    use_pyrnnoise,  # <-- add this
                    use_auto_editor, ae_threshold, ae_margin, export_formats, enable_batching,
                    to_lowercase, normalize_spacing, fix_dot_letters, remove_reference_numbers, keep_original_wav,
                    smart_batch_short_sentences, disable_watermark, num_generations,
                    normalize_audio, normalize_method, normalize_level, normalize_tp,
                    normalize_lra, num_candidates_per_chunk, max_attempts_per_candidate,
                    bypass_whisper_checking, whisper_model_name, enable_parallel,
                    num_parallel_workers, use_longest_transcript_on_fail, sound_words_field, use_faster_whisper,
                    quiet=quiet
                )
                all_outputs.extend(output_paths)
            return all_outputs  # Return list of output files

        # ELSE (default: join all text files as one, as before)
        all_text = []
        basenames = []
        for fobj in files:
            try:
                fname = os.path.basename(fobj.name)
                base = os.path.splitext(fname)[0]
                base = re.sub(r'[^a-zA-Z0-9_\-]', '_', base)
                basenames.append(base)
                with open(fobj.name, "r", encoding="utf-8") as f:
                    all_text.append(f.read())
            except Exception as e:
                if not quiet: print(f"[ERROR] Failed to read file: {getattr(fobj, 'name', repr(fobj))} | {e}")
        text = "\n\n".join(all_text)
        input_basename = "_".join(basenames) + "_"

        return process_text_for_tts(
            text, input_basename, audio_prompt_path_input,
            exaggeration_input, temperature_input, seed_num_input, cfgw_input,
            use_pyrnnoise,
            use_auto_editor, ae_threshold, ae_margin, export_formats, enable_batching,
            to_lowercase, normalize_spacing, fix_dot_letters, remove_reference_numbers, keep_original_wav,
            smart_batch_short_sentences, disable_watermark, num_generations,
            normalize_audio, normalize_method, normalize_level, normalize_tp,
            normalize_lra, num_candidates_per_chunk, max_attempts_per_candidate,
            bypass_whisper_checking, whisper_model_name, enable_parallel,
            num_parallel_workers, use_longest_transcript_on_fail, sound_words_field, use_faster_whisper,
            output_dir=output_dir,
            add_filename_suffix=add_filename_suffix,
            quiet=quiet
        )
    else:
        # No text file: just process the Text Input box as one job
        input_basename = "text_input"
        return process_text_for_tts(
            text, input_basename, audio_prompt_path_input,
            exaggeration_input, temperature_input, seed_num_input, cfgw_input,
            use_pyrnnoise,
            use_auto_editor, ae_threshold, ae_margin, export_formats, enable_batching,
            to_lowercase, normalize_spacing, fix_dot_letters, remove_reference_numbers, keep_original_wav,
            smart_batch_short_sentences, disable_watermark, num_generations,
            normalize_audio, normalize_method, normalize_level, normalize_tp,
            normalize_lra, num_candidates_per_chunk, max_attempts_per_candidate,
            bypass_whisper_checking, whisper_model_name, enable_parallel,
            num_parallel_workers, use_longest_transcript_on_fail, sound_words_field, use_faster_whisper,
            output_dir=output_dir,
            add_filename_suffix=add_filename_suffix,
            quiet=quiet
        )

def process_text_for_tts(
    text,
    input_basename,
    audio_prompt_path_input,
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float,
    use_pyrnnoise: bool,
    use_auto_editor: bool,
    ae_threshold: float,
    ae_margin: float,
    export_formats: list,
    enable_batching: bool,
    to_lowercase: bool,
    normalize_spacing: bool,
    fix_dot_letters: bool,
    remove_reference_numbers: bool,
    keep_original_wav: bool,
    smart_batch_short_sentences: bool,
    disable_watermark: bool,
    num_generations: int,
    normalize_audio: bool,
    normalize_method: str,
    normalize_level: float,
    normalize_tp: float,
    normalize_lra: float,
    num_candidates_per_chunk: int,
    max_attempts_per_candidate: int,
    bypass_whisper_checking: bool,
    whisper_model_name: str,
    enable_parallel: bool = True,
    num_parallel_workers: int = 4,
    use_longest_transcript_on_fail: bool = False,
    sound_words_field: str = "",
    use_faster_whisper: bool = False,
    output_dir: str = "output",
    add_filename_suffix: bool = True,
    quiet: bool = False,
) -> list[str]:
    if not quiet:
        print(f"🚀 Running on device: {DEVICE}")
        print(f"📝 Text: {text[:100]}...")
        print(f"🎤 Audio Prompt: {audio_prompt_path_input}")
        print(f"🌱 Seed: {seed_num_input}")
        print(f"🔥 Temperature: {temperature_input}")
        print(f"⚖️ CFG Weight: {cfgw_input}")
        print(f"😲 Exaggeration: {exaggeration_input}")
        print(f"💧 Watermark Disabled: {disable_watermark}")
        print(f"🔄 Generations: {num_generations}")

    model = get_or_load_model(quiet=quiet)
    if not audio_prompt_path_input or not os.path.isfile(audio_prompt_path_input):
        audio_prompt_path_input = None

    # --- Text cleaning ---
    sound_words = parse_sound_word_field(sound_words_field)
    if sound_words:
        text = smart_remove_sound_words(text, sound_words)
    if to_lowercase:
        text = text.lower()
    if normalize_spacing:
        text = normalize_whitespace(text)
    if fix_dot_letters:
        text = replace_letter_period_sequences(text)
    if remove_reference_numbers:
        text = remove_inline_reference_numbers(text)

    # --- Sentence chunking ---
    sentences = split_into_sentences(text)
    if enable_batching:
        if smart_batch_short_sentences:
            sentence_groups = smart_append_short_sentences(sentences)
        else:
            sentence_groups = group_sentences(sentences, quiet=quiet)
    else:
        sentence_groups = sentences
    sentence_groups = [s for s in sentence_groups if s.strip()]
    if not sentence_groups:
        if not quiet: print("[ERROR] No sentence groups left after cleaning. Aborting.")
        return []

    # --- Setup for generation ---
    os.makedirs("temp", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    all_output_files = []
    whisper_model = None
    if not bypass_whisper_checking:
        whisper_model_name_code = whisper_model_map.get(whisper_model_name, whisper_model_name)
        whisper_model = load_whisper_backend(whisper_model_name_code, use_faster_whisper, DEVICE, quiet=quiet)

    # --- Main generation loop ---
    for gen_index in range(num_generations):
        this_seed = seed_num_input if seed_num_input != 0 else random.randint(1, 2**32 - 1)
        if not quiet: print(f"\n--- Generation {gen_index + 1}/{num_generations} (Seed: {this_seed}) ---")

        all_candidates = [[] for _ in sentence_groups]
        if enable_parallel:
            with ThreadPoolExecutor(max_workers=num_parallel_workers) as executor:
                futures = [
                    executor.submit(
                        process_one_chunk_deterministic,
                        model, group, idx, gen_index, this_seed,
                        audio_prompt_path_input, exaggeration_input, temperature_input, cfgw_input,
                        disable_watermark, num_candidates_per_chunk, max_attempts_per_candidate,
                        bypass_whisper_checking, quiet=quiet
                    ) for idx, group in enumerate(sentence_groups)
                ]
                for future in as_completed(futures):
                    idx, candidates = future.result()
                    all_candidates[idx] = candidates
        else:
            for idx, group in enumerate(sentence_groups):
                _, candidates = process_one_chunk_deterministic(
                    model, group, idx, gen_index, this_seed,
                    audio_prompt_path_input, exaggeration_input, temperature_input, cfgw_input,
                    disable_watermark, num_candidates_per_chunk, max_attempts_per_candidate,
                    bypass_whisper_checking, quiet=quiet
                )
                all_candidates[idx] = candidates

        # --- Whisper validation and selection ---
        best_audio_paths = []
        if bypass_whisper_checking:
            if not quiet: print("[INFO] Bypassing Whisper validation. Using first generated candidate for each chunk.")
            for candidates in all_candidates:
                if candidates:
                    best_audio_paths.append(candidates[0]['path'])
        else:
            if not quiet: print("\n--- Whisper Validation ---")
            all_whisper_results = [[] for _ in all_candidates]
            flat_candidate_paths = [c['path'] for candidates in all_candidates for c in candidates]

            if enable_parallel:
                with ThreadPoolExecutor(max_workers=num_parallel_workers) as executor:
                    futures = {
                        executor.submit(whisper_check_mp, path, next(c['sentence_group'] for cands in all_candidates for c in cands if c['path'] == path), whisper_model, use_faster_whisper, quiet=quiet): path
                        for path in flat_candidate_paths
                    }
                    for future in as_completed(futures):
                        path, score, transcript = future.result()
                        for i, candidates in enumerate(all_candidates):
                            for j, cand in enumerate(candidates):
                                if cand['path'] == path:
                                    all_whisper_results[i].append({'path': path, 'score': score, 'transcript': transcript})
            else:
                for i, candidates in enumerate(all_candidates):
                    for cand in candidates:
                        _, score, transcript = whisper_check_mp(cand['path'], cand['sentence_group'], whisper_model, use_faster_whisper, quiet=quiet)
                        all_whisper_results[i].append({'path': path, 'score': score, 'transcript': transcript})


            for i, results in enumerate(all_whisper_results):
                if not results:
                    if not quiet: print(f"[WARNING] No candidates generated for chunk {i}. It will be silent.")
                    continue
                
                target_text = sentence_groups[i]
                perfect_matches = [r for r in results if fuzzy_match(r['transcript'], target_text)]
                if perfect_matches:
                    best_audio_paths.append(perfect_matches[0]['path'])
                    if not quiet: print(f"✅ Chunk {i}: Perfect match found.")
                else:
                    if use_longest_transcript_on_fail:
                        best_fallback = max(results, key=lambda r: len(r['transcript']))
                        if not quiet: print(f"⚠️ Chunk {i}: No perfect match. Using longest transcript: '{best_fallback['transcript'][:50]}...'")
                        best_audio_paths.append(best_fallback['path'])
                    else:
                        best_fallback = max(results, key=lambda r: r['score'])
                        if not quiet: print(f"⚠️ Chunk {i}: No perfect match. Using best score ({best_fallback['score']:.2f}): '{best_fallback['transcript'][:50]}...'")
                        best_audio_paths.append(best_fallback['path'])

        # --- Audio concatenation and post-processing ---
        if not best_audio_paths:
            if not quiet: print("[ERROR] No audio was generated. Aborting post-processing.")
            continue

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")[:-3]
        
        base_filename = f"{input_basename}_audio_{timestamp}_gen{gen_index+1}_seed{this_seed}"
        if not add_filename_suffix:
            base_filename = f"{input_basename}"

        final_wav_path = os.path.join(output_dir, f"{base_filename}.wav")

        if not quiet: print(f"\n--- Concatenating {len(best_audio_paths)} audio chunks... ---")
        
        # Concatenate using pydub for robustness
        combined = AudioSegment.empty()
        for path in best_audio_paths:
            try:
                segment = AudioSegment.from_wav(path)
                combined += segment
            except Exception as e:
                if not quiet: print(f"[ERROR] Failed to load/append chunk {path}: {e}")
        
        combined.export(final_wav_path, format="wav")
        if not quiet: print(f"✅ Combined WAV saved to: {final_wav_path}")

        original_wav_for_export = final_wav_path
        if use_auto_editor:
            if not quiet: print("--- Applying Auto-Editor ---")
            edited_wav_path = final_wav_path.replace(".wav", "_edited.wav")
            command = [
                "auto-editor", final_wav_path,
                "--no-open",
                "--silent-threshold", str(ae_threshold),
                "--frame-margin", str(ae_margin),
                "-o", edited_wav_path
            ]
            subprocess.run(command, check=True, capture_output=True)
            if not keep_original_wav:
                os.replace(edited_wav_path, final_wav_path)
                if not quiet: print(f"✅ Auto-edited file replaced original: {final_wav_path}")
            else:
                final_wav_path = edited_wav_path
                if not quiet: print(f"✅ Auto-edited file saved to: {final_wav_path}")
                # Keep original, so we save it with a suffix
                os.rename(original_wav_for_export, original_wav_for_export.replace(".wav", "_original.wav"))

        if use_pyrnnoise:
            if not quiet: print("--- Applying RNNoise Denoising ---")
            _apply_pyrnnoise_in_place(final_wav_path, quiet=quiet)

        if normalize_audio:
            if not quiet: print(f"--- Applying {normalize_method.upper()} Normalization ---")
            norm_wav_path = final_wav_path.replace(".wav", "_norm.wav")
            normalize_with_ffmpeg(final_wav_path, norm_wav_path, normalize_method, normalize_level, normalize_tp, normalize_lra)
            if not quiet: print(f"✅ Normalized audio saved to: {final_wav_path}")

        # --- Export to other formats ---
        generation_output_files = [final_wav_path]
        if export_formats:
            if not quiet: print(f"--- Exporting to: {', '.join(export_formats)} ---")
            for fmt in export_formats:
                if fmt.lower() == "wav": continue
                export_path = os.path.join(output_dir, f"{base_filename}.{fmt.lower()}")
                try:
                    AudioSegment.from_wav(final_wav_path).export(export_path, format=fmt.lower())
                    generation_output_files.append(export_path)
                    if not quiet: print(f"✅ Exported to {fmt.upper()}: {export_path}")
                except Exception as e:
                    if not quiet: print(f"[ERROR] Failed to export to {fmt.upper()}: {e}")
        
        all_output_files.extend(generation_output_files)

        # --- Save settings for this generation ---
        settings_data = {
            "generation_time": datetime.datetime.now().isoformat(),
            "input_basename": input_basename,
            "text": text,
            "audio_prompt_path_input": audio_prompt_path_input,
            "exaggeration_input": exaggeration_input,
            "temperature_input": temperature_input,
            "seed_num_input": this_seed,
            "cfgw_input": cfgw_input,
            "use_pyrnnoise": use_pyrnnoise,
            "use_auto_editor": use_auto_editor,
            "ae_threshold": ae_threshold,
            "ae_margin": ae_margin,
            "export_formats": export_formats,
            "enable_batching": enable_batching,
            "to_lowercase": to_lowercase,
            "normalize_spacing": normalize_spacing,
            "fix_dot_letters": fix_dot_letters,
            "remove_reference_numbers": remove_reference_numbers,
            "keep_original_wav": keep_original_wav,
            "smart_batch_short_sentences": smart_batch_short_sentences,
            "disable_watermark": disable_watermark,
            "num_generations": num_generations,
            "normalize_audio": normalize_audio,
            "normalize_method": normalize_method,
            "normalize_level": normalize_level,
            "normalize_tp": normalize_tp,
            "normalize_lra": normalize_lra,
            "num_candidates_per_chunk": num_candidates_per_chunk,
            "max_attempts_per_candidate": max_attempts_per_candidate,
            "bypass_whisper_checking": bypass_whisper_checking,
            "whisper_model_name": whisper_model_name,
            "enable_parallel": enable_parallel,
            "num_parallel_workers": num_parallel_workers,
            "use_longest_transcript_on_fail": use_longest_transcript_on_fail,
            "sound_words_field": sound_words_field,
            "use_faster_whisper": use_faster_whisper,
            "output_audio_files": generation_output_files,
        }
        
        csv_path = os.path.join(output_dir, f"{base_filename}.settings.csv")
        json_path = os.path.join(output_dir, f"{base_filename}.settings.json")
        save_settings_csv(settings_data, generation_output_files, csv_path)
        save_settings_json(settings_data, json_path)

    # --- Cleanup ---
    if os.path.exists("temp"):
        for f in os.listdir("temp"):
            try:
                os.remove(os.path.join("temp", f))
            except OSError:
                pass # Ignore if file is already gone
        # os.rmdir("temp") # Might fail if another process is fast

    if not quiet: print("\n\n✅ Batch generation complete.")
    return all_output_files

# =====================================================================================
# Gradio UI components and layout
# =====================================================================================

whisper_model_map = {
    "tiny (~1 GB VRAM OpenAI / ~0.5 GB faster-whisper)": "tiny",
    "base (~1.2–2 GB OpenAI / ~0.7–1 GB faster-whisper)": "base",
    "small (~2–3 GB OpenAI / ~1.2–1.7 GB faster-whisper)": "small",
    "medium (~5–8 GB OpenAI / ~2.5–4.5 GB faster-whisper)": "medium",
    "large (~10–13 GB OpenAI / ~4.5–6.5 GB faster-whisper)": "large"
}

def create_ui(defaults):
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue")) as demo:
        gr.Markdown("# 🗣️ Chatterbox TTS Extended")
        gr.Markdown("A user-friendly interface for the Chatterbox TTS model, with extended features for batch processing, audio cleanup, and validation.")

        with gr.Tab("Text-to-Speech"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Inputs
                    text_input = gr.Textbox(label="Text Input", lines=10, placeholder="Enter text here...", value=defaults["text_input"])
                    text_file_input = gr.File(label="Or Upload Text File(s)", file_count="multiple", type="filepath")
                    separate_files_checkbox = gr.Checkbox(label="Generate separate audio file for each uploaded text file", value=defaults["separate_files_checkbox"])
                    audio_prompt_input = gr.Audio(label="Audio Prompt (Optional)", type="filepath")
                    
                    with gr.Accordion("⚙️ Generation Settings", open=True):
                        with gr.Row():
                            num_generations_input = gr.Number(label="Number of Generations", value=defaults["num_generations_input"], step=1, minimum=1, precision=0)
                            seed_input = gr.Number(label="Seed (0 for random)", value=defaults["seed_input"], precision=0)
                        with gr.Row():
                            temp_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label="Temperature", value=defaults["temp_slider"])
                            cfg_weight_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, label="CFG Weight", value=defaults["cfg_weight_slider"])
                            exaggeration_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label="Exaggeration", value=defaults["exaggeration_slider"])
                        disable_watermark_checkbox = gr.Checkbox(label="Disable Audio Watermark", value=defaults["disable_watermark_checkbox"])

                    with gr.Accordion("✍️ Text Processing", open=False):
                        with gr.Row():
                            to_lowercase_checkbox = gr.Checkbox(label="To Lowercase", value=defaults["to_lowercase_checkbox"])
                            normalize_spacing_checkbox = gr.Checkbox(label="Normalize Spacing", value=defaults["normalize_spacing_checkbox"])
                            fix_dot_letters_checkbox = gr.Checkbox(label="Fix 'A.B.C.' -> 'A B C'", value=defaults["fix_dot_letters_checkbox"])
                            remove_reference_numbers_checkbox = gr.Checkbox(label="Remove Ref Numbers [1]", value=defaults["remove_reference_numbers_checkbox"])
                        sound_words_field = gr.Textbox(label="Sound Words to Remove/Replace", lines=3, placeholder="e.g., um\nooh=>oh", value=defaults["sound_words_field"])

                    with gr.Accordion("📦 Batching & Chunking", open=False):
                        enable_batching_checkbox = gr.Checkbox(label="Enable Sentence Grouping into Chunks", value=defaults["enable_batching_checkbox"])
                        smart_batch_short_sentences_checkbox = gr.Checkbox(label="Intelligently Group Short Sentences", value=defaults["smart_batch_short_sentences_checkbox"])

                with gr.Column(scale=1):
                    # Outputs & Controls
                    generate_button = gr.Button("Generate Audio", variant="primary")
                    output_files_box = gr.Files(label="Generated Audio Files", interactive=False)
                    
                    with gr.Row():
                        output_audio_player = gr.Audio(label="Preview Generated Audio", type="filepath", interactive=True)
                        output_audio_dropdown = gr.Dropdown(label="Select Audio to Preview", choices=[], interactive=True)

                    with gr.Accordion("🔬 Whisper Validation", open=True):
                        bypass_whisper_checkbox = gr.Checkbox(label="Bypass Whisper Validation", value=defaults["bypass_whisper_checkbox"])
                        whisper_model_dropdown = gr.Dropdown(label="Whisper Model", choices=list(whisper_model_map.keys()), value=defaults["whisper_model_dropdown"])
                        use_faster_whisper_checkbox = gr.Checkbox(label="Use faster-whisper (recommended)", value=defaults["use_faster_whisper_checkbox"])
                        num_candidates_slider = gr.Slider(minimum=1, maximum=10, step=1, label="Candidates per Chunk", value=defaults["num_candidates_slider"])
                        max_attempts_slider = gr.Slider(minimum=1, maximum=5, step=1, label="Max Attempts per Candidate", value=defaults["max_attempts_slider"])
                        use_longest_transcript_on_fail_checkbox = gr.Checkbox(label="On Fail, Use Longest Transcript (vs. Best Score)", value=defaults["use_longest_transcript_on_fail_checkbox"])

                    with gr.Accordion("🧹 Post-processing", open=False):
                        export_format_checkboxes = gr.CheckboxGroup(label="Export Formats", choices=["wav", "mp3", "flac"], value=defaults["export_format_checkboxes"])
                        
                        with gr.Group():
                            use_pyrnnoise_checkbox = gr.Checkbox(label="Enable RNNoise Denoising", value=defaults["use_pyrnnoise_checkbox"], visible=_PYRNNOISE_AVAILABLE)
                        
                        with gr.Group():
                            use_auto_editor_checkbox = gr.Checkbox(label="Enable Auto-Editor (Remove Silences)", value=defaults["use_auto_editor_checkbox"])
                            threshold_slider = gr.Slider(minimum=0.01, maximum=0.2, step=0.01, label="AE Silence Threshold (lower=more sensitive)", value=defaults["threshold_slider"])
                            margin_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label="AE Margin Around Cuts (seconds)", value=defaults["margin_slider"])
                            keep_original_checkbox = gr.Checkbox(label="Keep Original WAV when using Auto-Editor", value=defaults["keep_original_checkbox"])

                        with gr.Group():
                            normalize_audio_checkbox = gr.Checkbox(label="Enable FFmpeg Audio Normalization", value=defaults["normalize_audio_checkbox"])
                            normalize_method_dropdown = gr.Dropdown(label="Normalization Method", choices=["ebu", "peak"], value=defaults["normalize_method_dropdown"])
                            normalize_level_slider = gr.Slider(minimum=-30, maximum=-10, step=1, label="Target Loudness (LUFS for EBU)", value=defaults["normalize_level_slider"])
                            normalize_tp_slider = gr.Slider(minimum=-3, maximum=0, step=0.5, label="True Peak (for EBU)", value=defaults["normalize_tp_slider"])
                            normalize_lra_slider = gr.Slider(minimum=1, maximum=20, step=1, label="Loudness Range (for EBU)", value=defaults["normalize_lra_slider"])

                    with gr.Accordion("🚀 Parallelism", open=False):
                        enable_parallel_checkbox = gr.Checkbox(label="Enable Parallel Processing", value=defaults["enable_parallel_checkbox"])
                        num_parallel_workers_slider = gr.Slider(minimum=1, maximum=os.cpu_count() or 1, step=1, label="Number of Parallel Workers", value=defaults["num_parallel_workers_slider"])

        with gr.Tab("Voice Conversion"):
            gr.Markdown("Convert one voice to another. Upload a source audio containing the speech you want to convert, and a target audio containing the voice you want to use.")
            with gr.Row():
                with gr.Column():
                    vc_input_audio = gr.Audio(label="Source Audio (to be converted)", type="filepath")
                    vc_target_audio = gr.Audio(label="Target Voice (the voice to use)", type="filepath")
                    vc_pitch_shift = gr.Slider(minimum=-12, maximum=12, step=1, label="Pitch Shift (semitones)", value=0)
                    vc_disable_watermark = gr.Checkbox(label="Disable Audio Watermark", value=True)
                    vc_button = gr.Button("Convert Voice", variant="primary")
                with gr.Column():
                    vc_output_audio = gr.Audio(label="Converted Audio", type="filepath")

        with gr.Tab("Settings"):
            gr.Markdown("Manage application settings. Changes are saved automatically.")
            # Create a list of all UI components that should be saved
            ui_components = {
                "text_input": text_input, "separate_files_checkbox": separate_files_checkbox,
                "export_format_checkboxes": export_format_checkboxes, "disable_watermark_checkbox": disable_watermark_checkbox,
                "num_generations_input": num_generations_input, "num_candidates_slider": num_candidates_slider,
                "max_attempts_slider": max_attempts_slider, "bypass_whisper_checkbox": bypass_whisper_checkbox,
                "whisper_model_dropdown": whisper_model_dropdown, "use_faster_whisper_checkbox": use_faster_whisper_checkbox,
                "enable_parallel_checkbox": enable_parallel_checkbox, "use_longest_transcript_on_fail_checkbox": use_longest_transcript_on_fail_checkbox,
                "num_parallel_workers_slider": num_parallel_workers_slider, "exaggeration_slider": exaggeration_slider,
                "cfg_weight_slider": cfg_weight_slider, "temp_slider": temp_slider, "seed_input": seed_input,
                "enable_batching_checkbox": enable_batching_checkbox, "smart_batch_short_sentences_checkbox": smart_batch_short_sentences_checkbox,
                "to_lowercase_checkbox": to_lowercase_checkbox, "normalize_spacing_checkbox": normalize_spacing_checkbox,
                "fix_dot_letters_checkbox": fix_dot_letters_checkbox, "remove_reference_numbers_checkbox": remove_reference_numbers_checkbox,
                "use_auto_editor_checkbox": use_auto_editor_checkbox, "keep_original_checkbox": keep_original_checkbox,
                "threshold_slider": threshold_slider, "margin_slider": margin_slider,
                "normalize_audio_checkbox": normalize_audio_checkbox, "normalize_method_dropdown": normalize_method_dropdown,
                "normalize_level_slider": normalize_level_slider, "normalize_tp_slider": normalize_tp_slider,
                "normalize_lra_slider": normalize_lra_slider, "sound_words_field": sound_words_field,
                "use_pyrnnoise_checkbox": use_pyrnnoise_checkbox
            }
            
            # Use a loop to attach the save_settings function to each component's change event
            for key, component in ui_components.items():
                # Use a lambda to capture the current state of all components
                component.change(
                    fn=lambda *values: save_settings({k: v for k, v in zip(ui_components.keys(), values)}),
                    inputs=list(ui_components.values()),
                    outputs=[]
                )

        # --- Event Handlers ---
        
        # TTS Generation
        tts_inputs = [
            text_input, text_file_input, audio_prompt_input,
            exaggeration_slider, temp_slider, seed_input, cfg_weight_slider,
            use_pyrnnoise_checkbox,
            use_auto_editor_checkbox, threshold_slider, margin_slider,
            export_format_checkboxes, enable_batching_checkbox,
            to_lowercase_checkbox, normalize_spacing_checkbox, fix_dot_letters_checkbox,
            remove_reference_numbers_checkbox, keep_original_checkbox,
            smart_batch_short_sentences_checkbox, disable_watermark_checkbox,
            num_generations_input, normalize_audio_checkbox, normalize_method_dropdown,
            normalize_level_slider, normalize_tp_slider, normalize_lra_slider,
            num_candidates_slider, max_attempts_slider, bypass_whisper_checkbox,
            whisper_model_dropdown, enable_parallel_checkbox, num_parallel_workers_slider,
            use_longest_transcript_on_fail_checkbox, sound_words_field, use_faster_whisper_checkbox,
            separate_files_checkbox
        ]
        tts_outputs = [output_files_box, output_audio_dropdown, output_audio_player]
        
        generate_button.click(
            fn=generate_and_preview,
            inputs=tts_inputs,
            outputs=tts_outputs
        )
        
        output_audio_dropdown.change(
            fn=update_audio_preview,
            inputs=[output_audio_dropdown],
            outputs=[output_audio_player]
        )

        # Voice Conversion
        vc_button.click(
            fn=voice_conversion,
            inputs=[vc_input_audio, vc_target_audio, vc_disable_watermark, vc_pitch_shift],
            outputs=[vc_output_audio]
        )

    return demo

if __name__ == "__main__":
    # Load settings from file, or use defaults
    settings = load_settings()
    
    # Create the UI with the loaded settings
    demo = create_ui(settings)
    
    # Launch the Gradio app
    demo.launch()
