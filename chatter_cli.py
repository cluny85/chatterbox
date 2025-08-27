import argparse
import os
import sys
import json
import traceback
from chatter_logic import process_text_for_tts, default_settings, whisper_model_map

def exit_with_error_json(message, details=None):
    """Prints a JSON error message to stdout and exits."""
    error_response = {
        "status": "error",
        "message": message,
    }
    if details:
        error_response["details"] = details
    print(json.dumps(error_response, indent=2))
    sys.exit(1)

def main():
    """
    CLI for Chatterbox TTS.
    """
    parser = argparse.ArgumentParser(description="Generate audio from text using Chatterbox TTS.")

    # Get default settings
    defaults = default_settings()

    # --- Input/Output Arguments ---
    parser.add_argument("--text", type=str, help="Text to synthesize. If not provided, reads from stdin.")
    parser.add_argument("--text-file", type=str, help="Path to a text file to synthesize.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save the output audio files.")
    parser.add_argument("--audio-prompt", type=str, default=None, help="Path to an audio prompt file.")
    parser.add_argument("--basename", type=str, default=None, help="Base name for the output file(s). If not provided, a name is automatically generated with timestamp and seed.")
    parser.add_argument("--no-suffix", action="store_true", help="Do not add timestamp/seed suffix to the filename when using --basename.")
    parser.add_argument("--quiet", action="store_true", help="Suppress all non-JSON output (logs, progress, etc.).")

    # --- Generation Parameters ---
    parser.add_argument("--num-generations", type=int, default=defaults["num_generations_input"], help="Number of different audio files to generate.")
    parser.add_argument("--seed", type=int, default=defaults["seed_input"], help="Seed for random number generation. 0 for random.")
    parser.add_argument("--temperature", type=float, default=defaults["temp_slider"], help="Generation temperature.")
    parser.add_argument("--cfg-weight", type=float, default=defaults["cfg_weight_slider"], help="CFG weight.")
    parser.add_argument("--exaggeration", type=float, default=defaults["exaggeration_slider"], help="Exaggeration factor.")
    parser.add_argument("--disable-watermark", action="store_true", default=defaults["disable_watermark_checkbox"], help="Disable the audio watermark.")

    # --- Text Processing ---
    parser.add_argument("--no-lowercase", action="store_false", dest="to_lowercase", default=defaults["to_lowercase_checkbox"], help="Disable converting text to lowercase.")
    parser.add_argument("--no-normalize-spacing", action="store_false", dest="normalize_spacing", default=defaults["normalize_spacing_checkbox"], help="Disable normalizing whitespace.")
    parser.add_argument("--no-fix-dot-letters", action="store_false", dest="fix_dot_letters", default=defaults["fix_dot_letters_checkbox"], help="Disable fixing sequences like 'A.B.C.' to 'A B C'.")
    parser.add_argument("--no-remove-reference-numbers", action="store_false", dest="remove_reference_numbers", default=defaults["remove_reference_numbers_checkbox"], help="Disable removing bracketed reference numbers.")
    parser.add_argument("--sound-words", type=str, default=defaults["sound_words_field"], help="Sound words to remove or replace, separated by newlines. e.g., 'ooh=>oh'")

    # --- Batching ---
    parser.add_argument("--enable-batching", action="store_true", default=defaults["enable_batching_checkbox"], help="Enable sentence grouping into chunks.")
    parser.add_argument("--smart-batch-short-sentences", action="store_true", default=defaults["smart_batch_short_sentences_checkbox"], help="Intelligently group short sentences together.")

    # --- Whisper Validation ---
    parser.add_argument("--bypass-whisper", action="store_true", default=defaults["bypass_whisper_checkbox"], help="Bypass Whisper validation.")
    parser.add_argument("--whisper-model", type=str, default="medium", choices=list(whisper_model_map.values()), help="Whisper model to use for validation.")
    parser.add_argument("--use-faster-whisper", action="store_true", default=defaults["use_faster_whisper_checkbox"], help="Use faster-whisper implementation.")
    parser.add_argument("--num-candidates", type=int, default=defaults["num_candidates_slider"], help="Number of candidates to generate per chunk.")
    parser.add_argument("--max-attempts", type=int, default=defaults["max_attempts_slider"], help="Max attempts per candidate if Whisper validation fails.")
    parser.add_argument("--use-longest-transcript-on-fail", action="store_true", default=defaults["use_longest_transcript_on_fail_checkbox"], help="If all candidates fail, use the one with the longest transcript.")

    # --- Post-processing ---
    parser.add_argument("--export-formats", nargs="+", default=defaults["export_format_checkboxes"], help="List of formats to export (e.g., mp3 flac wav).")
    parser.add_argument("--use-pyrnnoise", action="store_true", default=defaults["use_pyrnnoise_checkbox"], help="Enable RNNoise denoising.")
    parser.add_argument("--use-auto-editor", action="store_true", default=defaults["use_auto_editor_checkbox"], help="Enable auto-editor to remove silences.")
    parser.add_argument("--ae-threshold", type=float, default=defaults["threshold_slider"], help="Auto-editor silence threshold (dB).")
    parser.add_argument("--ae-margin", type=float, default=defaults["margin_slider"], help="Auto-editor margin around cuts (seconds).")
    parser.add_argument("--keep-original-wav", action="store_true", default=defaults["keep_original_checkbox"], help="Keep the original WAV file when using auto-editor.")
    parser.add_argument("--normalize-audio", action="store_true", default=defaults["normalize_audio_checkbox"], help="Enable ffmpeg audio normalization.")
    parser.add_argument("--normalize-method", type=str, default=defaults["normalize_method_dropdown"], choices=["ebu", "peak"], help="Normalization method.")
    parser.add_argument("--normalize-level", type=float, default=defaults["normalize_level_slider"], help="Target loudness in LUFS for EBU normalization.")
    parser.add_argument("--normalize-tp", type=float, default=defaults["normalize_tp_slider"], help="True Peak limit for EBU normalization.")
    parser.add_argument("--normalize-lra", type=float, default=defaults["normalize_lra_slider"], help="Loudness Range for EBU normalization.")

    # --- Parallelism ---
    parser.add_argument("--disable-parallel", action="store_false", dest="enable_parallel", default=defaults["enable_parallel_checkbox"], help="Disable parallel processing.")
    parser.add_argument("--num-workers", type=int, default=defaults["num_parallel_workers_slider"], help="Number of parallel workers for generation.")

    args = parser.parse_args()

    # --- Suppress stderr if in quiet mode ---
    if args.quiet:
        sys.stderr = open(os.devnull, 'w')

    # --- Get Text Input ---
    text_input = ""
    input_basename = None # Default to None
    if args.text_file:
        if not os.path.exists(args.text_file):
            exit_with_error_json(f"Text file not found at {args.text_file}")
        with open(args.text_file, "r", encoding="utf-8") as f:
            text_input = f.read()
        # Default basename from filename, but can be overridden by the arg
        input_basename = os.path.splitext(os.path.basename(args.text_file))[0]
    elif args.text:
        text_input = args.text
    else:
        if not args.quiet:
            print("Reading text from stdin. Press Ctrl+D (or Ctrl+Z on Windows) to end.", file=sys.stderr)
        text_input = sys.stdin.read()

    # If the user explicitly provides a basename, it takes precedence
    if args.basename:
        input_basename = args.basename
    elif not input_basename: # If no text file was used and no basename was given
        input_basename = "cli_input"


    if not text_input.strip():
        exit_with_error_json("No text provided. Use --text, --text-file, or pipe from stdin.")

    # --- Create output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.quiet:
        print("Starting TTS generation...", file=sys.stderr)
    try:
        output_files = process_text_for_tts(
            text=text_input,
            input_basename=input_basename,
            audio_prompt_path_input=args.audio_prompt,
            exaggeration_input=args.exaggeration,
            temperature_input=args.temperature,
            seed_num_input=args.seed,
            cfgw_input=args.cfg_weight,
            use_pyrnnoise=args.use_pyrnnoise,
            use_auto_editor=args.use_auto_editor,
            ae_threshold=args.ae_threshold,
            ae_margin=args.ae_margin,
            export_formats=args.export_formats,
            enable_batching=args.enable_batching,
            to_lowercase=args.to_lowercase,
            normalize_spacing=args.normalize_spacing,
            fix_dot_letters=args.fix_dot_letters,
            remove_reference_numbers=args.remove_reference_numbers,
            keep_original_wav=args.keep_original_wav,
            smart_batch_short_sentences=args.smart_batch_short_sentences,
            disable_watermark=args.disable_watermark,
            num_generations=args.num_generations,
            normalize_audio=args.normalize_audio,
            normalize_method=args.normalize_method,
            normalize_level=args.normalize_level,
            normalize_tp=args.normalize_tp,
            normalize_lra=args.normalize_lra,
            num_candidates_per_chunk=args.num_candidates,
            max_attempts_per_candidate=args.max_attempts,
            bypass_whisper_checking=args.bypass_whisper,
            whisper_model_name=args.whisper_model,
            enable_parallel=args.enable_parallel,
            num_parallel_workers=args.num_workers,
            use_longest_transcript_on_fail=args.use_longest_transcript_on_fail,
            sound_words_field=args.sound_words,
            use_faster_whisper=args.use_faster_whisper,
            output_dir=args.output_dir,
            add_filename_suffix=(not args.no_suffix), # Pass the new flag
            quiet=args.quiet
        )
        if not args.quiet:
            print("\nGeneration complete.", file=sys.stderr)
        
        mp3_files = [os.path.basename(f) for f in output_files if f.endswith(".mp3")]
        
        result = {
            "status": "success",
            "path": os.path.abspath(args.output_dir),
            "files": mp3_files
        }
        
        print(json.dumps(result, indent=2))

    except Exception as e:
        tb_str = traceback.format_exc()
        # Log the full error to stderr for debugging purposes
        if not args.quiet:
            print(f"\nAn error occurred during generation: {e}", file=sys.stderr)
            print(tb_str, file=sys.stderr)
        # Exit with a clean JSON error message to stdout
        exit_with_error_json(f"An error occurred during generation: {str(e)}", details=tb_str)

if __name__ == "__main__":
    main()
