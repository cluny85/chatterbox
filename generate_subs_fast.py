import os
import sys
import torch
import argparse # Importamos la librería para manejar argumentos
from faster_whisper import WhisperModel
from datetime import timedelta

def format_timestamp(seconds: float) -> str:
    """Convierte segundos a un formato de timestamp SRT: HH:MM:SS,ms"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def generate_srt_file(video_path: str, segments):
    """Genera y guarda el contenido del archivo SRT."""
    srt_filename = os.path.splitext(video_path)[0] + ".srt"
    
    with open(srt_filename, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(segments):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            text = segment.text.strip()
            
            srt_file.write(f"{i + 1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")
            
    return srt_filename

def main():
    # --- 1. Configuración de Argumentos con argparse ---
    parser = argparse.ArgumentParser(
        description="Genera subtítulos (.srt) para un archivo de video usando faster-whisper.",
        formatter_class=argparse.RawTextHelpFormatter # Para un formato de ayuda más limpio
    )
    parser.add_argument(
        "video_path", 
        type=str, 
        help="Ruta al archivo de video que se va a transcribir."
    )
    parser.add_argument(
        "-l", "--language", 
        type=str, 
        default="en", 
        help="Código de idioma de dos letras (ej. 'es', 'fr', 'de').\nPor defecto es 'en' (inglés)."
    )
    args = parser.parse_args()
    
    video_path = args.video_path
    language = args.language
    
    if not os.path.exists(video_path):
        print(f"Error: El archivo '{video_path}' no fue encontrado.")
        sys.exit(1)

    # --- 2. Configuración del Modelo y Dispositivo ---
    model_size = "small"
    
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        print("GPU detectada. Usando 'cuda' con cómputo 'float16'.")
    else:
        device = "cpu"
        compute_type = "float32"
        print("No se detectó GPU. Usando 'cpu' con cómputo 'float32'.")

    # --- 3. Carga del Modelo ---
    print(f"Cargando el modelo whisper '{model_size}'...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print("Modelo cargado exitosamente.")

    # --- 4. Transcripción del Audio ---
    print(f"Iniciando la transcripción de '{os.path.basename(video_path)}' en idioma '{language}'...")
    print("Esto puede tomar un momento...")
    
    segments, info = model.transcribe(video_path, beam_size=5, language=language)
    
    # El modelo puede detectar un idioma diferente al especificado, esto es solo informativo.
    print(f"Idioma detectado por el modelo: '{info.language}' con una probabilidad de {info.language_probability:.2f}")

    # --- 5. Generación del Archivo SRT ---
    srt_path = generate_srt_file(video_path, segments)
    
    print("-" * 50)
    print(f"¡Proceso completado! Subtítulos guardados en:")
    print(srt_path)
    print("-" * 50)


if __name__ == "__main__":
    main()