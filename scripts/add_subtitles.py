import os
import sys
import json
import subprocess
import argparse

def hex_to_ass_color(hex_color: str) -> str:
    """
    Convierte un color hexadecimal de 6 dígitos RRGGBB al formato ASS &HBBGGRR&.
    Ejemplo: "FFFFFF" -> "&HFFFFFF&", "FFFF00" -> "&H00FFFF&"
    """
    if len(hex_color) != 6:
        raise ValueError(f"El color hexadecimal '{hex_color}' debe tener 6 caracteres.")
    try:
        # Valida que sea un número hexadecimal válido
        int(hex_color, 16)
    except ValueError:
        raise ValueError(f"'{hex_color}' no es un código hexadecimal válido.")
    
    rr = hex_color[0:2].upper()
    gg = hex_color[2:4].upper()
    bb = hex_color[4:6].upper()
    return f"&H{bb}{gg}{rr}&"

def main():
    """
    Incrusta subtítulos en un video con un control de estilo robusto y profesional,
    aceptando colores en formato hexadecimal RRGGBB estándar.
    """
    default_font_dir = os.path.expanduser("~/.local/share/fonts")
    
    parser = argparse.ArgumentParser(
        description="Incrusta subtítulos con estilos personalizables de forma robusta.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- ARGUMENTOS DE LÍNEA DE COMANDOS ---
    parser.add_argument("video_path", help="Ruta al archivo de video de entrada.")
    parser.add_argument("srt_path", help="Ruta al archivo de subtítulos (.srt).")
    
    parser.add_argument("-fn", "--font_name", required=True, help="El NOMBRE EXACTO de la fuente (ej: 'Varsity Team').")
    parser.add_argument("-fd", "--font_dir", default=default_font_dir, help=f"Ruta al directorio de la fuente.\n(Defecto: {default_font_dir})")
    parser.add_argument("-pc", "--primary_color", default="FFFF00", help="Color del texto en formato hex RRGGBB.\n(Defecto: FFFF00 - Amarillo)")
    parser.add_argument("-oc", "--outline_color", default="000000", help="Color del borde en formato hex RRGGBB.\n(Defecto: 000000 - Negro)")
    parser.add_argument("-va", "--vertical_alignment", default="down", choices=["down", "center", "up"], help="Posición vertical de los subtítulos.")
    parser.add_argument("-ha", "--horizontal_alignment", default="center", choices=["left", "center", "right"], help="Justificación horizontal de los subtítulos.")
    parser.add_argument("-fs", "--font_size", type=int, default=None, help="Tamaño de la fuente en puntos.")
    
    args = parser.parse_args()

    # --- VALIDACIÓN DE RUTAS ---
    if not os.path.isdir(args.font_dir):
        error_response = { "status": "error", "message": "El directorio de fuentes no fue encontrado.", "path_checked": args.font_dir }
        print(json.dumps(error_response, indent=2))
        sys.exit(1)

    for path in [args.video_path, args.srt_path]:
        if not os.path.isfile(path):
            error_response = { "status": "error", "message": "El archivo de entrada no fue encontrado.", "path": path }
            print(json.dumps(error_response, indent=2))
            sys.exit(1)
            
    # --- CONSTRUCCIÓN DEL COMANDO ---
    output_path = os.path.splitext(args.video_path)[0] + "_sub.mp4"

    try:
        # Conversión y validación de colores
        ass_primary_color = hex_to_ass_color(args.primary_color)
        ass_outline_color = hex_to_ass_color(args.outline_color)
    except ValueError as e:
        error_response = { "status": "error", "message": "Formato de color hexadecimal inválido.", "details": str(e) }
        print(json.dumps(error_response, indent=2))
        sys.exit(1)

    # Lógica de alineación Numpad
    alignment_map = {
        "up":     {"left": 4, "center": 6, "right": 7},
        "center": {"left": 11, "center": 7, "right": 9},
        "down":   {"left": 1, "center": 2, "right": 3}
    }
    alignment_value = alignment_map[args.vertical_alignment][args.horizontal_alignment]
# 7: arriba derecha
# 8: centro izquierda
# 9: centro izquierda
# 5: arriba izquierda
# 4: arriba izquierda
# 6: arriba centro
# 3: abajo derecha
# 1: abajo izquierda
# 2: centro centro

    # Construcción segura de la cadena de estilo
    style_components = [
        f"FontName='{args.font_name}'",
        f"PrimaryColour={ass_primary_color}",
        f"OutlineColour={ass_outline_color}",
        f"Alignment={alignment_value}",
        "Outline=1"
    ]
    if args.font_size is not None:
        style_components.append(f"FontSize={args.font_size}")
    
    style_string = ",".join(style_components)

    filter_string = (
        f"subtitles='{args.srt_path}':"
        f"fontsdir='{args.font_dir}':"
        f"force_style='{style_string}'"
    )

    ffmpeg_command = ["ffmpeg", "-y", "-i", args.video_path, "-vf", filter_string, "-c:a", "copy", output_path]
    
    # --- EJECUCIÓN ---
    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        success_response = { "status": "success", "message": "Video procesado exitosamente.", "data": { "output_video": output_path } }
        print(json.dumps(success_response, indent=2))
    except subprocess.CalledProcessError as e:
        error_response = {
            "status": "error", "message": "FFmpeg falló.",
            "data": { "command_executed": " ".join(e.cmd), "ffmpeg_stderr": e.stderr.strip() }
        }
        print(json.dumps(error_response, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()