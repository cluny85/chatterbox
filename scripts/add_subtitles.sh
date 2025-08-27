#!/bin/bash

# --- Script para añadir subtítulos a un video con estilo y fuente totalmente personalizables ---
# Versión Definitiva y Robusta: Usa un array para evitar errores de expansión del shell.

# --- 1. Verificación de Parámetros ---
if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then
    echo "Error: Número de argumentos incorrecto."
    echo "Uso: $0 \"video\" \"srt\" [\"fuente\"] [\"color_primario\"] [\"color_borde\"]"
    exit 1
fi

VIDEO_IN="$1"
SRT_FILE="$2"

# --- 2. Asignación de Parámetros Opcionales con Valores por Defecto ---
FONT_NAME="Ubuntu"
PRIMARY_COLOUR="&H00FFFF&"  # Amarillo
OUTLINE_COLOUR="&H000000&"  # Negro

if [ "$#" -ge 3 ]; then
    FONT_NAME="$3"
fi
if [ "$#" -ge 4 ]; then
    PRIMARY_COLOUR="$4"
fi
if [ "$#" -eq 5 ]; then
    OUTLINE_COLOUR="$5"
fi

if [ ! -f "$VIDEO_IN" ]; then echo "Error: El archivo de video no existe: $VIDEO_IN"; exit 1; fi
if [ ! -f "$SRT_FILE" ]; then echo "Error: El archivo de subtítulos no existe: $SRT_FILE"; exit 1; fi

# --- 3. Definición del Nombre de Salida ---
VIDEO_OUT="${VIDEO_IN%.*}_sub.mp4"

# --- 4. Ejecución del Comando FFmpeg (Método de Array a Prueba de Balas) ---
echo "--- Configuración del Proceso ---"
echo "Video Entrada: $VIDEO_IN"
echo "Subtítulos:    $SRT_FILE"
echo "Video Salida:  $VIDEO_OUT"
echo "Fuente:        $FONT_NAME"
echo "Color Texto:   $PRIMARY_COLOUR"
echo "Color Borde:   $OUTLINE_COLOUR"
echo "---------------------------------"
echo "Iniciando proceso de FFmpeg..."

# *** SECCIÓN CRÍTICA CORREGIDA ***
# 1. Construimos la cadena completa del filtro -vf en una sola variable.
#    Las comillas simples internas se encargarán de que los nombres con espacios se traten como una sola unidad.
FILTER_STRING="subtitles='$SRT_FILE':force_style='FontName=$FONT_NAME,PrimaryColour=$PRIMARY_COLOUR,OutlineColour=$OUTLINE_COLOUR,Outline=2'"

# 2. Creamos un array con todos los argumentos para ffmpeg.
#    Cada elemento del array es un argumento separado. Esto evita CUALQUIER error de comillas o espacios.
FFMPEG_ARGS=(
    -i "$VIDEO_IN"
    -vf "$FILTER_STRING"   # El filtro complejo es un único argumento.
    -c:a copy
    "$VIDEO_OUT"
)

# 3. Ejecutamos el comando pasando el array de forma segura.
ffmpeg "${FFMPEG_ARGS[@]}"


# --- 5. Verificación y Mensaje Final ---
if [ $? -eq 0 ]; then
    echo "¡Éxito! El video con subtítulos ha sido guardado en: $VIDEO_OUT"
else
    echo "Error: FFmpeg encontró un problema. Revisa los nombres de archivo, fuente y los códigos de color."
fi