# Procesador de Subtítulos de Video (`process_video.py`)

Este es un script de Python robusto diseñado para incrustar subtítulos en un archivo de video utilizando FFmpeg. El script se controla a través de la línea de comandos y devuelve siempre un objeto JSON para indicar el resultado de la operación, haciéndolo ideal para flujos de trabajo automatizados.

Permite personalizar completamente el estilo de los subtítulos, incluyendo la fuente, colores (en formato hexadecimal estándar), alineación vertical y horizontal, y tamaño.

## Requisitos

*   Python 3.x y un entorno Conda con las dependencias necesarias.
*   FFmpeg instalado y accesible en el `PATH` del sistema.

## Uso

### Sintaxis

```bash
python process_video.py <ruta_video> <ruta_srt> --font_name <nombre_fuente> [opciones...]
```

### Parámetros

| Argumento                  | Flag                | Descripción                                        | Por Defecto                          |
| -------------------------- | ------------------- | -------------------------------------------------- | ------------------------------------ |
| `video_path`               | (Obligatorio)       | Ruta al archivo de video de entrada.               | N/A                                  |
| `srt_path`                 | (Obligatorio)       | Ruta al archivo de subtítulos `.srt`.              | N/A                                  |
| `--font_name`              | `-fn`               | **(Obligatorio)** El nombre exacto de la fuente a usar.  | N/A                                  |
| `--font_dir`               | `-fd`               | Ruta al directorio que contiene la fuente.         | `~/.local/share/fonts`               |
| `--primary_color`          | `-pc`               | Color del texto en formato hex **RRGGBB**.         | `FFFF00` (Amarillo)                  |
| `--outline_color`          | `-oc`               | Color del borde del texto en formato hex **RRGGBB**. | `000000` (Negro)                     |
| `--vertical_alignment`     | `-va`               | Posición vertical: `down`, `center`, `up`.         | `down`                               |
| `--horizontal_alignment`   | `-ha`               | Justificación horizontal: `left`, `center`, `right`.| `center`                             |
| `--font_size`              | `-fs`               | Tamaño de la fuente en puntos (ej: `48`).          | El tamaño nativo del subtítulo       |

## Ejecución Remota con Conda (SSH)

Para ejecutar el script remotamente dentro de un entorno Conda específico, es necesario forzar un shell interactivo (`bash -i`) para que `conda` esté disponible.

### Ejemplo Completo

```bash
JSON_RESULT=$(ssh -i <ruta_clave> -t <user@host> "
bash -i -c '
conda activate <entorno_conda> && \\
python /ruta/al/script/process_video.py \\
  \"/ruta/remota/video.mp4\" \\
  \"/ruta/remota/subs.srt\" \\
  --font_name \"Varsity Team\" \\
  --primary_color \"FFFFFF\" \\
  --outline_color \"000080\" \\
  --vertical_alignment \"up\"
'
")

echo "$JSON_RESULT"
```

