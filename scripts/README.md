# Incrustador de Subtítulos con FFmpeg (`add_subtitles.sh`)

Este script de Bash utiliza **FFmpeg** para incrustar de forma permanente (hardcode) subtítulos en un archivo de video. Permite personalizar la fuente, el color del texto y el color del borde a través de parámetros opcionales en la línea de comandos.

## Requisitos

Es necesario tener **`FFmpeg`** instalado en el sistema donde se ejecutará el script.

## Instalación

1.  Copia el script `add_subtitles.sh` a tu servidor (por ejemplo, en el directorio home `~/`).
2.  Otorga permisos de ejecución al script:
    ```bash
    chmod +x add_subtitles.sh
    ```

## Uso

### Sintaxis

El script acepta de 2 a 5 argumentos. Los dos primeros son obligatorios.

```bash
./add_subtitles.sh "<ruta_video>" "<ruta_srt>" ["<fuente>"] ["<color_primario>"] ["<color_borde>"]
```

### Parámetros

*   **`ruta_video`** (Obligatorio): La ruta completa al archivo de video de entrada.
*   **`ruta_srt`** (Obligatorio): La ruta completa al archivo de subtítulos `.srt`.
*   **`fuente`** (Opcional): El nombre de la fuente instalada en el sistema que se usará.
    *   *Valor por defecto:* `"Ubuntu"`
*   **`color_primario`** (Opcional): El código de color para el texto de los subtítulos.
    *   *Valor por defecto:* `&H00FFFF&` (Amarillo)
*   **`color_borde`** (Opcional): El código de color para el borde del texto.
    *   *Valor por defecto:* `&H000000&` (Negro)

### Ejemplos de Ejecución

#### Ejemplo 1: Uso Básico (Valores por defecto)

Solo se proporcionan los archivos de video y subtítulos. El script usará la fuente "Ubuntu" con texto amarillo y borde negro.

```bash
./add_subtitles.sh "/media/proyectos/evento.mp4" "/media/proyectos/evento_subs.srt"
```

#### Ejemplo 2: Fuente Personalizada

Se especifica una fuente personalizada ("Varsity Team"). Los colores seguirán siendo los predeterminados (amarillo y negro).

```bash
./add_subtitles.sh "/media/proyectos/evento.mp4" "/media/proyectos/evento_subs.srt" "Varsity Team"
```

#### Ejemplo 3: Personalización Completa

Se especifican todos los parámetros: la fuente "Varsity Team", color de texto blanco (`&HFFFFFF&`) y color de borde azul oscuro (`&H800000&`).

```bash
./add_subtitles.sh "/media/proyectos/evento.mp4" "/media/proyectos/evento_subs.srt" "Varsity Team" "&HFFFFFF&" "&H800000&"
```



## Resultado

El script generará un nuevo archivo de video con el sufijo `_sub.mp4` en el mismo directorio que el video original.

## Notas Importantes

*   **Formato de Color:** El script espera los colores en el formato hexadecimal `ASS` de FFmpeg: `&HBBGGRR&` (Azul-Verde-Rojo), que es el inverso del formato web (`#RRGGBB`).
*   **Nombres con Espacios:** Siempre entrecomilla las rutas de archivo y los nombres de fuentes que contengan espacios para evitar errores.