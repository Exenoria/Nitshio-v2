# Solución de problemas — Nitshio

> Última actualización: 2026-11-11 (Fase 0: Qwen3.5-2B, `think=false` y VRAM).

## Ollama

### `No se pudo conectar con Ollama`

1. Comprueba que Ollama esté corriendo:

   ```powershell
   ollama list
   ollama serve
   ```

2. Revisa `OLLAMA_HOST` en `.env` y `ollama.host` en `config/config.yaml`.
3. Si usas otra máquina, asegúrate de que el puerto `11434` esté accesible.

### `model 'nitshio-qwen2b' not found`

El modelo activo se declara en `config/config.yaml`. Si no existe:

```powershell
ollama list
ollama create nitshio-qwen2b -f models/Modelfile.qwen2b
```

O cambia temporalmente `ollama.models.active` al modelo base (`qwen3.5:2b`).
Nunca cambies el nombre en el código: el proyecto lee el tag del YAML. Para una
versión Phi o un GGUF local, usa `training/build_model.py --gguf` y el
Modelfile generado con `TEMPLATE` explícito.

### Respuestas lentas

- Si la GPU tiene memoria limitada, mantén `num_ctx: 2048`.
- En 4 GB, `qwen3.5:2b` y `qwen3.5:0.8b` no caben completos a la vez. Mueve el
  router a CPU con `ollama.options.router.num_gpu: 0` (comentado en
  `config/config.yaml`) o usa el mismo modelo como router.
- No dejes cargados a la vez el modelo principal y el router si no hace falta.
- Usa `--no-tools` para saltarte el router en pruebas de personalidad.
- `scripts/check_models.py` mide tokens/s, la residencia simultánea de VRAM y
  sugiere alternativas más ligeras.
- La latencia de `training/evaluate.py` no es comparable con los tok/s de
  `check_models.py`: el evaluador usa prompts de ~2000 tokens y respuestas
  largas. Ver la sección de latencia en [TRAINING.md](TRAINING.md).

### El modelo devuelve texto fuera del JSON o una emoción inválida

El parser reintenta y repara. Si aun así falla, usa el fallback neutral e
imprime un aviso claro por `stderr`. En Qwen3.5 revisa que
`ollama.think: false` esté activo: si el modelo mezcla `thinking`/`reasoning`
con el JSON, el cliente los descarta y el parser activa el fallback. Para
diagnosticar:

```powershell
.\.venv\Scripts\python.exe run.py --mode text --once "Hola" --debug
```

Revisa `attempts`, `fallback`, `reparado` y las líneas `[crudo N]` de la salida
de depuración. En `--mode voice --once --debug` también se muestra el crudo.

## Memoria / ChromaDB

### `No se pudo abrir ChromaDB`

- Comprueba permisos sobre `data/chroma`.
- Borra la carpeta solo si aceptas perder recuerdos:
  `Remove-Item -Recurse data\chroma`.
- Asegúrate de que `chromadb` esté instalado:
  `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

### `Fallo de embeddings en Ollama`

- Verifica que `nomic-embed-text:latest` esté descargado.
- Comprueba que Ollama responda en `/api/embed`.
- Si usas una versión antigua de Ollama, el cliente cae a `/api/embeddings`
  automáticamente.

### La memoria no recuerda nada

- La primera conversación solo guarda; la recuperación aparece en turnos
  posteriores.
- `min_score` en `config/config.yaml` puede ser demasiado alto.
- Usa `/memoria <texto>` en el chat para ver el contexto recuperado.

## Voz

### `No existe la voz Piper`

```powershell
.\.venv\Scripts\python.exe scripts\download_voice.py
```

Verifica que existan:

- `data/models/piper/es_MX-claude-high.onnx`
- `data/models/piper/es_MX-claude-high.onnx.json`

### `No se pudo cargar la voz Piper`

- La ruta debe apuntar a un `.onnx`, no a la raíz del proyecto.
- El archivo `.onnx.json` debe estar junto al `.onnx`.
- Revisa `tts.piper.model_path` y `tts.piper.config_path`.

### No se oye audio

1. Ejecuta `run.py --mode diagnose` y copia el nombre exacto del dispositivo.
2. Ajusta `tts.audio.output_device` o usa el índice.
3. Si usas VB-Cable, recuerda: Nitshio escribe en `CABLE Input`; VSeeFace lee
   de `CABLE Output`.
4. Prueba sin VSeeFace con `--no-avatar`.

### El micrófono no captura

- Revisa el dispositivo de entrada en Windows.
- Prueba `python run.py --mode voice` y habla después de que el VAD detecte
  silencio inicial.
- Si `webrtcvad` falla, el segmentador usa el fallback por energía.
- Ajusta `stt.vad.aggressiveness`, `min_speech_ms` y `end_silence_ms`.

### Moonshine no transcribe bien

- El modelo español se descarga la primera vez en `data/models/moonshine`.
- Comprueba `stt.language: es`.
- Habla cerca del micrófono y evita ruido de fondo.
- Como fallback, cambia `stt.provider` a `whisper` si tienes
  `faster-whisper` instalado.

### La prueba integral de voz falla o no detecta audio

`scripts/test_voice_roundtrip.py` inyecta audio Piper en un micrófono falso y
comprueba la cadena completa. Si falla:

1. Ejecuta sin herramientas para descartar el router:
   `.\.venv\Scripts\python.exe scripts\test_voice_roundtrip.py --no-tools --json`
2. Si la transcripción sale vacía, revisa Moonshine y la voz Piper.
3. Si `CABLE Output` da RMS casi cero:
   - comprueba que `tts.audio.output_device` sea `CABLE Input`;
   - cierra otras aplicaciones que estén usando el cable;
   - ejecuta `run.py --mode diagnose` y confirma que el dispositivo existe.
4. Si el TTS no genera chunks, revisa `tts.piper.model_path` y el log
   `logs/nitshio.log`.
5. Si el avatar no recibe la emoción, revisa el puerto VMC y que VSeeFace esté
   abierto. La prueba lo avisa, pero no falla si VSeeFace está cerrado.

## Herramientas

### `search_google` no devuelve resultados

- Sin claves de Google usa DuckDuckGo; si también falla, abre Google en el
  navegador.
- Para usar Google CSE, rellena `GOOGLE_CSE_KEY` y `GOOGLE_CSE_CX` en `.env`.

### `play_song` no reproduce

- Comprueba que `ffplay` esté en el `PATH`:

  ```powershell
  ffplay -version
  ```

- Ajusta `tools.media.player` si el ejecutable tiene otro nombre.
- El límite `max_duration_seconds` evita descargar audio demasiado largo.

### `No se encontró el reproductor`

Instala FFmpeg o cambia `tools.media.player` a un reproductor disponible.

### Borrado cancelado

Es el comportamiento seguro por defecto. `confirm_delete: true` exige
confirmación. Si quieres desactivarlo, hazlo de forma explícita en
`config/config.yaml` y asume el riesgo.

### `Ruta fuera de allowed_paths`

La herramienta solo opera dentro de `tools.allowed_paths`. Añade la carpeta
necesaria al YAML o mueve el archivo a `workspace/`.

### `Symlink no permitido`

Por seguridad, los symlinks se rechazan. Usa una ruta real o activa
`allow_symlinks: true` solo si confías en el origen.

## Avatar / VSeeFace

Consulta [VSEEFACE.md](VSEEFACE.md). Los fallos más comunes son el puerto VMC,
el dispositivo de audio y los nombres de blendshape.

### Se oye el audio pero el avatar no mueve la boca

Este fue el fallo real en este equipo. El audio llegaba a `CABLE Output`, pero
VSeeFace usaba un proveedor de lip-sync que no reaccionaba. Solución:

1. Abre **Configuración General** en VSeeFace.
2. Activa **Lipsync por audio** (`AudioLipSync=1`).
3. Selecciona **`CABLE Output (VB-Audio Virtual Cable)`** como micrófono.
4. Baja a **Opciones avanzadas** y activa
   **“Usar el modelo viejo para lip sync (menor rendimiento)”**
   (`OldLipSyncProvider=1`).
5. Reinicia VSeeFace y verifica en el archivo `settings.ini` de VSeeFace,
   bajo `%USERPROFILE%\AppData\LocalLow\<proveedor>\VSeeFace\`:

   ```ini
   Microphone=CABLE Output (VB-Audio Virtual Cable)
   AudioLipSync=1
   OldLipSyncProvider=1
   VMCReceiveEnable=1
   ```

No envíes visemas `A/I/U/E/O` por VMC a la vez: en las pruebas no movieron la
boca y el lip-sync por audio los sobrescribía.

### El avatar no cambia de emoción

- Comprueba que `VMCReceiveEnable=1` y el puerto `39539`.
- Usa un VRM que declare grupos de blendshapes; un avatar sin esos grupos no
  mostrará las emociones aunque se reciba el mensaje VMC.
- Ejecuta `scripts/test_emotions.py --all` y revisa los valores impresos.

### La captura de VSeeFace sale en blanco

`PrintWindow` devuelve una imagen blanca si la ventana está minimizada.
Restaura VSeeFace antes de capturar. No es un fallo del avatar ni de Nitshio.

## Entrenamiento

### `ModuleNotFoundError: gradio`

La UI necesita Gradio, que está en `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install "gradio>=5.0,<7"
```

Si actualizaste desde Gradio 5, reinicia el proceso: Gradio 6 cambia la API de
`Chatbot` (la UI ya detecta ambos casos).

### `Faltan dependencias de entrenamiento`

`training/finetune_qlora.py` y `training/export_gguf.py` requieren
`requirements-train.txt` y una GPU. En local usa solo `--dry-run`:

```powershell
.\.venv\Scripts\python.exe training\finetune_qlora.py --dataset training\datasets\train.jsonl --dry-run
.\.venv\Scripts\python.exe training\export_gguf.py --adapter-dir training --dry-run
```

El dry-run del combinado da 348 ejemplos → **312 train / 36 val**. El
entrenamiento real se ejecuta en Colab con el notebook
`training/cloud/Nitshio_QLoRA.ipynb`.

### El notebook de Colab no encuentra el proyecto

El notebook usa `NITSHIO_WORKDIR=/content` y `NITSHIO_DIR=/content/Nitshio`.
Si subiste el zip con otro nombre o a otra carpeta:

1. Define `NITSHIO_WORKDIR` antes de ejecutar la primera celda.
2. O define `NITSHIO_DIR` con la ruta exacta que contiene `config/config.yaml`.
3. Comprueba que el zip no anide una carpeta extra; el notebook busca
   `config/config.yaml` en `/content` y en sus subcarpetas directas.

### El dataset tiene errores o duplicados

```powershell
.\.venv\Scripts\python.exe training\validate_dataset.py training\datasets\nitshio_dataset_300.jsonl --strict --json
.\.venv\Scripts\python.exe training\validate_dataset.py training\datasets\train.jsonl --strict --json
```

Revisa `issues`: JSON roto, emoción no declarada, campos vacíos o claves
extra. Los duplicados exactos y casi duplicados se descartan y se cuentan; no
son errores bloqueantes, pero conviene revisar el umbral
`training.synthetic.near_duplicate_threshold`. Al combinar `seed.jsonl` con
`nitshio_dataset_300.jsonl` se descartan 300 duplicados porque la semilla ya
los contiene; el resultado son 348 ejemplos únicos.

### El split queda vacío en validación o prueba

El split garantiza cobertura mínima cuando una emoción tiene 3 o más
ejemplos. Si una emoción tiene menos, añade ejemplos antes de particionar.
Con el dataset combinado de 348 ejemplos el reparto real es **276/36/36**; el
split interno del fine-tune (90/10) es **312/36**.

### `No se encontró convert_hf_to_gguf.py` o `llama-quantize`

`export_gguf.py` necesita `llama.cpp` clonado y compilado:

```powershell
git clone --depth 1 https://github.com/ggerganov/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --target llama-quantize
```

Pasa `--llama-cpp-dir` con la ruta correcta. En el notebook cloud estas celdas
ya están incluidas.

### `--gguf` falla o el `FROM` no encuentra el archivo

`training/build_model.py --gguf` valida que la ruta exista y que la extensión
sea `.gguf`. El `FROM` se escribe absoluto y con barras normales; si la ruta
tiene espacios se entrecomilla. Si Ollama no encuentra el archivo:

1. Ejecuta `--print` para ver el `FROM` exacto:
   `.\.venv\Scripts\python.exe training\build_model.py --version nitshio-v2 --gguf C:\ruta\nitshio-v2.gguf --print`
2. Comprueba que la ruta exista desde la máquina donde corre Ollama (no sirve
   una ruta de Colab).
3. Vuelve a ejecutar sin `--print` para escribir el Modelfile y registrar la
   versión con `type=gguf`.

### `ollama create` falla con `nitshio-v2`

- Comprueba que el GGUF exista y que `models/Modelfile` apunte a él.
- Verifica el tag con `ollama list`.
- Revisa que `models/versions.json` tenga la versión registrada con
  `type=gguf` y `gguf_path`.
- Si el GGUF se descargó de Colab, descomprímelo antes de crear el modelo.
- Usa `training/build_model.py --version nitshio-v2 --gguf <ruta> --create`
  para que el `FROM` y el registro queden sincronizados.

### La evaluación tarda mucho o falla por Ollama

- Usa `--no-judge` para evitar el juez opcional.
- Usa `--limit 5` para una prueba rápida.
- Con `num_ctx: 2048` el prompt de personalidad + few-shots ronda los 2000
  tokens; la evaluación completa es lenta en la T1200. No es un fallo del
  evaluador: es el límite de contexto de la GPU de 4 GB.
- La latencia media de **78.5 s** y **5.83 tok/s** no contradice los
  **15.6–17.5 tok/s** de `check_models.py`: el evaluador usa prompts de ~2029
  tokens y deja crecer la respuesta hasta 512, así que llena el contexto de
  2048 y llama.cpp desplaza la ventana. `check_models.py` usa 1026 tokens de
  entrada, 29 de salida y mide solo decodificación tras un calentamiento.
- Si un prompt agota `request_timeout_seconds` (180 s por defecto), el reporte
  lo registra como error de red y sigue con los demás. En la comparación real
  el modelo base agotó una vez el timeout en `angry-celosa`.
- Confirma que el modelo activo esté descargado (`ollama list`).
- Si los embeddings fallan, el evaluador cae a similitud léxica; revisa
  `nomic-embed-text`.

### La UI no guarda correcciones

- Verifica permisos de escritura en `training/datasets/`.
- La corrección debe pasar la validación: `user`, `response` y `emotion`
  declarada.
- La memoria es opcional: si ChromaDB falla, la corrección igual se guarda en
  `corrections.jsonl`, pero no se recupera por RAG.

## Tests y documentación

### `pytest` falla por audio o VMC

Los tests usan dobles y no deberían abrir dispositivos reales. Si falla algo
del entorno, revisa que estés usando el Python del `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### `check_docs.py` dice que falta documentación

1. Añade la entrada en `docs/CHANGELOG.md` con `### Archivos afectados`.
2. Incluye la ruta, la carpeta (`brain/`) o el nombre del archivo.
3. Actualiza el documento correspondiente.
4. Vuelve a ejecutar:

   ```powershell
   .\.venv\Scripts\python.exe scripts\check_docs.py
   ```

### `check_docs.py` da un falso positivo por fecha

El script compara fechas de modificación. Si acabas de tocar un archivo y el
changelog tiene la misma marca de tiempo, añade un margen:

```powershell
.\.venv\Scripts\python.exe scripts\check_docs.py --tolerance 5
```

## Logs

- `logs/nitshio.log`: log rotativo general.
- `logs/deletions.jsonl`: registro de borrados.
- `data/cache/screens`: capturas de pantalla.
- `data/cache/audio`: audio descargado por `play_song`.
