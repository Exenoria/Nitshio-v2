# Nitshio

LLM creado para ser una acompañante en mi escritorio. Nitshio es una waifu IA
**100 % local** para Windows: personalidad tsundere en español
mexicano, voz, memoria, herramientas, avatar VRM en VSeeFace y un pipeline de
entrenamiento pensado para la nube.

El cerebro, el STT, el TTS local, la memoria y el avatar funcionan sin internet.
Solo necesitan red la búsqueda web, la reproducción de YouTube y el proveedor
opcional Fish Audio.

> **Estado actual:** Fase 0 migrada de Phi-3.5 a **Qwen3.5-2B**;
> `nitshio-qwen2b` está activo y evaluado con 18 prompts (score 0.6695 frente a
> 0.7192 de `nitshio-v1`, pero con 33.1 tok/s frente a 13.7). `nitshio-v2` ya
> existe como GGUF de Phi-3.5 registrado; falta reentrenarlo sobre Qwen3.5-2B y
> cerrar la aceptación real en Colab. Paso 2 (pipeline completo) sigue
> **cerrado** y el Paso 3 (entrenamiento) está **implementado en local**:
> dataset combinado de 348 ejemplos, generador sintético, evaluador, versionado,
> UI Gradio, notebook de Colab y carga de GGUF con `--gguf`.

## Requisitos

- Windows 10/11.
- Python 3.11.
- [Ollama](https://ollama.com/) con los modelos declarados en
  `config/config.yaml`.
- **Opcional para voz:** VB-Cable, VSeeFace y un avatar `.vrm`.
- **Opcional para canciones:** `ffplay` (viene con FFmpeg) en el `PATH`.
  En 4 GB de VRAM, el router `qwen3.5:0.8b` y `qwen3.5:2b` no caben completos a
  la vez; ver [Limitaciones conocidas](#limitaciones-conocidas).

No se instala PyTorch CUDA local. La inferencia usa Ollama y ONNX Runtime CPU
para Moonshine.

## Instalación rápida

```powershell
cd "$env:USERPROFILE\Nitshio"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Descarga la voz de Piper (solo la primera vez):

```powershell
.\.venv\Scripts\python.exe scripts\download_voice.py
```

Comprueba el entorno:

```powershell
.\.venv\Scripts\python.exe run.py --mode diagnose
```

## Uso

```powershell
# Chat de texto interactivo
.\.venv\Scripts\python.exe run.py --mode text

# Un solo mensaje, salida legible o JSON
.\.venv\Scripts\python.exe run.py --mode text --once "Hola, Nitshio"
.\.venv\Scripts\python.exe run.py --mode text --once "Hola" --json

# Texto con voz (TTS Piper)
.\.venv\Scripts\python.exe run.py --mode text --speak

# Conversación por micrófono (VAD + Moonshine)
.\.venv\Scripts\python.exe run.py --mode voice

# Modo llamada con barge-in
.\.venv\Scripts\python.exe run.py --mode call

# Prueba integral sin hablar al micrófono: Piper -> Moonshine -> Ollama -> TTS
.\.venv\Scripts\python.exe scripts\test_voice_roundtrip.py --no-tools --json
```

También puedes usar `run.bat`:

```powershell
.\run.bat --mode text --once "Hola"
```

### Banderas útiles

| Bandera | Efecto |
|---|---|
| `--config RUTA` | Usa otro YAML de configuración. |
| `--model TAG` | Sobrescribe `ollama.models.active`. |
| `--no-memory` | Desactiva ChromaDB para ese arranque. |
| `--no-tools` | Desactiva el router y las herramientas. |
| `--no-avatar` | No envía VMC a VSeeFace. |
| `--no-tts` | No sintetiza voz. |
| `--speak` | Lee las respuestas en modo texto. |
| `--json` | Salida JSON en `--once`. |
| `--debug` | Muestra intentos, memoria y herramientas. |

## Cómo funciona

```text
Texto / micrófono
      │
      ▼
STT Moonshine + VAD (solo voz)
      │
      ▼
Router qwen3.5:0.8b ──► herramienta (plugin) ──► resultado
      │
      ▼
LLM de personalidad (nitshio-qwen2b en Ollama, think=false)
      │
      ├── memoria ChromaDB (RAG)
      ├── parser JSON + reintento + fallback
      ├── TTS Piper por chunks ──► CABLE Input
      └── avatar VMC ──► VSeeFace
```

- La personalidad de Qwen vive en `brain/persona/nitshio_short.md` (200–300
  tokens) y sus 4 ejemplos en `brain/persona/few_shots_short.jsonl`. Los
  archivos largos de Phi (`nitshio.md` y `few_shots.jsonl`) se conservan.
- La salida del modelo es siempre
  `{"response": "...", "emotion": "..."}`; las emociones permitidas se leen de
  `config/emotions.yaml`.
- `ollama.think: false` se envía en cada llamada y el cliente descarta
  `thinking`/`reasoning` para que el razonamiento de Qwen no contamine el JSON.
  Si el parser agota los reintentos, el CLI avisa y `--debug` muestra la
  respuesta cruda.
- Los nombres de modelo, rutas, dispositivos y herramientas están en
  `config/config.yaml`. **No hay nombres de modelo hardcodeados.**
- Los secretos van en `.env` (ver `.env.example`), nunca en el YAML ni en el
  código.

### Migración a Qwen3.5-2B (Fase 0)

`nitshio-qwen2b` se creó desde `qwen3.5:2b` con `build_model.py --output
models/Modelfile.qwen2b`. El Modelfile de Qwen **no** lleva `TEMPLATE` ni stops
de Phi: Ollama usa la plantilla nativa del modelo. Los GGUF antiguos cargados
con `--gguf` sí conservan la plantilla Phi.

Evaluación real con los mismos 18 prompts, embeddings y sin juez
(`training/eval/reports/eval_20261111_115248.md`):

| Modelo | JSON | Emoción | Similitud | Latencia | Tok/s | Score |
|---|---:|---:|---:|---:|---:|---:|
| `nitshio-qwen2b` | 100% | 44.4% | 0.673 | 2.74 s | 33.1 | **0.6695** |
| `nitshio-v1` (persona breve) | 100% | 61.1% | 0.660 | 10.43 s | 13.74 | **0.7192** |

Qwen tiene ~3.8× menos latencia y 2.4× más tokens/s, y mejora la similitud,
pero todavía falla más emociones que `nitshio-v1`; el fine-tune del Paso 3 queda
pendiente. En hardware,
`check_models.py` midió 29.3 tok/s en el activo y 101.7 tok/s en el router. La
prueba simultánea no logró mantener ambos modelos **completos** en VRAM:
`qwen3.5:2b` quedó parcialmente en CPU (1808/2793 MB) y el router ocupó 544 MB.
Por eso el script recomienda `num_gpu: 0` en `ollama.options.router` o usar el
mismo modelo como router; ver [DECISIONS.md](docs/DECISIONS.md) y
[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Memoria

ChromaDB persiste en `data/chroma`. Las colecciones son:

- `memories`: hechos y preferencias.
- `conversations`: turnos de conversación.
- `self_knowledge`: código y estructura del proyecto.
- `documentation`: documentación indexada.

Los embeddings se calculan con Ollama (`nomic-embed-text`) fuera de Chroma.
Así se evita que Chroma descargue su propio modelo oculto.

## Herramientas

Habilitadas por defecto en `config/config.yaml`:

| Herramienta | Función |
|---|---|
| `search_google` | Busca en Google CSE o DuckDuckGo. |
| `search_youtube` | Busca videos con yt-dlp. |
| `play_song` | Descarga el audio y lo reproduce con `ffplay`. |
| `open_google` | Abre una búsqueda en el navegador. |
| `open_youtube` | Abre YouTube o una URL válida. |
| `files` | Lee, escribe, lista, mueve y borra archivos. |
| `screen` | Captura la pantalla y opcionalmente la analiza. |
| `read_own_code` | Lee el código y la documentación propios. |

### Seguridad de archivos

- Solo se opera dentro de `tools.allowed_paths` (por defecto `workspace/`).
- Se resuelven symlinks y se rechazan por defecto.
- Borrar pide confirmación (`confirm_delete: true`).
- El borrado va a la papelera (`send2trash`) y se registra en
  `logs/deletions.jsonl`.
- Se limitan tamaño de lectura y escritura.

## Entrenamiento

El fine-tune se ejecuta **solo en Kaggle/Colab**; la T1200 de 4 GB no entrena.
Localmente se preparan datos, se evalúa y se carga el GGUF resultante.

```powershell
# Validar el dataset importado y el combinado
.\.venv\Scripts\python.exe training\validate_dataset.py training\datasets\nitshio_dataset_300.jsonl --strict --json
.\.venv\Scripts\python.exe training\validate_dataset.py training\datasets\train.jsonl --strict --json

# Generar datos sintéticos (requiere Ollama)
.\.venv\Scripts\python.exe training\generate_synthetic.py --append --json

# Abrir la UI de entrenamiento y correcciones
.\.venv\Scripts\python.exe training\ui.py

# Comparar el modelo activo contra el base
.\.venv\Scripts\python.exe training\evaluate.py --compare-base --no-judge

# Crear y activar una versión desde un GGUF local
.\.venv\Scripts\python.exe training\build_model.py --version nitshio-v2 `
  --gguf C:\ruta\nitshio-v2.gguf --create --activate
```

El notebook `training/cloud/Nitshio_QLoRA.ipynb` está preparado para Colab:
descomprime el proyecto en `/content`, combina `seed.jsonl` con
`nitshio_dataset_300.jsonl`, entrena con QLoRA y exporta el GGUF cuantizado.
En local, `training/build_model.py --gguf` lo registra con `type=gguf` y lo
activa. La comparación histórica de Phi (18 prompts, sin juez) dio **0.5452** a
`nitshio-v1` frente a **0.4289** del base Phi-3.5. La migración a Qwen3.5-2B se
evaluó después con el mismo set y la persona breve: `nitshio-qwen2b` **0.6695**
frente a `nitshio-v1` **0.7192**, ambos con JSON válido al 100%. El reporte está
en `training/eval/reports/eval_20261111_115248.md`. La calidad del texto todavía
es irregular y el fine-tune cloud sobre Qwen sigue pendiente. El flujo completo,
los comandos y el estado de aceptación están en [TRAINING.md](docs/TRAINING.md).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_docs.py
```

La suite cubre configuración, parser, persona, cliente Ollama, router, memoria,
herramientas, seguridad de archivos, VAD, STT/TTS, avatar, orquestador, pipeline
de voz, micrófono, training y el propio `check_docs.py` (**275 tests**).

## Documentación

- [Plan y fases](docs/PLAN.md)
- [Arquitectura](docs/ARCHITECTURE.md)
- [Decisiones](docs/DECISIONS.md)
- [Changelog](docs/CHANGELOG.md)
- [Entrenamiento](docs/TRAINING.md)
- [Extender Nitshio](docs/EXTENDING.md)
- [Solución de problemas](docs/TROUBLESHOOTING.md)
- [VSeeFace](docs/VSEEFACE.md)

## Limitaciones conocidas

- El modelo activo rinde ~29.3 tok/s en la T1200 de 4 GB (base Qwen 31.4 tok/s,
  router 101.7 tok/s); conviene mantener `num_ctx: 2048`.
- En 4 GB de VRAM, `qwen3.5:2b` y el router `qwen3.5:0.8b` no caben completos a
  la vez: el principal deja ~1 GB en RAM. Propuesta: mover el router a CPU con
  `ollama.options.router.num_gpu: 0` (comentado en `config/config.yaml`) o usar
  el mismo modelo como router.
- La calidad de personalidad del modelo base todavía es irregular: acierta
  44.4% de las emociones y el texto queda por debajo de `nitshio-v1` en el score
  compuesto. Se ajustará con el fine-tune del Paso 3; el parser y el formato
  JSON ya están validados.
- La conversación humana por `--mode voice` queda como confirmación opcional:
  la cadena completa se prueba con `scripts/test_voice_roundtrip.py` y el
  micrófono real ya se validó por separado.
- El lip-sync de VSeeFace requiere el **proveedor viejo** (“Usar el modelo viejo
  para lip sync”). Con el nuevo, el audio llega pero la boca no se mueve. Ver
  [VSeeFace](docs/VSEEFACE.md).
- El fine-tune solo se ejecuta en Kaggle/Colab; la T1200 no entrena. `nitshio-v2`
  ya se generó y está registrado como GGUF de Phi-3.5; falta reentrenarlo sobre
  Qwen3.5-2B y cerrar su aceptación con el evaluador real.
- La UI Gradio requiere instalar `requirements.txt` (incluye `gradio>=5,<7`).
- Discord está previsto como adaptador opcional y aún no está conectado.
