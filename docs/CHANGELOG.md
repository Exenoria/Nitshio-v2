# Changelog — Nitshio

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).
Cada entrada debe indicar fecha, qué cambió, por qué y archivos afectados.

## [0.4.0] — 2026-11-11

### Añadido

- **Fase 0 de la migración a Qwen3.5-2B:** `ollama.models.base` apunta a
  `qwen3.5:2b`, `base_family` a `qwen` y la versión activa a
  `nitshio-qwen2b`. **Por qué:** dejar el pipeline listo para el fine-tune
  sobre Qwen sin cambiar el resto de la arquitectura.
- **`ollama.think: false`** en `config/config.yaml`: el cliente envía
  `"think": false` en `/api/chat` y descarta `thinking`, `reasoning` y
  `reasoning_content` antes de parsear o medir. **Por qué:** Qwen3.5 puede
  devolver razonamiento y romper el contrato JSON de personalidad.
- **Persona corta activa** `brain/persona/nitshio_short.md` (200–300 tokens) y
  `brain/persona/few_shots_short.jsonl` (4 ejemplos variados). **Por qué:** el
  prompt largo de Phi y sus 14 few-shots producían copia literal; la persona
  breve define identidad, doble fachada tsundere, primera persona, formato JSON
  y guía de emociones sin la sección «Ejemplos de tono».
- **Aviso de fallback** en `pipeline/text.py`, `scripts/chat_cli.py` y `run.py`
  (texto y voz): se imprime un mensaje claro por `stderr` cuando el JSON es
  inválido y la respuesta cruda con `--debug`.
- **Medición de residencia simultánea** en `scripts/check_models.py`: comprueba
  si el modelo principal y el router caben juntos en VRAM y sugiere
  alternativas.
- **Evaluación de la migración** en
  `training/eval/reports/eval_20261111_115248.{md,json}`: `nitshio-qwen2b`
  0.6695 frente a `nitshio-v1` 0.7192 con los mismos 18 prompts, 100 % de JSON
  válido y 33.1 tok/s frente a 13.7.
- **Tests nuevos:** cobertura de `think=false`, descarte de razonamiento,
  familia `qwen` en el Modelfile, persona corta, aviso de fallback y residencia
  de VRAM (`tests/test_text_pipeline.py` y ampliaciones en la suite).

### Cambiado

- **`training/build_model.py` y `models/Modelfile.template`:** el `TEMPLATE` y
  los `stop` de Phi solo se emiten cuando `base_family` es `phi` o cuando se
  usa `--gguf` con plantilla explícita. Para Qwen, Ollama usa su renderer y
  parser nativos. El Modelfile activo generado es `models/Modelfile.qwen2b`.
- **Dry-runs de entrenamiento:** `finetune_qlora.py` y `export_gguf.py`
  reportan `Qwen/Qwen3.5-2B` como base y `q4_K_M` como cuantización (348
  ejemplos → 312 train / 36 val).
- **Registro de versiones:** `models/versions.json` marca `nitshio-qwen2b`
  como activo, conserva `nitshio-v1` como anterior y registra `nitshio-v2`
  como GGUF de Phi-3.5 (`type=gguf`), que no es la versión Qwen pendiente.
- **Router en CPU documentado:** `config/config.yaml` incluye `num_gpu: 0`
  comentado para el router. **Por qué:** en 4 GB, `qwen3.5:2b` ocupa
  1808/2793 MB en GPU y deja ~985 MB en RAM; con `qwen3.5:0.8b` (544 MB) no
  caben completos a la vez.

### Corregido

- La documentación ya no afirma que `nitshio-v2` no se generó: existe como GGUF
  de Phi-3.5 y falta reentrenarlo sobre Qwen3.5-2B.
- El modo voz muestra el aviso de fallback y, con `--debug`, las respuestas
  crudas, igual que el modo texto.
- Los ejemplos de few-shots activos no se copian literalmente porque el SYSTEM
  ya no incluye la sección «Ejemplos de tono».

### Verificado

- `pytest`: **275 tests en verde**.
- `scripts/check_models.py`: activo 29.3 tok/s, base 31.4 tok/s, router
  101.7 tok/s, embeddings 0.23 s/768 dims; los modelos no co-residen completos.
- `training/evaluate.py`: 18 prompts, embeddings reales, sin juez; reporte y
  métricas adjuntos.
- `python scripts/check_docs.py`: changelog y documentos obligatorios al día.

### Documentación

- `README.md`, `docs/PLAN.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`,
  `docs/TRAINING.md` y `docs/TROUBLESHOOTING.md` actualizados con el estado de
  Qwen3.5-2B, `think=false`, la limitación de VRAM y el estado real de
  `nitshio-v2`.

### Archivos afectados

- `config/config.yaml`
- `brain/llm/ollama_client.py`
- `brain/llm/orchestrator.py`
- `brain/llm/parser.py`
- `brain/persona/nitshio_short.md`
- `brain/persona/few_shots_short.jsonl`
- `pipeline/text.py`
- `run.py`
- `scripts/chat_cli.py`
- `scripts/check_models.py`
- `scripts/test_voice_roundtrip.py`
- `training/build_model.py`
- `training/evaluate.py`
- `training/generate_synthetic.py`
- `training/service.py`
- `training/eval/reports/eval_20261111_115248.md`
- `training/eval/reports/eval_20261111_115248.json`
- `models/Modelfile.template`
- `models/Modelfile.qwen2b`
- `models/Modelfile`
- `models/Modelfile.slim`
- `models/versions.json`
- `tests/test_ollama_client.py`
- `tests/test_orchestrator.py`
- `tests/test_persona.py`
- `tests/test_training_build_model.py`
- `tests/test_check_models.py`
- `tests/test_text_pipeline.py`
- `README.md`
- `docs/PLAN.md`
- `docs/ARCHITECTURE.md`
- `docs/DECISIONS.md`
- `docs/TRAINING.md`
- `docs/TROUBLESHOOTING.md`
- `docs/CHANGELOG.md`

## [0.3.0] — 2026-11-08

### Añadido

- **Pipeline de entrenamiento del Paso 3 (núcleo local):** esquema JSONL
  estricto, validación, deduplicación, hashes, split estratificado, generador
  sintético, evaluador, registro de versiones, construcción de Modelfile, UI
  Gradio y scripts cloud. **Por qué:** preparar el fine-tune en Kaggle/Colab,
  evaluar antes/después y activar o revertir versiones sin editar el YAML a
  mano.
- **Dataset semilla** `training/datasets/seed.jsonl`: 48 ejemplos, 8 por
  emoción, sin errores ni duplicados.
- **Set de evaluación** `training/eval/prompts.yaml`: 18 prompts fijos con
  emoción esperada y respuesta de referencia.
- **Generador sintético** `training/generate_synthetic.py`: lotes por emoción,
  filtrado, deduplicación y manifiesto reproducible.
- **QLoRA y exportación GGUF** `training/finetune_qlora.py` y
  `training/export_gguf.py`, con `--dry-run` local y manifiestos de
  entrenamiento/exportación.
- **Notebook cloud** `training/cloud/Nitshio_QLoRA.ipynb` para Kaggle/Colab:
  validación, QLoRA, `llama.cpp`, GGUF y empaquetado.
- **Versionado y rollback** `training/versions.py`: registro canónico,
  activación segura de `config.yaml`, rollback y métricas por versión.
- **UI Gradio** `training/ui.py` con chat, 👍/👎, correcciones, editor de
  dataset, versiones, reconstrucción del Modelfile y plan cloud.
- **Configuración** `training.hf_model` y `training.synthetic` en
  `config/config.yaml`.
- **Tests de training:** 56 tests nuevos más cobertura de persona para el
  comportamiento de few-shots.

### Corregido

- El split ahora usa el método del mayor resto y garantiza al menos un ejemplo
  en cada partición activa cuando la emoción tiene 3 o más ejemplos. **Por
  qué:** validación y prueba quedaban desproporcionadas con el dataset pequeño.
- `training/ui.py` detecta Gradio 5 y 6 (`Chatbot.type` ya no existe en 6).
  **Por qué:** la UI debía arrancar con la versión instalada.
- `training/generate_synthetic.py` acepta listas JSON directas además del
  objeto `{"examples": [...]}`. **Por qué:** el modelo no siempre respeta el
  envoltorio.
- `training/service.py` crea el directorio de correcciones antes de escribir.
  **Por qué:** el primer feedback podía fallar si `training/datasets` no
  existía.
- `scripts/check_docs.py` incluye la carpeta `training/` en la comprobación.
  **Por qué:** el código nuevo debía quedar cubierto por la regla de
  documentación.
- `scripts/build_model.py` pasa a ser un wrapper compatible de
  `training/build_model.py`, que es la implementación canónica. **Por qué:**
  evitar dos renderizadores del Modelfile que se desincronicen.

### Verificado

- `pytest -q`: **249 tests en verde**.
- Dataset semilla: 48 ejemplos, 0 errores, 0 duplicados, hash
  `ef7320f770caa821a49000c64663dd8fae5e33dfcce0113a64975a1c279ca26d`.
- Split real con ratios 0.8/0.1/0.1 y `seed=42`: **36 train / 6 val / 6 test**.
- Dry-run real de QLoRA: 48 ejemplos → **42 train / 6 val**, base
  `microsoft/Phi-3.5-mini-instruct`.
- Dry-run real de exportación GGUF: cuantización **q4_K_M**.
- UI Gradio construida y arrancada en `http://127.0.0.1:7861/` con Gradio
  6.31.0.
- Evaluación real de `nitshio-v1` contra el modelo base con
  `training/evaluate.py --compare-base --no-judge` (18 prompts, embeddings de
  Ollama, sin juez): `nitshio-v1` score **0.5452** (JSON 100%, emoción 33.3%,
  similitud 0.477, 78.5 s/prompt) frente al base **0.4289** (JSON 94.4%,
  emoción 5.6%, similitud 0.451, 91.7 s/prompt). Reporte:
  `training/eval/reports/eval_20261109_051355.md`.
- Métricas de `nitshio-v1` adjuntadas a `models/versions.json` mediante
  `training/versions.py metrics`.

### Documentación

- `docs/TRAINING.md` reescrito con el flujo real, comandos, estado y
  pendientes.
- `docs/PLAN.md`: Paso 3 en curso con evidencias locales y aceptación cloud
  pendiente.
- `docs/ARCHITECTURE.md`: módulo `training/`, flujo de entrenamiento y estado
  de pruebas.
- `docs/DECISIONS.md`: D-017 (split con cobertura mínima) y D-018 (la UI no
  entrena en local).
- `docs/TROUBLESHOOTING.md`: sección de entrenamiento, Gradio, dataset, GGUF,
  evaluación y contexto.
- `README.md`: estado del Paso 3, comandos de entrenamiento y 249 tests.

### Archivos afectados

`training/`, `tests/`, `scripts/check_docs.py`, `scripts/build_model.py`,
`config/config.yaml`, `models/versions.json`, `README.md`, `docs/PLAN.md`,
`docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/TRAINING.md`,
`docs/TROUBLESHOOTING.md`, `docs/CHANGELOG.md`.

## [0.2.2] — 2026-11-08

### Corregido

- Se generalizaron rutas locales, datos del equipo y nombres de archivos de
  avatares en la documentación y el notebook. **Por qué:** preparar una copia
  pública sin publicar detalles identificables del entorno local.

### Archivos afectados

`README.md`, `docs/PLAN.md`, `docs/VSEEFACE.md`, `docs/TROUBLESHOOTING.md`,
`docs/DECISIONS.md`, `docs/TRAINING.md`, `docs/CHANGELOG.md`,
`training/cloud/Nitshio_QLoRA.ipynb`.

## [0.2.1] — 2026-11-07

### Añadido

- **Prueba integral de voz:** `scripts/test_voice_roundtrip.py` inyecta una
  frase Piper en un micrófono falso, usa Moonshine real, el orquestador real
  (Ollama + memoria), el TTS Piper y el avatar VMC, y graba `CABLE Output`
  para verificar el audio. **Por qué:** cerrar el Paso 2 sin depender de que
  una persona hable al micrófono y dejar una prueba reproducible.
- **Router con rutas exactas:** `config/prompts/router.md` incluye ejemplos y
  reglas para `config/config.yaml`, listados de `config/` y
  `brain/persona/nitshio.md`. **Por qué:** qwen3.5:0.8b no elegía
  `read_own_code` sin ejemplos explícitos.
- **Tests de micrófono:** `tests/test_microphone.py` cubre el timeout por frame
  de `listen_once()` y los casos sin audio, stop y cierre.

### Corregido

- `Microphone.listen_once()` comprueba el timeout en cada frame, no solo al
  abrir el stream. **Por qué:** un micrófono sin utterances podía bloquear el
  pipeline indefinidamente.
- `scripts/test_emotions.py` imprime valores máximos y finales por emoción.
  **Por qué:** la prueba visual anterior no mostraba si el blendshape había
  llegado al valor esperado antes de volver a neutral.

### Verificado

- `pytest -q`: **192 tests en verde**.
- `scripts/test_voice_roundtrip.py --no-tools --json`:
  `"hola nizio cómo está?"` → respuesta de `nitshio-v1` → 2 chunks TTS
  (9.58 s) → `CABLE Output` RMS 0.068506, pico 0.283783; emoción `neutral`
  recibida por el avatar.
- TTS real a `CABLE Input`: audio confirmado en `CABLE Output` con RMS
  0.123407 y pico 0.985046.
- Micrófono real `USB MIC PRO`: abre y captura con RMS 0.001737 y pico
  0.019348.
- Router real: tres consultas dispararon `read_own_code` con las rutas
  correctas.
- `scripts/test_emotions.py --all`: las seis emociones se mapean, alcanzan su
  valor máximo y vuelven a neutral.
- Lip-sync de VSeeFace: confirmado visualmente con
  `OldLipSyncProvider=1`; el proveedor nuevo no movía la boca. El ajuste
  persiste tras cerrar y reiniciar VSeeFace.

### Documentación

- `docs/VSEEFACE.md`: procedimiento exacto de lip-sync por audio, proveedor
  viejo, micrófono `CABLE Output`, avatar validado y evidencia.
- `docs/TROUBLESHOOTING.md`: diagnóstico de “se oye pero no mueve la boca”,
  prueba integral de voz y avatares sin grupos VRM.
- `docs/PLAN.md`: Paso 2 cerrado, evidencias reales y siguiente fase.
- `docs/ARCHITECTURE.md`: estado de pruebas y nota de lip-sync por audio.
- `docs/DECISIONS.md`: decisión D-016 sobre el proveedor viejo de lip-sync y
  los visemas VMC.
- `README.md`: estado actual, comando de prueba integral y limitaciones.

### Archivos afectados

`scripts/test_voice_roundtrip.py`, `scripts/test_emotions.py`,
`stt/microphone.py`, `tests/test_microphone.py`,
`config/prompts/router.md`, `README.md`, `docs/VSEEFACE.md`,
`docs/TROUBLESHOOTING.md`, `docs/PLAN.md`, `docs/ARCHITECTURE.md`,
`docs/DECISIONS.md`, `docs/CHANGELOG.md`.

## [0.2.0] — 2026-11-07

### Añadido

- **STT completo:** `stt/base.py` (contrato `STTProvider`, `Transcript` y
  normalización de audio), `stt/vad.py` (WebRTC VAD con fallback por energía),
  `stt/microphone.py` (stream de `sounddevice` y utterances),
  `stt/moonshine_provider.py` (Moonshine español real) y
  `stt/whisper_provider.py` (fallback opcional). **Por qué:** el pipeline de
  voz necesita una entrada local, tolerante a fallos y sin depender de PyTorch
  CUDA.
- **TTS completo:** `tts/base.py`, `tts/chunker.py` (primer chunk corto para
  bajar latencia), `tts/piper_provider.py`, `tts/fish_provider.py`,
  `tts/player.py` (selección de dispositivo, `sounddevice`, barge-in) y
  `tts/manager.py`. **Por qué:** voz local por chunks con salida a VB-Cable y
  Fish Audio opcional.
- **Memoria RAG:** `memory/embeddings.py` (Ollama `/api/embed` con fallback a
  `/api/embeddings`) y `memory/store.py` (ChromaDB persistente, colecciones
  separadas, búsqueda y contexto). **Por qué:** recuerdos, conversaciones,
  código y documentación tienen ciclos de vida distintos.
- **Herramientas plugin:** `tools/base.py`, `tools/registry.py`, `tools/files.py`,
  `tools/search_google.py`, `tools/search_youtube.py`, `tools/play_song.py`,
  `tools/open_google.py`, `tools/open_youtube.py`, `tools/screen.py` y
  `tools/read_own_code.py`. **Por qué:** el agente debe actuar sin hardcodear
  acciones en el orquestador.
- **Router de herramientas:** `brain/router/tool_router.py` con esquema
  `{tool, arguments, confidence, reason}`, umbral de confianza, reparación de
  JSON y catálogo dinámico. **Por qué:** decidir herramientas con un modelo
  pequeño evita gastar el modelo de personalidad.
- **Avatar VMC:** `avatar/emotions.py` (carga de `config/emotions.yaml` y
  transiciones fade in/hold/fade out), `avatar/vmc.py` (OSC a VSeeFace) y
  `avatar/lip_sync.py` (utilidad opcional). **Por qué:** emociones visibles y
  vuelta a neutral sin saltos bruscos.
- **Orquestador:** `brain/llm/orchestrator.py` une memoria, router,
  herramientas, LLM, historial, avatar y TTS con degradación elegante.
  **Por qué:** un único punto de entrada para texto y voz.
- **Pipelines:** `pipeline/text.py` (chat con comandos), `pipeline/voice.py`
  (`VoicePipeline`, `MicrophoneListener` y `CallMode` con barge-in).
- **Entrada principal:** `run.py` con modos `text`, `voice`, `call` y
  `diagnose`, y `run.bat`. Banderas `--once`, `--json`, `--speak`, `--debug`,
  `--no-memory`, `--no-tools`, `--no-avatar`, `--no-tts`, `--model` y
  `--config`.
- **Scripts:** `scripts/download_voice.py` (voz Piper), `scripts/test_emotions.py`
  (prueba VMC con `--dry-run`) y `scripts/check_docs.py` (changelog y docs
  obligatorios).
- **Documentación:** `README.md`, `docs/ARCHITECTURE.md`, `docs/TRAINING.md`,
  `docs/EXTENDING.md`, `docs/TROUBLESHOOTING.md` y `docs/VSEEFACE.md`.
  **Por qué:** el Paso 2 añade demasiados módulos para dejarlos sin guía.

### Corregido

- Los adaptadores de STT/TTS aceptan tanto la configuración completa como su
  subsección (`stt`/`tts`). **Por qué:** `crear_stt()`/`crear_tts()` pasan la
  subsección y los tests pasan la configuración completa; antes Piper intentaba
  leer una ruta inexistente en la carpeta de usuario.
- `PiperProvider` valida que `model_path` sea un `.onnx` real antes de cargar.
  **Por qué:** un valor vacío o la raíz del proyecto producía errores confusos.
- `FilesTool.execute()` convierte `ToolError` y `OSError` en
  `ToolResult.fallo()`; la comprobación de symlinks ocurre antes de resolver la
  ruta real. **Por qué:** llamar a la herramienta directamente no debe lanzar
  excepciones y un symlink debe reportarse como symlink, no como ruta externa.
- `AvatarController.stop()` completa la transición a neutral antes del último
  envío VMC. **Por qué:** al cerrar, VSeeFace debe recibir la cara neutral, no
  un valor intermedio.
- `json-repair` se instaló y se usa como segunda pasada del parser y del
  router. **Por qué:** el modelo devuelve comas finales y JSON casi válido.

### Verificado

- `pytest -q`: **186 tests en verde**.
- Round-trip real Piper → Moonshine: audio de 2.21 s transcrito como
  `"hola soy nichio cómo está?"`.
- Pipeline real de texto con Ollama (`nitshio-v1`) y memoria Chroma:
  `{"response":"...","emotion":"neutral"}` válido.
- `run.py --mode diagnose`: Ollama 0.40.15, `CABLE Input` visible, voz Piper
  presente, Moonshine instalado y ocho herramientas habilitadas.

### Pendiente

- Probar micrófono real (`--mode voice`), salida física por `CABLE Input` y
  recepción de las seis emociones en VSeeFace.
- Probar una herramienta real disparada por el router.
- El Paso 2 no se cierra formalmente hasta completar esas pruebas.

### Archivos afectados

`brain/`, `stt/`, `tts/`, `memory/`, `tools/`, `avatar/`, `pipeline/`,
`scripts/`, `tests/`, `config/`, `run.py`, `run.bat`, `README.md`, `docs/`.

## [0.1.1] — 2026-11-07

### Corregido

- `scripts/check_models.py` mide primero el modelo activo y luego la base,
  descargando cada uno con `ollama stop` antes de la siguiente medición.
  **Por qué:** en 4 GB de VRAM dos modelos compitiendo falseaban el veredicto
  con un “NO SOPORTADO” que no reflejaba el rendimiento real.
- Añadidos tests de `parar_modelo()` (ejecuta `ollama stop`, no falla si falta
  el binario y tolera errores del subproceso). **Por qué:** la función toca el
  entorno y debía quedar cubierta sin depender de Ollama real.

### Verificado

- `pytest -q`: 40 tests en verde.
- `check_models.py` real: `nitshio-v1` 15.6 tok/s, base 19.0 tok/s, router
  73.2 tok/s, embeddings 0.15 s / 768 dims, JSON válido en los tres modelos.
  **Veredicto: AJUSTADO** (usable con margen justo por los 4 GB de VRAM).
- Paso 1 cerrado: personalidad, few-shots, Modelfile, parser con reparación y
  fallback, cliente Ollama con reintentos y CLI de chat funcionando.

### Archivos afectados

`scripts/check_models.py`, `tests/test_check_models.py`, `docs/PLAN.md`,
`docs/CHANGELOG.md`.

## [0.1.0] — 2026-11-07

### Añadido

- Estructura inicial del proyecto en una carpeta independiente:
  `brain/`, `stt/`, `tts/`, `memory/`, `tools/`, `avatar/`, `training/`,
  `config/`, `docs/`, `scripts/`, `tests/`, `data/`, `logs/` y `workspace/`.
  **Por qué:** separar responsabilidades desde el inicio y evitar mezclar
  inferencia, entrenamiento, avatar y herramientas.
- `docs/PLAN.md` con entorno detectado, decisiones, fases, riesgos y criterios
  de aceptación. **Por qué:** el proyecto es grande y necesita un gate por fase.
- `config/config.yaml` como única fuente de nombres de modelo, rutas, audio,
  avatar, memoria, herramientas y entrenamiento. **Por qué:** requisito de no
  hardcodear modelos y facilitar cambios sin tocar código.
- `config/emotions.yaml` con las seis emociones iniciales, blendshapes VRM y
  tiempos de transición. **Por qué:** las emociones deben ser configurables y
  visibles en VSeeFace.
- `AGENTS.md` con la regla obligatoria de actualizar changelog y documentación
  en la misma tarea. **Por qué:** la documentación es parte del producto.
- `docs/DECISIONS.md` con las decisiones iniciales y sus motivos.
- `.env.example`, `.gitignore`, `requirements.txt`, `requirements-dev.txt`,
  `requirements-train.txt` y `requirements-optional.txt`.
  **Por qué:** separar dependencias de inferencia local, desarrollo,
  entrenamiento cloud y extras opcionales.

### Completado en esta fase (Paso 0)

- Descargado `phi3.5:3.8b-mini-instruct-q4_K_M` (2.4 GB).
- Implementado `scripts/check_models.py` con:
  - comprobación de Ollama y tags,
  - medición de latencia y tokens/s,
  - lectura de GPU/RAM y modelos cargados,
  - veredicto y alternativas más ligeras,
  - modos `--pull`, `--skip-perf` y `--json`.
- Añadidos `tests/test_check_models.py` (12 tests) y `pytest.ini`.
- Prueba real en el entorno local: principal 20.9 tok/s, router 92.1 tok/s,
  embeddings 0.12 s, JSON válido en ambos. Veredicto: SOPORTADO.
- **Por qué:** validar la base obligatoria antes de escribir el cerebro evita
  construir sobre un entorno que no rinde.

### Archivos afectados

`docs/PLAN.md`, `docs/CHANGELOG.md`, `docs/DECISIONS.md`, `AGENTS.md`,
`config/config.yaml`, `config/emotions.yaml`, `.env.example`, `.gitignore`,
`requirements*.txt`, `scripts/check_models.py`, `tests/test_check_models.py`,
`pytest.ini`.
