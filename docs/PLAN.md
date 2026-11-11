# Plan de construcción — Nitshio

> Documento vivo. Se actualiza al cerrar cada fase.
> Última actualización: 2026-11-11

## 1. Resumen

Nitshio será una compañera IA 100 % local para Windows, con personalidad tsundere extrema,
voz, memoria, herramientas, avatar VRM en VSeeFace y un pipeline de entrenamiento con
versionado, evaluación y interfaz web. El único componente que puede requerir red es la
búsqueda web, la reproducción de YouTube y el proveedor opcional Fish Audio; el cerebro,
STT, TTS local, memoria y avatar funcionan sin conexión.

## 2. Entorno de validación (2026-11-07)

| Elemento | Estado |
|---|---|
| SO | Windows 10/11 |
| Python | 3.11.0 |
| Ollama | API local respondiendo |
| Modelos | Se configuran en `config/config.yaml` |
| Avatar | VRM con grupos de blendshapes, opcional |
| Audio virtual | VB-Cable, opcional |

### Proyecto previo encontrado

Existe un borrador independiente (Node.js) con una personalidad similar y un
dataset. **No se borrará ni modificará**. Nitshio se construirá aparte. Si el
usuario lo autoriza, se importarán ejemplos de ese dataset como semilla.

### Viabilidad de hardware

- **Inferencia:** Qwen3.5-2B Q4_K_M con contexto 2048 cabe en los 4 GB de VRAM;
  el router 0.8B también, pero **no ambos completos a la vez**. La migración
  desde Phi-3.5 se midió con `check_models.py` (ver “Medición real de la Fase
  A” más abajo).
- **Fine-tuning Qwen3.5-2B:** decisión confirmada: **solo cloud** (Kaggle/Colab con
  GPU gratuita). El entorno local no se usará para entrenar. El pipeline de
  entrenamiento, exportación GGUF y evaluación se ejecutará en la nube; Ollama
  local solo hace inferencia y recibe el modelo resultante.
- **PyTorch local:** no se instala build CUDA. La inferencia local usa Ollama (GPU vía
  su propio runtime) y ONNX Runtime CPU para STT. `requirements-train.txt` queda para
  el entorno cloud.

## 3. Decisiones confirmadas

1. **Raíz del proyecto:** carpeta independiente del borrador Nova y de la
   instalación global de npm.
2. **Modelo principal:** `qwen3.5:2b` como base de inferencia y fine-tune
   (`Qwen/Qwen3.5-2B` en Hugging Face). Nombres en `config/config.yaml`, nunca en
   código. La plantilla Phi-3.5 queda solo para modelos de la familia `phi` y
   para GGUF antiguos cargados con `--gguf`.
3. **Router:** `qwen3.5:0.8b` (ya existe). Configurable para cambiar a otro tag.
4. **STT por defecto:** Moonshine en español (`moonshine-voice`, modelo streaming
   small/tiny es). Hay soporte oficial de español en 2026. Fallback: faster-whisper.
5. **TTS local por defecto:** Piper con voz española (es_MX). Fish Audio opcional por
   `.env`. Alternativa local futura: Moonshine TTS.
6. **Borrado de archivos:** por defecto pide confirmación (`confirm_delete: true`),
   manda a papelera con `send2trash`, limita a `allowed_paths` y registra en log.
   El usuario puede poner `false` explícitamente.
7. **Búsqueda Google:** sin API key se abre Google en el navegador y se usa
   DuckDuckGo/HTML como fuente de resultados. Si el usuario aporta clave de Google
   Custom Search o SerpAPI, se usa esa.
8. **Discord:** adaptador opcional; se implementa y se prueba la lógica local, pero la
   conexión real requiere token y servidor del usuario.
9. **Audio:** salida a `CABLE Input` (VB-Cable) para que VSeeFace lea de
   `CABLE Output`; dispositivo configurable por nombre o índice.
10. **VSeeFace:** VMC/OSC en `127.0.0.1:39539`; blendshapes VRM estándar.
11. **ChromaDB:** colecciones `memories`, `conversations`, `self_knowledge` (código)
    y `documentation` (docs). Embeddings vía Ollama `nomic-embed-text`.
12. **Idioma:** español mexicano por defecto; configurable.

## 4. Arquitectura objetivo

```text
Usuario
  │  (texto o micrófono)
  ▼
STT Moonshine + VAD ──────────────┐
  ▼                               │
Tool Router (qwen3.5:0.8b)        │
  │                               │
  ├── Herramienta (plugin) ───────┤
  │                               ▼
  └──────────────────► LLM personalidad (Qwen3.5-2B / nitshio-qwen2b)
                                  │
                                  ▼
                        Memoria ChromaDB (RAG)
                                  │
                                  ▼
                     Parser JSON + reintento/reparación
                                  │
             ┌────────────────────┼────────────────────┐
             ▼                    ▼                    ▼
       TTS por chunks        Avatar VMC          Acciones agente
       (Piper/Fish)          (VSeeFace)          (plugins)
             │                    │
             ▼                    ▼
       VB-Cable ──────────► VSeeFace (lip-sync + emociones)
```

## 5. Estructura de carpetas

```text
Nitshio/
├─ AGENTS.md
├─ README.md
├─ requirements.txt
├─ requirements-train.txt
├─ requirements-dev.txt
├─ .env.example
├─ .gitignore
├─ install.ps1
├─ run.py
├─ run.bat
├─ config/
│  ├─ config.yaml
│  ├─ emotions.yaml
│  └─ prompts/router.md
├─ brain/
│  ├─ persona/nitshio_short.md
│  ├─ persona/few_shots_short.jsonl
│  ├─ persona/nitshio.md        (histórico, rollback)
│  ├─ llm/ollama_client.py
│  ├─ llm/parser.py
│  ├─ llm/orchestrator.py
│  └─ router/tool_router.py
├─ stt/
│  ├─ base.py
│  ├─ microphone.py
│  ├─ moonshine_provider.py
│  ├─ whisper_provider.py
│  └─ vad.py
├─ tts/
│  ├─ base.py
│  ├─ piper_provider.py
│  ├─ fish_provider.py
│  ├─ chunker.py
│  ├─ manager.py
│  └─ player.py
├─ pipeline/
│  ├─ text.py
│  └─ voice.py
├─ memory/
│  ├─ store.py
│  ├─ embeddings.py
│  └─ indexer.py
├─ tools/
│  ├─ base.py
│  ├─ registry.py
│  ├─ search_google.py
│  ├─ search_youtube.py
│  ├─ play_song.py
│  ├─ open_google.py
│  ├─ open_youtube.py
│  ├─ files.py
│  ├─ screen.py
│  └─ read_own_code.py
├─ avatar/
│  ├─ vmc.py
│  ├─ emotions.py
│  └─ lip_sync.py
├─ training/
│  ├─ datasets/
│  ├─ build_model.py
│  ├─ validate_dataset.py
│  ├─ generate_synthetic.py
│  ├─ finetune_qlora.py
│  ├─ export_gguf.py
│  ├─ evaluate.py
│  ├─ versions.py
│  ├─ ui.py
│  └─ eval/prompts.yaml
├─ scripts/
│  ├─ check_models.py
│  ├─ check_docs.py
│  ├─ download_voice.py
│  ├─ test_emotions.py
│  ├─ test_voice_roundtrip.py
│  ├─ build_manifest.py
│  └─ index_self.py
├─ models/
│  ├─ Modelfile.template
│  └─ versions.json
├─ tests/
├─ docs/
│  ├─ PLAN.md
│  ├─ ARCHITECTURE.md
│  ├─ CHANGELOG.md
│  ├─ DECISIONS.md
│  ├─ TRAINING.md
│  ├─ EXTENDING.md
│  ├─ TROUBLESHOOTING.md
│  └─ VSEEFACE.md
├─ data/
│  ├─ chroma/
│  ├─ audio/
│  └─ cache/
└─ logs/
```

## 6. Fases de ejecución

Cada fase termina con: pruebas ejecutadas, resumen de lo hecho, cómo probarlo y qué sigue.
No se avanza si la fase anterior falla.

### Estado de fases

| Fase | Estado | Evidencia |
|---|---|---|
| A — Paso 0: Ollama/hardware | ✅ Completada | `check_models.py` verde; `nitshio-qwen2b` 29.3 tok/s, base 31.4, router 101.7; en 4 GB principal + router no caben completos (2026-11-11) |
| B — Paso 1: cerebro | ✅ Completada | 40 tests; personalidad migrada a Qwen (`nitshio-qwen2b`, 0.6695); JSON válido y fallback con aviso |
| C — Paso 2: pipeline | ✅ Completada | 192 tests; prueba integral de voz Piper→Moonshine→Ollama→TTS→CABLE Output (RMS 0.0685); lip-sync visual confirmado; seis emociones VMC; router real |
| D — Paso 3: entrenamiento | 🔄 En curso (local completo; aceptación pendiente de nube) | 275 tests; datasets 300/348 y split 276/36/36; dry-run QLoRA Qwen 312/36; `--gguf` y notebook Colab listos; `nitshio-v2` GGUF Phi registrado; evaluación qwen2b 0.6695 vs v1 0.7192 |
| E — Paso 4: extensibilidad | ⏳ Pendiente | — |
| F — Paso 5: documentación | ⏳ Pendiente | — |
| G — Paso 6: emociones | ⏳ Pendiente | — |
| H — Paso 7: autoconciencia | ⏳ Pendiente | — |
| I — Cierre | ⏳ Pendiente | — |

### Fase A — Paso 0: base Ollama y hardware

- Crear estructura, `config/config.yaml`, `.env.example`, `requirements.txt`.
- Verificar GPU y el runtime GPU de Ollama (sin instalar PyTorch CUDA local).
- Descargar `qwen3.5:2b` y el router `qwen3.5:0.8b`.
- Implementar `scripts/check_models.py`:
  - comprueba Ollama y tags,
  - mide latencia y tokens/s de router, principal y embeddings,
  - lee VRAM/RAM y comprueba si principal + router caben juntos,
  - emite veredicto (bien/ajustado/no soportado) y alternativa más ligera.
- Prueba real con ambos modelos.
- **Aceptación:** script en verde, `qwen3.5:2b` responde JSON, router responde
  JSON, GPU visible y `think: false` evita mezclar razonamiento con la salida.

### Fase A.1 — Migración a Qwen3.5-2B (2026-11-11)

Objetivo: sustituir Phi-3.5 por Qwen3.5-2B sin tocar el resto del pipeline ni
empezar el Paso 3.

Cambios de configuración y runtime:

- [x] `ollama.models.base` = `qwen3.5:2b`, `active` = `nitshio-qwen2b`,
      `base_family` = `qwen`; `training.base_model` y `training.hf_model`
      actualizados a Qwen.
- [x] `ollama.think: false` se envía en cada llamada y el cliente descarta
      `thinking`/`reasoning`/`reasoning_content`.
- [x] Persona breve `nitshio_short.md` (200–300 tokens estimados) y 4 few-shots
      variados; los archivos largos de Phi se conservan.
- [x] `models/Modelfile.qwen2b` generado con `build_model.py --output`; no lleva
      `TEMPLATE` ni stops de Phi. `--gguf` conserva la plantilla Phi.
- [x] Fallback JSON con aviso explícito en el CLI y respuestas crudas con
      `--debug`.
- [x] `nitshio-qwen2b` creado y activado en Ollama.

Medición real de `scripts/check_models.py --json` en la T1200 de 4 GB, sin otros
modelos cargados:

| Rol | Modelo | Tok/s | JSON |
|---|---|---:|---:|
| active | `nitshio-qwen2b` | 29.3 | válido |
| base | `qwen3.5:2b` | 31.4 | válido |
| router | `qwen3.5:0.8b` | 101.7 | válido |

Los embeddings (`nomic-embed-text`, 768 dimensiones) respondieron en 0.23 s.

Prueba simultánea: `qwen3.5:2b` ocupa **1808 de 2793 MB** en GPU (deja ~985 MB
en RAM) y `qwen3.5:0.8b` ocupa **544.5 MB** completos. Ollama mantiene ambos
cargados (**2352.5 MB** combinados frente a un límite de **3758 MB**), pero el
principal no cabe entero: **no caben juntos en 4 GB de VRAM**. Forzar
`num_gpu: 99` dejó el principal completo en **2228.9 MB**, pero al cargar el
router Ollama lo expulsó; por eso no se adoptó ese ajuste. Alternativas
propuestas:

- Mover el router a CPU con `ollama.options.router.num_gpu: 0` (está comentado
  en `config/config.yaml`), dejando toda la VRAM al modelo de personalidad.
- Usar el mismo modelo como router (`ollama.models.router` = modelo activo) y
  bajar `ollama.options.router.num_ctx`.

Evaluación real con los mismos 18 prompts, embeddings y sin juez
(`training/eval/reports/eval_20261111_115248.md`):

| Modelo | JSON | Emoción | Similitud | Latencia | Tok/s | Score |
|---|---:|---:|---:|---:|---:|---:|
| `nitshio-qwen2b` | 100% | 44.4% | 0.673 | 2.74 s | 33.1 | **0.6695** |
| `nitshio-v1` (persona breve) | 100% | 61.1% | 0.660 | 10.43 s | 13.74 | **0.7192** |

Qwen es ~4× más rápido y mejora la similitud, pero todavía acierta menos
emociones que `nitshio-v1`; el fine-tune cloud queda pendiente. La comparación
histórica de Phi (0.5452 vs 0.4289) se conserva en `models/versions.json`.

### Fase B — Paso 1: cerebro y personalidad

- `brain/persona/nitshio_short.md` con identidad, doble fachada, formato JSON
  y emociones; `brain/persona/nitshio.md` se conserva para rollback.
- Cuatro few-shots variados en `few_shots_short.jsonl`, sin sección «Ejemplos
  de tono» que el modelo pueda copiar.
- `Modelfile.template` con SYSTEM, parámetros y MESSAGE; el `TEMPLATE` y los
  stops de Phi solo se conservan para la familia `phi` o para `--gguf`.
- `brain/llm/parser.py`: modelo Pydantic, validación de emociones, reintento con
  instrucción correctiva, reparación de JSON y fallback neutral.
- `brain/llm/ollama_client.py`: API chat con `format` = JSON Schema, `think=false`.
- CLI mínima de chat para probar personalidad.
- Tests: JSON válido, emoción inválida, texto extra, JSON roto, reintento.
- **Aceptación:** 10/10 prompts de prueba devuelven JSON válido; el fallback funciona.

### Fase C — Paso 2: pipeline completo

- STT Moonshine español + VAD; entrada de texto.
- Router con esquema `{tool, arguments, confidence, reason}` y catálogo dinámico.
- Memoria ChromaDB: guardar/recuperar recuerdos, hechos y resúmenes; embeddings Ollama.
- Herramientas plugin: Google, YouTube, reproducir, abrir, archivos, pantalla,
  autoconocimiento.
- Seguridad de archivos: `allowed_paths`, resolución de symlinks, papelera, log,
  confirmación por defecto.
- TTS Piper por chunks + reproductor con selección de dispositivo; Fish Audio opcional.
- Avatar VMC: emociones, transiciones y lip-sync (VSeeFace micrófono virtual).
- Modo llamada local con barge-in; adaptador Discord opcional.
- Tests unitarios por módulo e integración de texto; prueba de micrófono y VSeeFace.
- **Aceptación:** conversación por texto y voz; herramientas funcionan; audio sale por
  VB-Cable; VSeeFace recibe emoción; borrado protegido.

#### Estado real del Paso 2 (2026-11-07)

Implementado y validado:

- [x] STT Moonshine español + VAD WebRTC con fallback por energía.
- [x] Micrófono con `sounddevice` y segmentación de utterances.
- [x] Router con esquema validado, umbral de confianza y catálogo dinámico.
- [x] Memoria ChromaDB con embeddings de Ollama y RAG.
- [x] Ocho herramientas plugin y registro con validación de argumentos.
- [x] Seguridad de archivos: `allowed_paths`, symlinks, tamaño, papelera, log y
      confirmación por defecto.
- [x] TTS Piper por chunks, Fish opcional, reproductor con selección de
      dispositivo y barge-in.
- [x] Avatar VMC con seis emociones y transiciones.
- [x] Orquestador con degradación elegante.
- [x] Pipelines de texto, voz y modo llamada.
- [x] `run.py`, `run.bat`, `download_voice.py`, `test_emotions.py`,
      `test_voice_roundtrip.py` y `check_docs.py`.
- [x] 192 tests en verde (incluye memoria, herramientas, router, avatar,
      orquestador, voz, micrófono y check_docs).
- [x] Round-trip real Piper → Moonshine: audio de 2.21 s transcrito como
      “hola soy nichio cómo está?”.
- [x] Pipeline real de texto con Ollama: JSON válido, memoria activa y router
      disponible.
- [x] Micrófono real abre y captura: `[1] Micrófono (USB MIC PRO)`,
      RMS 0.001737, pico 0.019348.
- [x] TTS real a `CABLE Input`: 1 chunk de 123948 bytes; audio confirmado en
      `CABLE Output` con RMS 0.123407 y pico 0.985046.
- [x] Router real con tres rutas exactas (`config/config.yaml`, listado de
      `config`, `brain/persona/nitshio.md`).
- [x] Seis emociones VMC confirmadas visualmente y vuelta a neutral.
- [x] Lip-sync visual confirmado con `OldLipSyncProvider=1`; el ajuste
      persiste tras cerrar y reiniciar VSeeFace.
- [x] Prueba integral de voz (`scripts/test_voice_roundtrip.py`): frase Piper
      inyectada en micrófono falso → Moonshine real
      (`"hola nizio cómo está?"`) → `nitshio-v1` → 2 chunks TTS (9.58 s) →
      `CABLE Output` con RMS 0.068506 y pico 0.283783; el avatar recibió la
      emoción del turno.

Confirmación humana opcional (no bloquea la fase):

- [ ] Hablar por el micrófono real en `run.py --mode voice` y confirmar de oído
      la respuesta. La cadena ya está cubierta por la prueba integral y el
      micrófono real se probó por separado.

### Fase D — Paso 3: entrenamiento real

- Dataset JSONL con validación estricta (JSON, emoción, roles, longitudes).
- Generador de datos sintéticos con filtrado y deduplicación.
- Construcción de Modelfile y `ollama create nitshio-vN`.
- Fine-tune QLoRA con Unsloth en Kaggle/Colab (GPU gratuita), exportación a GGUF y
  carga en Ollama local.
- Versionado `models/versions.json`, activar/rollback.
- Evaluación antes/después con set fijo: JSON válido, emoción, consistencia, similitud
  semántica y juicio opcional del router; reporte Markdown lado a lado.
- UI Gradio: chat, 👍/👎, corrección de respuesta/emoción, editor de dataset,
  reconstruir Modelfile, lanzar fine-tune, activar versión.
- Correcciones guardadas también en ChromaDB.
- **Aceptación:** crear `nitshio-v2`, comparar contra `v1`, ver mejora o diferencia
  medible, activar y volver atrás.

#### Estado real del Paso 3 (actualizado 2026-11-11)

Implementado y probado en local:

- [x] Esquema estricto JSONL (`user`, `response`, `emotion`), validación,
      deduplicación exacta/casi duplicada y hashes reproducibles.
- [x] Dataset importado `training/datasets/nitshio_dataset_300.jsonl`: 300
      ejemplos, 50 por emoción, 0 errores y 0 duplicados.
- [x] Dataset combinado `training/datasets/train.jsonl`: 348 ejemplos únicos
      (seed + 300, 300 duplicados descartados), 58 por emoción.
- [x] Split determinista y estratificado del combinado: 276 train / 36 val /
      36 test con ratios 0.8/0.1/0.1 y `seed=42`.
- [x] Generador sintético por emoción con lotes, filtrado, deduplicación y
      manifiesto (`training/generate_synthetic.py`).
- [x] Evaluador con 18 prompts fijos, métricas de JSON, emoción, similitud,
      longitud, latencia, tokens/s, juez opcional y reporte Markdown + JSON
      (`training/evaluate.py`).
- [x] Análisis de latencia histórico (Phi): 78.5 s de pared y 5.83 tok/s bajo
      contexto lleno. Con Qwen y persona breve bajó a 2.74 s y 33.1 tok/s.
- [x] Registro de versiones, activación y rollback sin perder comentarios del
      YAML (`training/versions.py`).
- [x] Construcción de Modelfile y `ollama create` (`training/build_model.py`;
      `scripts/build_model.py` queda como wrapper compatible).
- [x] Carga de un GGUF local con `--gguf`: `FROM` absoluto, `TEMPLATE` de
      Phi-3.5 y registro `type=gguf` + `gguf_path`.
- [x] Fine-tune QLoRA con Unsloth/TRL y manifiesto (`training/finetune_qlora.py`).
- [x] Fusión LoRA, conversión GGUF y cuantización q4_K_M
      (`training/export_gguf.py`).
- [x] Notebook Colab con rutas `/content`, descompresión del zip y dataset
      combinado (`training/cloud/Nitshio_QLoRA.ipynb`).
- [x] UI Gradio con chat, 👍/👎, correcciones, editor de dataset, versiones,
      reconstrucción del Modelfile y plan cloud (`training/ui.py`).
- [x] Correcciones guardadas en `training/datasets/corrections.jsonl` y en
      ChromaDB.
- [x] Suite completa de **275 tests** en verde (incluye Qwen, `think=false`,
      persona corta, aviso de fallback y residencia de VRAM).
- [x] Dry-run real histórico de QLoRA (configuración Phi): 348 ejemplos →
      312 train / 36 val. No se repitió con Qwen en la Fase 0.
- [x] Dry-run real histórico de exportación GGUF (Phi): cuantización `q4_K_M`.
- [x] UI construida y arrancada en `http://127.0.0.1:7861/` con Gradio 6.31.0.
- [x] Evaluación real de `nitshio-v1` contra el base (18 prompts, embeddings,
      sin juez): score **0.5452** vs **0.4289**; métricas adjuntadas a
      `models/versions.json`.
- [x] `nitshio-v2` generado como GGUF de Phi-3.5 y registrado con `type=gguf`
      (`models/nitshio-v2/nitshio-q4_K_M.gguf`).
- [x] Migración Fase 0 a Qwen3.5-2B: `nitshio-qwen2b` activo y evaluado con los
      mismos 18 prompts (0.6695 vs 0.7192 de `nitshio-v1`).

Pendiente para cerrar la aceptación:

- [ ] Reentrenar sobre Qwen3.5-2B en Kaggle/Colab y descargar el nuevo GGUF.
- [ ] Comparar `v2` contra `v1` y el base con el evaluador real.
- [ ] Activar el `v2` basado en Qwen y probar rollback con el GGUF real.
- [ ] Definir el umbral de mejora aceptable.

### Fase E — Paso 4: extensibilidad

- Interfaces abstractas para LLM, STT, TTS, memoria y avatar.
- Proveedores intercambiables por config: Piper/Fish, Moonshine/Whisper.
- Emociones nuevas por `config/emotions.yaml` + blendshape.
- Herramientas nuevas por archivo en `tools/`.
- Discord opcional y modo llamada.
- `docs/EXTENDING.md` con ejemplos completos.
- **Aceptación:** añadir un proveedor/tool/emoción de prueba sin tocar el núcleo.

### Fase F — Paso 5: documentación y control

- README, ARCHITECTURE, CHANGELOG, DECISIONS, TRAINING, EXTENDING, TROUBLESHOOTING,
  VSEEFACE.
- `AGENTS.md` con la regla obligatoria de changelog + docs en la misma tarea.
- `scripts/check_docs.py`: compara cambios de código con entradas del changelog.
- Indexación de docs en ChromaDB.
- **Aceptación:** `check_docs.py` falla si se modifica código sin changelog y pasa
  cuando se documenta.

### Fase G — Paso 6: emociones en VSeeFace

- Mapeo en `config/emotions.yaml`: neutral, fun, angry, joy, sorrow, surprised.
- Envío VMC `/VMC/Ext/Blend/Val` + `/VMC/Ext/Blend/Apply`.
- Fade in/out configurable, duración y retorno a neutral.
- `scripts/test_emotions.py` recorre todas las emociones.
- Guía VSeeFace: VMC, VRM, micrófono virtual, lip-sync.
- **Aceptación:** las seis emociones se ven en VSeeFace y vuelven a neutral.

### Fase H — Paso 7: autoconciencia

- Contexto de arranque con raíz, SO, versión, modelos activos y herramientas.
- `MANIFEST.json` automático con árbol, módulos y propósito.
- Indexación de código y docs en ChromaDB; reindexado por hash al iniciar/cambiar.
- Herramienta `read_own_code` con validación de rutas.
- **Aceptación:** preguntar “¿dónde está tu config?” o “¿cómo funcionas?” responde en
  personaje y con datos correctos.

### Fase I — Cierre

- Suite completa de tests, instalación limpia, `run.bat`.
- Revisión de criterios de aceptación.
- Resumen final y guía de uso.

## 7. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Memoria GPU local insuficiente para QLoRA | Entrenamiento solo en Kaggle/Colab |
| PyTorch CUDA no instalado | No hace falta localmente: inferencia con Ollama + ONNX |
| Moonshine español cambia de API | Adaptador aislado + fallback faster-whisper |
| Piper no tiene voz es_MX | Descargar voz es_ES/es_MX; fallback Fish o SAPI |
| VMC no llega a VSeeFace | Script de diagnóstico OSC + guía de puerto/firewall |
| Latencia de Qwen3.5-2B en 4 GB | Contexto 2048, `think: false`, router separado o en CPU, TTS por chunks |
| Borrado accidental | Papelera, allowed_paths, confirmación, log, tests |
| Respuestas JSON inválidas | JSON Schema, reintento, reparación, fallback neutral |
| Discord requiere token | Adaptador opcional, no bloquea el resto |
| Búsqueda Google bloqueada | DuckDuckGo + abrir navegador; API key opcional |
| Dataset escaso | Generación sintética + correcciones desde la UI |

## 8. Criterios de aceptación globales

- [x] Ollama local con Qwen3.5-2B (`nitshio-qwen2b`) y router, sin nombres hardcodeados.
- [ ] Personalidad tsundere en español, salida JSON válida y parser robusto.
      El parser y el JSON ya están validados; la calidad de personalidad se
      ajusta en el Paso 3.
- [x] Pipeline texto + voz + VAD + barge-in.
- [ ] Memoria RAG funcional y correcciones recordadas al instante. La memoria
      RAG funciona; las correcciones desde la UI llegan en el Paso 3.
- [x] Herramientas con plugins y borrado seguro.
- [x] TTS por chunks, salida a VB-Cable, Fish opcional.
- [x] VSeeFace recibe emociones y lip-sync.
- [ ] Fine-tune QLoRA + GGUF + Ollama, versionado y rollback. El pipeline
      local está implementado y `nitshio-v2` (Phi) ya está registrado; falta
      reentrenar sobre Qwen3.5-2B y validar el rollback.
- [ ] Evaluación antes/después con reporte. El evaluador se ejecutó contra
      `nitshio-v1`, el base Phi y `nitshio-qwen2b`; falta comparar el `v2`
      basado en Qwen.
- [x] UI Gradio de entrenamiento y correcciones.
- [x] Documentación completa, changelog obligatorio y `check_docs.py`.
- [ ] Autoconciencia de ruta/código con `MANIFEST.json` y `read_own_code`.
      `read_own_code` ya existe; falta el manifiesto y la indexación completa.
- [x] Tests mínimos por módulo y scripts de instalación/arranque.

## 9. Orden de trabajo inmediato

1. Paso 3: núcleo local completo y probado. Falta la aceptación real en la nube
   (QLoRA sobre Qwen3.5-2B → GGUF → comparación → activar/rollback). El GGUF
   `nitshio-v2` de Phi ya existe, pero no es la versión Qwen de esta fase.
2. No iniciar el Paso 4 hasta que el usuario confirme el cierre del Paso 3.
3. Mantener `pytest -q` + `check_docs.py` como gate de cada subfase.
4. La confirmación humana de `run.py --mode voice` queda como prueba opcional
   cuando el usuario quiera hablar por el micrófono.
