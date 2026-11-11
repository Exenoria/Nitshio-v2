# Entrenamiento — Nitshio

> **Estado: Fase 0 migrada a Qwen3.5-2B; Paso 3 implementado en local y
> aceptación final pendiente del fine-tune real en Kaggle/Colab.** Todo el
> pipeline local (dataset combinado de 348 ejemplos, generador sintético,
> evaluación, versionado, UI, notebook de Colab y carga de GGUF con `--gguf`)
> está implementado y probado. `nitshio-qwen2b` está activo y evaluado;
> `nitshio-v2` existe como GGUF de Phi-3.5, pero falta el reentrenamiento sobre
> Qwen3.5-2B. La T1200 de 4 GB no entrena: el QLoRA se ejecuta en la nube y el
> GGUF resultante se carga en Ollama local.

## 1. Decisión principal

El fine-tune se hace **solo en la nube** (Kaggle o Colab con GPU gratuita).
Localmente solo se hace inferencia con Ollama y se recibe el GGUF resultante.

Motivos:

- QLoRA de Qwen3.5-2B (o de un Phi-3.5 de 3.8B) no cabe con comodidad en 4 GB.
- El entrenamiento local bloquea la máquina durante horas.
- Kaggle/Colab ofrecen GPU gratuita suficiente para QLoRA.

## 2. Flujo completo

```text
dataset JSONL (seed + synthetic + corrections)
   │
   ▼
training/validate_dataset.py  ──► errores, duplicados y split estratificado
   │
   ▼
training/generate_synthetic.py ──► lotes por emoción + deduplicación + manifiesto
   │
   ▼
training/finetune_qlora.py (Kaggle/Colab) ──► adaptadores LoRA + manifiesto
   │
   ▼
training/export_gguf.py ──► fusión LoRA + GGUF f16 + cuantización q4_K_M
   │
   ▼
ollama create nitshio-vN -f models/Modelfile
   │
   ▼
training/evaluate.py ──► reporte antes/después (Markdown + JSON)
   │
   ▼
training/versions.py ──► activar / rollback
```

## 3. Formato del dataset

Cada línea es un objeto JSON estricto con tres claves, sin campos extra:

```json
{"user":"Hola, ¿quién eres?","response":"Hmph. Soy Nitshio, no que te importe.","emotion":"angry"}
```

Reglas de validación (`training/schema.py`):

- JSON válido, una línea por ejemplo; se ignoran líneas en blanco.
- `user` no vacío y con longitud máxima.
- `response` no vacío y con longitud razonable.
- `emotion` declarada en `config/emotions.yaml`.
- Sin claves adicionales (`extra="forbid"`).
- Duplicados exactos y casi duplicados descartados y contados.
- Split determinista y estratificado por emoción (train/val/test).

### Datasets actuales

| Archivo | Ejemplos | Distribución | Uso |
|---|---:|---|---|
| `training/datasets/seed.jsonl` | 348 | 58 por emoción | Semilla completa (48 originales + 300 importados) |
| `training/datasets/nitshio_dataset_300.jsonl` | 300 | 50 por emoción | Lote importado, validado aparte |
| `training/datasets/train.jsonl` | 348 | 58 por emoción | Combinado `seed + 300`, deduplicado |

`seed.jsonl` ya contiene los 300 ejemplos importados, así que al combinar
`seed + nitshio_dataset_300.jsonl` el validador descarta 300 duplicados y deja
**348 ejemplos únicos**. No es un error: el combinado existe para que el
notebook de Colab sea reproducible aunque el zip incluya ambos archivos.

Validación real:

```powershell
# Lote de 300
.\.venv\Scripts\python.exe training\validate_dataset.py training\datasets\nitshio_dataset_300.jsonl --strict --json

# Combinado final (escribe train.jsonl deduplicado)
.\.venv\Scripts\python.exe training\validate_dataset.py data\cache\combined_raw.jsonl `
  --output training\datasets\train.jsonl --json
```

Resultados:

- `nitshio_dataset_300.jsonl`: 300 ejemplos, 0 errores, 0 duplicados, 50 por
  emoción, hash
  `465694d3bc9d5a40cb92b83d1e8e0cd909acbdbc61786591e1f4755139941c2e`.
- `train.jsonl`: 348 ejemplos, 0 errores, 0 duplicados, 58 por emoción, hash
  `8e96d86b1293b107a0f22b313596fa0813b92692537b51c92a5f1d4b005a8108`.

### Particiones

```powershell
.\.venv\Scripts\python.exe training\validate_dataset.py training\datasets\train.jsonl `
  --strict --split-dir training\datasets\splits --json
```

Con ratios 0.8/0.1/0.1 y `seed=42` el reparto real del combinado es **276
train / 36 val / 36 test**. El split es determinista y garantiza al menos un
ejemplo por emoción en cada partición activa cuando el grupo tiene 3 o más
ejemplos. `training/datasets/train.jsonl` es el dataset completo; el train real
que consume el fine-tune es `training/datasets/splits/train.jsonl`.

## 4. Generación sintética

`training/generate_synthetic.py` pide lotes JSON a Ollama, valida cada ejemplo
con el mismo esquema del dataset y descarta duplicados exactos y casi
duplicados. No entrena: solo prepara datos.

```powershell
# Prueba sin escribir archivos
.\.venv\Scripts\python.exe training\generate_synthetic.py --per-emotion 4 --dry-run

# Generar todas las emociones y añadir a lo existente
.\.venv\Scripts\python.exe training\generate_synthetic.py --append --json

# Solo dos emociones
.\.venv\Scripts\python.exe training\generate_synthetic.py --emotion angry joy --per-emotion 8
```

La configuración vive en `training.synthetic` de `config/config.yaml`:
modelo por rol (`active`/`base`), cantidad por emoción, tamaño de lote, rondas,
temperatura, umbral de casi duplicado y temas permitidos.

Salidas:

- `training/datasets/synthetic.jsonl`
- `training/datasets/synthetic.manifest.json` (modelo, conteos, hash, errores)

## 5. Fine-tune QLoRA (solo nube)

Parámetros de partida en `config/config.yaml`:

```yaml
training:
  hf_model: "Qwen/Qwen3.5-2B"
  qlora:
    max_seq_length: 2048
    load_in_4bit: true
    lora_r: 16
    lora_alpha: 16
    lora_dropout: 0.0
    batch_size: 2
    grad_accum: 4
    epochs: 3
    learning_rate: 0.0002
```

`training/finetune_qlora.py` usa Unsloth + TRL y guarda:

- adaptadores LoRA y tokenizador;
- `training_manifest.json` con dataset, split, hiperparámetros, métricas y GPU.

Dry-run local (no requiere GPU ni dependencias de entrenamiento):

```powershell
.\.venv\Scripts\python.exe training\finetune_qlora.py `
  --dataset training\datasets\train.jsonl --dry-run `
  --output-dir data\cache\train_dryrun_348
```

Resultado real: 348 ejemplos, **312 train / 36 val** (split interno 90/10),
base `Qwen/Qwen3.5-2B`, QLoRA 4-bit, LoRA r=16, 3 épocas.

### Notebook Colab

`training/cloud/Nitshio_QLoRA.ipynb` es el punto de entrada en Colab:

1. Verifica GPU con `nvidia-smi`.
2. Localiza el proyecto: `NITSHIO_WORKDIR` (`/content` por defecto) y
   `NITSHIO_DIR` (`/content/Nitshio`). Si no existe, descomprime los `*.zip`
   subidos a `/content` y detecta la carpeta que contiene `config/config.yaml`.
3. Instala `requirements-train.txt`.
4. Combina `seed.jsonl` + `nitshio_dataset_300.jsonl` en
   `training/datasets/train.jsonl`, deduplicado con el validador.
5. Valida con `--strict` y genera el split 80/10/10 en
   `training/datasets/splits/`.
6. Ejecuta `training/finetune_qlora.py` sobre el combinado.
7. Clona y compila `llama.cpp` con `-DGGML_CUDA=OFF` (solo conversión y
   cuantización).
8. Ejecuta `training/export_gguf.py` y empaqueta el GGUF para descargarlo.

El notebook no hardcodea modelos: lee `training.hf_model`, `training.qlora` y la
cuantización de `config/config.yaml`. En Kaggle se puede ajustar
`NITSHIO_WORKDIR` a `/kaggle/working` y `NITSHIO_DIR` a la carpeta del proyecto.

## 6. Exportación a GGUF y Ollama

```powershell
.\.venv\Scripts\python.exe training\export_gguf.py `
  --adapter-dir /kaggle/working/nitshio-lora `
  --llama-cpp-dir /kaggle/working/llama.cpp `
  --output-dir /kaggle/working/nitshio-gguf
```

Pasos:

1. Fusionar los adaptadores con el modelo base (`peft`).
2. Convertir a GGUF f16 con `convert_hf_to_gguf.py`.
3. Cuantizar a `q4_K_M` con `llama-quantize`.
4. Escribir `export_manifest.json` con hash y tamaño del GGUF.

Dry-run local validado:

```powershell
.\.venv\Scripts\python.exe training\export_gguf.py --adapter-dir training --dry-run
```

Devuelve base `Qwen/Qwen3.5-2B` y cuantización `q4_K_M`.

Después, en local, hay dos caminos:

```powershell
# A) Crear el tag a mano desde el Modelfile ya renderizado
ollama create nitshio-v2 -f models/Modelfile

# B) Recomendado: renderizar FROM <gguf>, registrar type=gguf y crear/activar
.\.venv\Scripts\python.exe training\build_model.py `
  --version nitshio-v2 --gguf C:\ruta\nitshio-v2.gguf --create --activate
```

`--gguf` valida que el archivo exista y sea `.gguf`, escribe el `FROM` con la
ruta absoluta en formato POSIX (barras normales; entre comillas si hay
espacios), conserva el `TEMPLATE` de Phi-3.5 del template y registra la versión
con `type=gguf` y `gguf_path` en `models/versions.json`. El campo `base` sigue
guardando el tag base configurado para poder comparar y revertir.

Para solo revisar el Modelfile sin escribir ni registrar nada:

```powershell
.\.venv\Scripts\python.exe training\build_model.py `
  --version nitshio-v2 --gguf C:\ruta\nitshio-v2.gguf --print
```

El nombre del modelo se registra en `models/versions.json`, nunca en el código.

## 7. Versionado y rollback

El registro canónico es `training/versions.py`; `models/versions.json` guarda
por versión:

- nombre del tag de Ollama;
- modelo base;
- fecha;
- hash de la persona y del Modelfile;
- dataset usado (hash);
- métricas de evaluación;
- notas y estado.

```powershell
# Listar versiones
.\.venv\Scripts\python.exe training\versions.py list

# Activar una versión (actualiza config.yaml sin perder comentarios)
.\.venv\Scripts\python.exe training\versions.py activate nitshio-v2

# Volver a la anterior
.\.venv\Scripts\python.exe training\versions.py rollback

# Adjuntar métricas manualmente (objeto JSON)
.\.venv\Scripts\python.exe training\versions.py metrics nitshio-v2 metricas.json
```

La activación reescribe solo la línea `ollama.models.active` y conserva el
resto del YAML. El rollback usa `previous_active`.

## 8. Evaluación

Set fijo de **18 prompts** en `training/eval/prompts.yaml`, con emoción
esperada y respuesta de referencia.

```powershell
# Evaluar el modelo activo
.\.venv\Scripts\python.exe training\evaluate.py

# Comparar activo contra base y guardar métricas
.\.venv\Scripts\python.exe training\evaluate.py --compare-base --record-metrics

# Sin juez (más rápido y determinista)
.\.venv\Scripts\python.exe training\evaluate.py --compare-base --no-judge
```

Métricas:

- **JSON válido:** porcentaje de respuestas parseables.
- **Emoción correcta:** coincidencia con la emoción esperada.
- **Consistencia de personaje:** juez opcional con el modelo router.
- **Similitud semántica:** embeddings de Ollama; fallback léxico si fallan.
- **Longitud, latencia y tokens/s.**

El reporte Markdown y el JSON se guardan en `training/eval/reports` y comparan
las versiones lado a lado.

> **Rendimiento:** con `num_ctx: 2048` y los few-shots en runtime, cada prompt
> ronda los 2000 tokens. La evaluación completa de 18 prompts puede tardar
> decenas de minutos en la T1200. Para una prueba rápida usa `--limit 5`. En la
> nube, el modelo fine-tuneado se evalúa con el mismo set y la misma
> configuración.

### Resultado real (2026-11-09 05:13 UTC)

Comparación ejecutada con `training/evaluate.py --compare-base --no-judge`,
18 prompts y embeddings reales de Ollama (sin juez):

| Modelo | JSON válido | Emoción | Similitud | Latencia media | Tok/s | Score |
|---|---:|---:|---:|---:|---:|---:|
| `nitshio-v1` | 100.0% | 33.3% | 0.477 | 78.5 s | 5.83 | **0.5452** |
| `phi3.5:3.8b-mini-instruct-q4_K_M` | 94.4% | 5.6% | 0.451 | 91.7 s | 6.72 | 0.4289 |

`nitshio-v1` gana por **0.116 puntos**, con más JSON válido y casi seis veces
más aciertos de emoción que el base. La calidad del texto sigue siendo baja:
las respuestas son incoherentes y la emoción acierta solo un tercio de las
veces. Eso confirma que el evaluador discrimina y que hace falta el fine-tune
real.

El reporte completo está en
`training/eval/reports/eval_20261109_051355.md` (y `.json`). Las métricas de
`nitshio-v1` quedaron adjuntas en `models/versions.json`.

### Resultado real de la migración a Qwen3.5-2B (2026-11-11)

Mismos 18 prompts, embeddings reales y sin juez:

| Modelo | JSON válido | Emoción | Similitud | Latencia media | Tok/s | Score |
|---|---:|---:|---:|---:|---:|---:|
| `nitshio-qwen2b` | 100.0% | 44.4% | 0.673 | 2.74 s | 33.10 | **0.6695** |
| `nitshio-v1` | 100.0% | 61.1% | 0.660 | 10.43 s | 13.74 | **0.7192** |

`nitshio-qwen2b` es más rápido (2.4× en tokens/s y 3.8× menos latencia), pero
todavía pierde 0.05 puntos de score por aciertos de emoción. La migración de
Fase 0 no busca mejorar calidad, sino dejar el pipeline sobre Qwen con
`think=false`; el fine-tune del Paso 3 debe recuperar la precisión emocional.
Reporte: `training/eval/reports/eval_20261111_115248.md`.

### Por qué la evaluación marca 78 s y 5.8 tok/s

La última medición de `scripts/check_models.py --json` dio **17.5 tok/s** para
`nitshio-v1` (la medición previa era 15.6 tok/s), mientras que el evaluador
reportó **5.83 tok/s** y **78.5 s** por prompt. No es una regresión: miden
cargas distintas.

| Condición | `check_models.py` | `evaluate.py` |
|---|---|---|
| Prompt | 1026 tokens en la última corrida | ~2029 tokens (system + few-shots + usuario) |
| Salida | 29 tokens (tope 64) | Hasta 512 tokens |
| Contexto | 1026 + 29 dentro de `num_ctx: 2048` | ~2029 + salida supera los 2048 |
| Peticiones | Una, con calentamiento previo | 18 seguidas, sin calentamiento por prompt |
| Tok/s | Solo decodificación (`eval_count / eval_duration`) | Solo decodificación, bajo presión de contexto |
| Latencia | 8.12 s de pared | 78.5 s de pared (prompt + generación + red/parseo) |

El factor principal es el **contexto**: el prompt de evaluación roza los 2048
tokens y la respuesta intenta seguir generando, así que llama.cpp tiene que
desplazar la ventana y reprocesar. Eso baja el ritmo de decodificación y dispara
el tiempo de pared. El modelo base sufre lo mismo (6.72 tok/s, 91.7 s), lo que
confirma que no es un problema de `nitshio-v1` sino de la configuración
`num_ctx: 2048` con prompts largos en una GPU de 4 GB.

Para comparar versiones, usa siempre `evaluate.py` con el mismo set y la misma
configuración. `check_models.py` sirve para comprobar que Ollama y el hardware
responden, no para predecir la latencia de una conversación con few-shots.

> **Nota:** el base agotó una vez el timeout de 180 s en `angry-celosa`; el
> evaluador lo registró como error de red y continuó con el resto. No invalida
> la comparación, pero conviene tenerlo en cuenta al leer la latencia media.

## 9. UI Gradio

```powershell
.\.venv\Scripts\python.exe training\ui.py
# http://127.0.0.1:7860
```

Pestañas:

- **Chat:** prueba el modelo activo, muestra emoción/intentos/fallback, botones
  👍/👎 y corrección de respuesta/emoción.
- **Dataset:** carga, valida y guarda JSONL; no escribe si hay errores.
- **Versiones:** tabla, activar, rollback y reconstrucción del Modelfile con
  opción de `ollama create`.
- **Nube:** muestra el comando de entrenamiento y el notebook; no lanza QLoRA
  en local.

Las correcciones se guardan en `training/datasets/corrections.jsonl` y también
en ChromaDB (colección `memories`) para que Nitshio las recuerde sin
reentrenar.

## 10. Estado real y pendientes

Implementado y probado en local:

- [x] Esquema, validación, deduplicación, hashes y split estratificado.
- [x] Datasets validados: 300 importados, 348 combinados y split 276/36/36.
- [x] Generador sintético con lotes por emoción y manifiesto.
- [x] Evaluador con embeddings, fallback léxico, juez opcional y reportes.
- [x] Evaluación real de `nitshio-v1` contra el base: 0.5452 vs 0.4289.
- [x] Análisis de latencia: 78.5 s de pared y 5.83 tok/s bajo contexto lleno.
- [x] Métricas adjuntadas a `models/versions.json`.
- [x] Registro de versiones, activación y rollback seguros.
- [x] Construcción de Modelfile y `ollama create`.
- [x] Carga de un GGUF local con `--gguf` y registro `type=gguf`.
- [x] `nitshio-v2` generado como GGUF de Phi-3.5 y registrado en Ollama local.
- [x] Scripts de fine-tune QLoRA y exportación GGUF con dry-run.
- [x] Dry-run de QLoRA y exportación con la base `Qwen/Qwen3.5-2B`.
- [x] Notebook Colab con `/content`, unzip del proyecto y dataset combinado.
- [x] UI Gradio construida y arrancada en local.
- [x] Tests de training en la suite general (275 tests totales).
- [x] Migración a Qwen3.5-2B evaluada con los mismos 18 prompts.

Pendiente para cerrar la aceptación del Paso 3:

- [ ] Reentrenar sobre Qwen3.5-2B en Kaggle/Colab y obtener el GGUF nuevo.
- [ ] Crear la versión Qwen en Ollama local.
- [ ] Comparar esa versión contra `nitshio-v1` y el base con el evaluador real.
- [ ] Activar la versión Qwen y probar rollback con el GGUF real.
- [ ] Decidir el umbral de mejora aceptable.

## 11. Solución rápida de problemas

| Síntoma | Causa probable | Acción |
|---|---|---|
| `ModuleNotFoundError: gradio` | UI sin dependencias | `pip install -r requirements.txt` |
| `Faltan dependencias de entrenamiento` | Script cloud ejecutado en local | Usar Kaggle/Colab o solo `--dry-run` |
| `No se encontró convert_hf_to_gguf.py` | `llama.cpp` no clonado | Pasar `--llama-cpp-dir` o ejecutar el notebook |
| `No existe el GGUF` | Ruta incorrecta o archivo no descargado | Verificar la ruta y que termine en `.gguf` |
| Dataset con errores | JSON/emoción/campos inválidos | `training/validate_dataset.py --strict` |
| Split desbalanceado | Grupo pequeño por emoción | Revisar conteos; el split garantiza cobertura mínima |
| `ollama create` falla | GGUF o Modelfile incorrectos | Revisar `models/Modelfile` y la ruta del GGUF |

Ver también [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
