#!/usr/bin/env python
"""Genera ejemplos sintéticos para el dataset de Nitshio con Ollama.

El generador pide lotes por emoción, valida cada ejemplo con el mismo esquema
del dataset y descarta duplicados exactos o casi duplicados. No entrena nada:
solo prepara datos para el fine-tune en la nube.

Ejemplos:
    python training/generate_synthetic.py --per-emotion 4 --dry-run
    python training/generate_synthetic.py --emotion angry joy --per-emotion 8
    python training/generate_synthetic.py --append --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.config import cargar_config, host_ollama
from brain.llm.ollama_client import OllamaClient, OllamaError
from brain.llm.parser import ParseError, extract_json_object, repair_json_text
from brain.persona.loader import cargar_emociones, cargar_persona
from training.schema import (
    DatasetError,
    DatasetExample,
    contar_emociones,
    hash_examples,
    load_dataset,
    merge_examples,
    parse_example,
    save_jsonl,
)


class SyntheticError(RuntimeError):
    """No se pudo generar o validar el lote sintético."""


def _ruta(valor: str | Path) -> Path:
    path = Path(valor)
    return path if path.is_absolute() else ROOT / path


def _schema_lote() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "user": {"type": "string"},
                        "response": {"type": "string"},
                        "emotion": {"type": "string"},
                    },
                    "required": ["user", "response", "emotion"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["examples"],
        "additionalProperties": False,
    }


def extraer_ejemplos(crudo: str) -> list[dict[str, Any]]:
    """Extrae la lista de ejemplos de una respuesta del generador."""
    texto = str(crudo or "").strip()
    if not texto:
        raise SyntheticError("El generador devolvió una respuesta vacía.")
    if texto.lstrip().startswith("["):
        objeto = texto
    else:
        try:
            objeto, _ = extract_json_object(texto)
        except ParseError:
            objeto = texto
    try:
        data = json.loads(objeto)
    except json.JSONDecodeError:
        try:
            data = json.loads(repair_json_text(objeto))
        except (json.JSONDecodeError, TypeError, ParseError) as exc:
            raise SyntheticError(f"El lote no es JSON válido: {exc}") from exc

    if isinstance(data, list):
        crudos = data
    elif isinstance(data, dict):
        crudos = data.get("examples")
    else:
        raise SyntheticError("El lote debe ser un objeto con 'examples' o una lista.")
    if not isinstance(crudos, list):
        raise SyntheticError("El campo 'examples' no es una lista.")
    return [item for item in crudos if isinstance(item, dict)]


def _prompt_lote(
    emocion: str,
    cantidad: int,
    temas: Iterable[str],
    evitar: Iterable[str],
) -> list[dict[str, str]]:
    temas_texto = "; ".join(temas) or "conversación cotidiana"
    evitar_texto = "\n".join(f"- {texto}" for texto in evitar) or "- (ninguno)"
    system = (
        "Eres un generador de datos de entrenamiento para Nitshio, una IA "
        "tsundere que habla español mexicano coloquial. Nitshio insulta con "
        "cariño (pendejo, idiota, tonto, menso, cabrón), pero nunca usa "
        "discriminación, contenido sexual explícito, amenazas reales ni "
        "humillación cruel. Cuando el usuario está triste, primero consuela y "
        "después responde con su carácter. Devuelve SOLO JSON válido."
    )
    user = (
        f"Genera exactamente {cantidad} ejemplos nuevos de conversación.\n"
        f"Emoción obligatoria para todos: {emocion}.\n"
        f"Temas permitidos: {temas_texto}.\n"
        "Cada ejemplo debe tener las claves exactas 'user', 'response' y "
        "'emotion'. 'user' es lo que dice el usuario; 'response' es la "
        "respuesta de Nitshio (1 a 3 frases, en personaje); 'emotion' debe ser "
        f"'{emocion}'.\n"
        "No repitas estas entradas:\n"
        f"{evitar_texto}\n"
        'Formato: {"examples":[{"user":"...","response":"...","emotion":"..."}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _pedir_lote(
    client: OllamaClient,
    model: str,
    emocion: str,
    cantidad: int,
    *,
    temas: Iterable[str],
    evitar: Iterable[str],
    options: dict[str, Any],
    intentos: int = 2,
) -> list[dict[str, Any]]:
    mensajes = _prompt_lote(emocion, cantidad, temas, evitar)
    ultimo_error: Exception | None = None
    for intento in range(1, max(1, intentos) + 1):
        data = client.chat(
            model,
            mensajes,
            schema=_schema_lote(),
            options=options,
        )
        mensaje = data.get("message")
        crudo = (
            str(mensaje.get("content", ""))
            if isinstance(mensaje, dict)
            else str(data.get("response", ""))
        )
        try:
            return extraer_ejemplos(crudo)
        except SyntheticError as exc:
            ultimo_error = exc
            if intento < intentos:
                mensajes = mensajes + [
                    {"role": "assistant", "content": crudo},
                    {
                        "role": "user",
                        "content": (
                            "Tu respuesta anterior no cumplió el formato. "
                            "Devuelve SOLO un objeto JSON con la clave "
                            "'examples' y una lista de ejemplos válidos."
                        ),
                    },
                ]
    raise SyntheticError(f"No se pudo obtener un lote válido: {ultimo_error}")


def _resolver_modelo(config: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    ollama_models = (config.get("ollama") or {}).get("models") or {}
    synthetic_cfg = (config.get("training") or {}).get("synthetic") or {}
    rol = str(synthetic_cfg.get("model_role", "active"))
    if rol == "active":
        modelo = ollama_models.get("active")
    elif rol == "base":
        modelo = ollama_models.get("base")
    else:
        modelo = ollama_models.get(rol)
    if not modelo:
        raise SyntheticError(
            f"No se pudo resolver el modelo para el rol {rol!r}. "
            "Revisa ollama.models en config/config.yaml."
        )
    return str(modelo)


def generar(
    config: dict[str, Any],
    *,
    emociones: list[str],
    per_emotion: int,
    batch_size: int,
    max_rounds: int,
    model: str | None = None,
    client: OllamaClient | None = None,
    seed_examples: list[DatasetExample] | None = None,
    output_examples: list[DatasetExample] | None = None,
    dry_run: bool = False,
    output: Path | str | None = None,
    append: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Genera ejemplos y devuelve un resumen con los datos escritos."""
    synthetic_cfg = (config.get("training") or {}).get("synthetic") or {}
    persona = cargar_persona(config)
    emociones_permitidas = list(persona.emotions)
    for emocion in emociones:
        if emocion not in emociones_permitidas:
            raise SyntheticError(
                f"Emoción {emocion!r} no declarada. Permitidas: "
                + ", ".join(emociones_permitidas)
            )
    if per_emotion <= 0:
        raise SyntheticError("per_emotion debe ser mayor que cero.")
    if batch_size <= 0:
        raise SyntheticError("batch_size debe ser mayor que cero.")
    if max_rounds <= 0:
        raise SyntheticError("max_rounds debe ser mayor que cero.")

    modelo = _resolver_modelo(config, model)
    cliente = client or OllamaClient(
        host_ollama(config),
        timeout=float((config.get("ollama") or {}).get("request_timeout_seconds", 180)),
        keep_alive=str((config.get("ollama") or {}).get("keep_alive", "30m")),
        think=bool((config.get("ollama") or {}).get("think", False)),
    )
    opciones_base = dict(
        ((config.get("ollama") or {}).get("options") or {}).get("primary") or {}
    )
    opciones_base["temperature"] = float(synthetic_cfg.get("temperature", 0.9))
    opciones_base["num_predict"] = max(512, int(batch_size) * 140)

    temas = list(synthetic_cfg.get("themes") or [])
    umbral = float(synthetic_cfg.get("near_duplicate_threshold", 0.9))
    semilla = list(seed_examples or [])
    previos_salida = list(output_examples or []) if append else []
    base = merge_examples(semilla, previos_salida, near_duplicate_threshold=umbral)
    usuarios_evitar = [ejemplo.user for ejemplo in base[-30:]]
    nuevos: list[DatasetExample] = []
    por_emocion: dict[str, int] = {emocion: 0 for emocion in emociones}
    descartados = 0
    errores: list[str] = []
    inicio = time.perf_counter()

    for emocion in emociones:
        objetivo = int(per_emotion)
        for ronda in range(1, max_rounds + 1):
            faltan = objetivo - por_emocion[emocion]
            if faltan <= 0:
                break
            cantidad = min(int(batch_size), faltan)
            try:
                lote = _pedir_lote(
                    cliente,
                    modelo,
                    emocion,
                    cantidad,
                    temas=temas,
                    evitar=usuarios_evitar[-30:],
                    options=opciones_base,
                )
            except (OllamaError, SyntheticError) as exc:
                errores.append(f"{emocion} ronda {ronda}: {exc}")
                if not quiet:
                    print(f"  aviso: {errores[-1]}", file=sys.stderr)
                continue
            validos: list[DatasetExample] = []
            for crudo in lote:
                try:
                    ejemplo = parse_example(crudo, emociones_permitidas)
                except DatasetError as exc:
                    descartados += 1
                    errores.append(f"{emocion}: ejemplo descartado ({exc})")
                    continue
                if ejemplo.emotion != emocion:
                    descartados += 1
                    errores.append(
                        f"{emocion}: el generador devolvió "
                        f"{ejemplo.emotion!r}; se descarta."
                    )
                    continue
                validos.append(ejemplo)
            combinados = merge_examples(
                base, nuevos, validos, near_duplicate_threshold=umbral
            )
            antes = len(base) + len(nuevos)
            aceptados = combinados[antes:]
            # ``merge_examples`` devuelve la lista completa; los aceptados son
            # los que quedaron al final. Recalculamos para evitar duplicados.
            nuevos = combinados[len(base) :]
            por_emocion[emocion] = sum(
                1 for ejemplo in nuevos if ejemplo.emotion == emocion
            )
            descartados += len(validos) - len(aceptados)
            usuarios_evitar.extend(ejemplo.user for ejemplo in aceptados)
            if not quiet:
                print(
                    f"  {emocion}: +{len(aceptados)} "
                    f"({por_emocion[emocion]}/{objetivo})",
                    file=sys.stderr,
                )
            if not aceptados:
                # Evita gastar rondas si el modelo se quedó atascado.
                if ronda >= 2:
                    break

    combinados_finales = merge_examples(
        base, nuevos, near_duplicate_threshold=umbral
    )
    nuevos_finales = combinados_finales[len(base) :]
    resumen: dict[str, Any] = {
        "model": modelo,
        "generated": len(nuevos_finales),
        "discarded": descartados,
        "by_emotion": contar_emociones(nuevos_finales),
        "total_dataset": len(combinados_finales),
        "hash": hash_examples(combinados_finales),
        "duration_seconds": round(time.perf_counter() - inicio, 2),
        "errors": errores,
        "dry_run": dry_run,
    }
    if dry_run:
        resumen["examples"] = [ejemplo.to_record() for ejemplo in nuevos_finales]
        resumen["dataset_examples"] = [
            ejemplo.to_record() for ejemplo in combinados_finales
        ]
        return resumen

    ruta_salida = _ruta(
        output or synthetic_cfg.get("output", "training/datasets/synthetic.jsonl")
    )
    save_jsonl(ruta_salida, combinados_finales)
    manifest = ruta_salida.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                **resumen,
                "output": str(ruta_salida),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "per_emotion_target": per_emotion,
                "seed_examples": len(semilla),
                "append": append,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    resumen["output"] = str(ruta_salida)
    resumen["manifest"] = str(manifest)
    return resumen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera datos sintéticos para el fine-tune de Nitshio."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default=None, help="Tag de Ollama a usar.")
    parser.add_argument(
        "--emotion",
        nargs="+",
        default=None,
        help="Emociones a generar (default: todas las declaradas).",
    )
    parser.add_argument("--per-emotion", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed-file", type=Path, default=None)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Conserva los ejemplos que ya estén en el archivo de salida.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = cargar_config(args.config)
        persona = cargar_persona(config)
        emociones = args.emotion or list(persona.emotions)
        synthetic_cfg = (config.get("training") or {}).get("synthetic") or {}
        seed_path = _ruta(
            args.seed_file
            or synthetic_cfg.get("seed_file", "training/datasets/seed.jsonl")
        )
        seed = load_dataset(seed_path, persona.emotions).examples if seed_path.exists() else []
        output_path = _ruta(
            args.output
            or synthetic_cfg.get("output", "training/datasets/synthetic.jsonl")
        )
        output_examples = (
            load_dataset(output_path, persona.emotions).examples
            if args.append and output_path.exists()
            else []
        )
        resumen = generar(
            config,
            emociones=list(emociones),
            per_emotion=int(args.per_emotion or synthetic_cfg.get("per_emotion", 8)),
            batch_size=int(args.batch_size or synthetic_cfg.get("batch_size", 4)),
            max_rounds=int(args.max_rounds or synthetic_cfg.get("max_rounds", 4)),
            model=args.model,
            seed_examples=seed,
            output_examples=output_examples,
            dry_run=args.dry_run,
            output=args.output,
            append=args.append,
            quiet=args.quiet,
        )
    except (OSError, ValueError, DatasetError, SyntheticError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(resumen, ensure_ascii=False, indent=2))
    else:
        print("=" * 64)
        print("  NITSHIO — generación sintética")
        print("=" * 64)
        print(f"Modelo: {resumen['model']}")
        print(f"Nuevos: {resumen['generated']} (descartados: {resumen['discarded']})")
        print(f"Dataset total: {resumen['total_dataset']}")
        print("Por emoción:", json.dumps(resumen["by_emotion"], ensure_ascii=False))
        if resumen.get("output"):
            print(f"Archivo: {resumen['output']}")
        if resumen.get("errors"):
            print(f"Avisos: {len(resumen['errors'])}")
    return 0 if resumen["generated"] > 0 or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
