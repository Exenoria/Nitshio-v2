#!/usr/bin/env python
"""Evalúa modelos de Nitshio contra un set fijo de prompts.

Mide JSON válido, emoción, similitud semántica, longitud, latencia y,
opcionalmente, una nota de personalidad del modelo juez. Genera un reporte
Markdown lado a lado y un JSON con todos los detalles.

Ejemplos:
    python training/evaluate.py --compare-base
    python training/evaluate.py --models nitshio-v1 nitshio-v2 --no-judge
    python training/evaluate.py --models nitshio-v2 --record-metrics
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.config import cargar_config, host_ollama
from brain.llm.ollama_client import OllamaClient, OllamaError
from brain.persona.loader import Persona, cargar_persona, construir_mensajes
from memory.embeddings import EmbeddingError, crear_embeddings
from training.versions import find_version, load_registry, update_metrics


class EvaluationError(RuntimeError):
    """No se pudo cargar el set o ejecutar la evaluación."""


@dataclass(frozen=True)
class EvalPrompt:
    """Un prompt fijo con emoción esperada y respuesta de referencia."""

    id: str
    user: str
    expected_emotion: str
    reference: str
    tags: tuple[str, ...] = ()


@dataclass
class PromptResult:
    """Resultado de un prompt para un modelo."""

    prompt_id: str
    user: str
    expected_emotion: str
    reference: str
    response: str
    emotion: str
    json_valid: bool
    emotion_correct: bool
    similarity: float = 0.0
    similarity_backend: str = ""
    length_chars: int = 0
    latency_seconds: float = 0.0
    tokens_per_second: float | None = None
    personality: float | None = None
    personality_reason: str = ""
    attempts: int = 0
    repaired: bool = False
    fallback: bool = False
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "user": self.user,
            "expected_emotion": self.expected_emotion,
            "reference": self.reference,
            "response": self.response,
            "emotion": self.emotion,
            "json_valid": self.json_valid,
            "emotion_correct": self.emotion_correct,
            "similarity": round(self.similarity, 4),
            "similarity_backend": self.similarity_backend,
            "length_chars": self.length_chars,
            "latency_seconds": round(self.latency_seconds, 4),
            "tokens_per_second": (
                round(self.tokens_per_second, 2)
                if self.tokens_per_second is not None
                else None
            ),
            "personality": (
                round(self.personality, 4)
                if self.personality is not None
                else None
            ),
            "personality_reason": self.personality_reason,
            "attempts": self.attempts,
            "repaired": self.repaired,
            "fallback": self.fallback,
            "error": self.error,
        }


@dataclass
class EvaluationResult:
    """Resultado agregado de un modelo."""

    model: str
    started_at: str
    duration_seconds: float
    prompts: list[PromptResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    judge_model: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "started_at": self.started_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "judge_model": self.judge_model,
            "metrics": self.metrics,
            "errors": self.errors,
            "prompts": [resultado.as_dict() for resultado in self.prompts],
        }


def _ruta(valor: str | Path) -> Path:
    path = Path(valor)
    return path if path.is_absolute() else ROOT / path


def _normalizar(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", str(texto).casefold())
    sin_acentos = "".join(
        caracter for caracter in descompuesto if not unicodedata.combining(caracter)
    )
    return " ".join(
        "".join(c if c.isalnum() else " " for c in sin_acentos).split()
    )


def cargar_prompts(
    path: Path | str,
    emociones_permitidas: Iterable[str],
) -> list[EvalPrompt]:
    """Carga y valida ``training/eval/prompts.yaml``."""
    ruta = _ruta(path)
    if not ruta.exists():
        raise EvaluationError(f"No existe el set de evaluación: {ruta}")
    try:
        data = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise EvaluationError(f"No se pudo leer {ruta}: {exc}") from exc
    crudos = data.get("prompts") if isinstance(data, dict) else None
    if not isinstance(crudos, list) or not crudos:
        raise EvaluationError(f"{ruta} no contiene una lista 'prompts'.")
    permitidas = set(emociones_permitidas)
    prompts: list[EvalPrompt] = []
    ids: set[str] = set()
    for indice, item in enumerate(crudos, start=1):
        if not isinstance(item, dict):
            raise EvaluationError(f"Prompt {indice}: se esperaba un objeto.")
        identificador = str(item.get("id", "")).strip()
        user = str(item.get("user", "")).strip()
        emocion = str(item.get("expected_emotion", "")).strip()
        reference = str(item.get("reference", "")).strip()
        if not identificador or not user or not reference:
            raise EvaluationError(
                f"Prompt {indice}: 'id', 'user' y 'reference' son obligatorios."
            )
        if identificador in ids:
            raise EvaluationError(f"Prompt duplicado: {identificador!r}.")
        if emocion not in permitidas:
            raise EvaluationError(
                f"Prompt {identificador}: emoción inválida {emocion!r}."
            )
        tags = item.get("tags") or []
        if not isinstance(tags, list):
            raise EvaluationError(
                f"Prompt {identificador}: 'tags' debe ser una lista."
            )
        ids.add(identificador)
        prompts.append(
            EvalPrompt(
                id=identificador,
                user=user,
                expected_emotion=emocion,
                reference=reference,
                tags=tuple(str(tag) for tag in tags),
            )
        )
    return prompts


def _coseno(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    producto = sum(x * y for x, y in zip(a, b))
    norma_a = math.sqrt(sum(x * x for x in a))
    norma_b = math.sqrt(sum(y * y for y in b))
    if norma_a == 0 or norma_b == 0:
        return 0.0
    return max(0.0, min(1.0, producto / (norma_a * norma_b)))


def _similitud_lexica(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalizar(a), _normalizar(b)).ratio()


def _calcular_similitudes(
    resultados: list[PromptResult],
    *,
    embedder: Any | None,
    forzar_lexica: bool = False,
) -> str:
    """Rellena ``similarity`` y devuelve el backend usado."""
    if not resultados:
        return "vacio"
    if embedder is not None and not forzar_lexica:
        textos: list[str] = []
        for resultado in resultados:
            textos.extend([resultado.response, resultado.reference])
        try:
            vectores = embedder.embed(textos)
            if len(vectores) == len(textos):
                for indice, resultado in enumerate(resultados):
                    resultado.similarity = _coseno(
                        vectores[indice * 2], vectores[indice * 2 + 1]
                    )
                    resultado.similarity_backend = "embeddings"
                return "embeddings"
        except (EmbeddingError, IndexError, ValueError, TypeError):
            pass
    for resultado in resultados:
        resultado.similarity = _similitud_lexica(
            resultado.response, resultado.reference
        )
        resultado.similarity_backend = "lexica"
    return "lexica"


def _tokens_por_segundo(chat_data: dict[str, Any] | None) -> float | None:
    if not chat_data:
        return None
    try:
        eval_count = float(chat_data.get("eval_count") or 0)
        eval_duration = float(chat_data.get("eval_duration") or 0)
    except (TypeError, ValueError):
        return None
    if eval_count <= 0 or eval_duration <= 0:
        return None
    return eval_count / (eval_duration / 1_000_000_000.0)


def _contenido(data: dict[str, Any]) -> str:
    mensaje = data.get("message")
    if isinstance(mensaje, dict):
        return str(mensaje.get("content", ""))
    return str(data.get("response", ""))


def _juzgar_personalidad(
    client: Any,
    judge_model: str,
    *,
    user: str,
    response: str,
    emotion: str,
    reference: str,
) -> tuple[float | None, str]:
    """Pide al modelo juez una nota de 1 a 5; nunca lanza excepción."""
    schema = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "reason": {"type": "string"},
        },
        "required": ["score", "reason"],
        "additionalProperties": False,
    }
    mensajes = [
        {
            "role": "system",
            "content": (
                "Eres un evaluador imparcial de diálogos de rol. Puntúa de 1 a 5 "
                "si la respuesta mantiene la personalidad tsundere mexicana, "
                "responde al usuario y no rompe el formato. 1 = fuera de "
                "personaje o incoherente; 5 = excelente. Devuelve SOLO JSON con "
                "las claves 'score' (entero) y 'reason' (texto breve)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Usuario: {user}\n"
                f"Emoción detectada: {emotion}\n"
                f"Respuesta: {response}\n"
                f"Referencia aproximada: {reference}"
            ),
        },
    ]
    try:
        data = client.chat(judge_model, mensajes, schema=schema, options={"temperature": 0.0})
        crudo = _contenido(data).strip()
        if crudo.startswith("```"):
            crudo = crudo.strip("`").removeprefix("json").strip()
        evaluado = json.loads(crudo)
        score = int(evaluado.get("score"))
        if not 1 <= score <= 5:
            raise ValueError("score fuera de rango")
        razon = str(evaluado.get("reason", "")).strip()
        return (score - 1) / 4.0, razon
    except (OllamaError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return None, f"juez no disponible: {exc}"


def _promedio(valores: Iterable[float | None]) -> float | None:
    limpios = [float(valor) for valor in valores if valor is not None]
    return statistics.fmean(limpios) if limpios else None


def calcular_metricas(
    resultados: list[PromptResult],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Agrega métricas y calcula el score compuesto configurado."""
    total = len(resultados)
    if total == 0:
        return {
            "count": 0,
            "json_valid_rate": 0.0,
            "emotion_accuracy": 0.0,
            "similarity_mean": 0.0,
            "length_mean": 0.0,
            "latency_mean_seconds": 0.0,
            "tokens_per_second_mean": None,
            "personality_mean": None,
            "composite_score": 0.0,
            "similarity_backend": "",
        }
    evaluacion = (config.get("training") or {}).get("evaluation") or {}
    json_rate = sum(1 for r in resultados if r.json_valid) / total
    emocion_rate = sum(1 for r in resultados if r.emotion_correct) / total
    similitud = statistics.fmean(r.similarity for r in resultados)
    longitud = statistics.fmean(r.length_chars for r in resultados)
    latencia = statistics.fmean(r.latency_seconds for r in resultados)
    tok_s = _promedio(r.tokens_per_second for r in resultados)
    personalidad = _promedio(r.personality for r in resultados)

    pesos = {
        "json": float(evaluacion.get("json_weight", 0.2)),
        "emotion": float(evaluacion.get("emotion_weight", 0.3)),
        "similarity": float(evaluacion.get("similarity_weight", 0.4)),
        "personality": float(evaluacion.get("personality_weight", 0.1)),
    }
    valores = {
        "json": json_rate,
        "emotion": emocion_rate,
        "similarity": similitud,
    }
    if personalidad is not None:
        valores["personality"] = personalidad
    peso_total = sum(pesos[clave] for clave in valores)
    compuesto = (
        sum(pesos[clave] * valor for clave, valor in valores.items()) / peso_total
        if peso_total > 0
        else 0.0
    )
    backends = {r.similarity_backend for r in resultados if r.similarity_backend}
    return {
        "count": total,
        "json_valid_rate": round(json_rate, 4),
        "emotion_accuracy": round(emocion_rate, 4),
        "similarity_mean": round(similitud, 4),
        "length_mean": round(longitud, 1),
        "latency_mean_seconds": round(latencia, 3),
        "tokens_per_second_mean": round(tok_s, 2) if tok_s is not None else None,
        "personality_mean": (
            round(personalidad, 4) if personalidad is not None else None
        ),
        "composite_score": round(compuesto, 4),
        "similarity_backend": "+".join(sorted(backends)),
    }


def evaluate_model(
    config: dict[str, Any],
    model: str,
    prompts: list[EvalPrompt],
    *,
    client: Any | None = None,
    embedder: Any | None = None,
    judge: bool = False,
    judge_model: str | None = None,
    forzar_lexica: bool = False,
    limit: int | None = None,
    progress: Any | None = None,
) -> EvaluationResult:
    """Evalúa un modelo y devuelve resultados por prompt y agregados."""
    persona: Persona = cargar_persona(config)
    ollama_cfg = config.get("ollama") or {}
    cliente = client or OllamaClient(
        host_ollama(config),
        timeout=float(ollama_cfg.get("request_timeout_seconds", 180)),
        keep_alive=str(ollama_cfg.get("keep_alive", "30m")),
        think=bool(ollama_cfg.get("think", False)),
    )
    opciones = dict((ollama_cfg.get("options") or {}).get("primary") or {})
    structured = ollama_cfg.get("structured_output") or {}
    retries = int(structured.get("retries", 2))
    fallback_emotion = str(
        structured.get("fallback_emotion", persona.default_emotion)
    )
    evaluacion_cfg = (config.get("training") or {}).get("evaluation") or {}
    if judge and judge_model is None:
        rol = str(evaluacion_cfg.get("judge_model_role", "router"))
        judge_model = str((ollama_cfg.get("models") or {}).get(rol, "") or "") or None
        if judge_model is None:
            judge = False

    seleccionados = prompts[: max(0, int(limit))] if limit else list(prompts)
    if not seleccionados:
        raise EvaluationError("No hay prompts para evaluar.")

    resultado = EvaluationResult(
        model=model,
        started_at=datetime.now(timezone.utc).isoformat(),
        duration_seconds=0.0,
        judge_model=judge_model if judge else None,
    )
    inicio_total = time.perf_counter()
    for indice, prompt in enumerate(seleccionados, start=1):
        mensajes = construir_mensajes(persona, prompt.user)
        inicio = time.perf_counter()
        try:
            respuesta = cliente.chat_structured(
                model,
                mensajes,
                allowed_emotions=persona.emotions,
                options=opciones,
                retries=retries,
                fallback_emotion=fallback_emotion,
            )
            latencia = time.perf_counter() - inicio
            parsed = respuesta.parsed
            item = PromptResult(
                prompt_id=prompt.id,
                user=prompt.user,
                expected_emotion=prompt.expected_emotion,
                reference=prompt.reference,
                response=parsed.data.response,
                emotion=parsed.data.emotion,
                json_valid=not parsed.fallback,
                emotion_correct=parsed.data.emotion == prompt.expected_emotion,
                length_chars=len(parsed.data.response),
                latency_seconds=latencia,
                tokens_per_second=_tokens_por_segundo(respuesta.chat_data),
                attempts=respuesta.attempts,
                repaired=parsed.repaired,
                fallback=parsed.fallback,
                error="; ".join(parsed.errors),
            )
        except OllamaError as exc:
            latencia = time.perf_counter() - inicio
            item = PromptResult(
                prompt_id=prompt.id,
                user=prompt.user,
                expected_emotion=prompt.expected_emotion,
                reference=prompt.reference,
                response="",
                emotion="",
                json_valid=False,
                emotion_correct=False,
                length_chars=0,
                latency_seconds=latencia,
                error=str(exc),
            )
            resultado.errors.append(f"{prompt.id}: {exc}")
        except Exception as exc:  # pragma: no cover - protección de la CLI
            latencia = time.perf_counter() - inicio
            item = PromptResult(
                prompt_id=prompt.id,
                user=prompt.user,
                expected_emotion=prompt.expected_emotion,
                reference=prompt.reference,
                response="",
                emotion="",
                json_valid=False,
                emotion_correct=False,
                length_chars=0,
                latency_seconds=latencia,
                error=f"error inesperado: {exc}",
            )
            resultado.errors.append(f"{prompt.id}: {exc}")
        resultado.prompts.append(item)
        if progress is not None:
            progress(indice, len(seleccionados), item)

    backend = _calcular_similitudes(
        resultado.prompts, embedder=embedder, forzar_lexica=forzar_lexica
    )
    if judge and judge_model:
        for item in resultado.prompts:
            if not item.response:
                continue
            nota, razon = _juzgar_personalidad(
                cliente,
                judge_model,
                user=item.user,
                response=item.response,
                emotion=item.emotion,
                reference=item.reference,
            )
            item.personality = nota
            item.personality_reason = razon
    resultado.duration_seconds = time.perf_counter() - inicio_total
    resultado.metrics = calcular_metricas(resultado.prompts, config)
    resultado.metrics["similarity_backend"] = backend
    return resultado


def _formatear_porcentaje(valor: Any) -> str:
    if valor is None:
        return "—"
    return f"{float(valor) * 100:.1f}%"


def _formatear_numero(valor: Any, decimales: int = 3) -> str:
    if valor is None:
        return "—"
    return f"{float(valor):.{decimales}f}"


def _recortar(texto: str, limite: int = 180) -> str:
    limpio = " ".join(str(texto).split())
    return limpio if len(limpio) <= limite else limpio[: limite - 1] + "…"


def generar_reporte(
    resultados: Sequence[EvaluationResult],
    *,
    prompts_path: str,
    judge: bool,
) -> str:
    """Genera un Markdown comparativo con resumen y detalle por prompt."""
    fecha = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lineas = [
        "# Evaluación de Nitshio",
        "",
        f"- **Fecha:** {fecha}",
        f"- **Set:** `{prompts_path}`",
        f"- **Modelos:** {', '.join(f'`{r.model}`' for r in resultados)}",
        f"- **Juez de personalidad:** {'sí' if judge else 'no'}",
        "",
        "## Resumen",
        "",
        "| Modelo | JSON válido | Emoción | Similitud | Longitud | Latencia | Tok/s | Personalidad | Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for resultado in resultados:
        metricas = resultado.metrics
        lineas.append(
            "| "
            + " | ".join(
                [
                    f"`{resultado.model}`",
                    _formatear_porcentaje(metricas.get("json_valid_rate")),
                    _formatear_porcentaje(metricas.get("emotion_accuracy")),
                    _formatear_numero(metricas.get("similarity_mean")),
                    _formatear_numero(metricas.get("length_mean"), 1),
                    _formatear_numero(metricas.get("latency_mean_seconds"), 2) + " s",
                    _formatear_numero(metricas.get("tokens_per_second_mean"), 2),
                    (
                        _formatear_porcentaje(metricas.get("personality_mean"))
                        if metricas.get("personality_mean") is not None
                        else "—"
                    ),
                    _formatear_numero(metricas.get("composite_score")),
                ]
            )
            + " |"
        )
    lineas.extend(
        [
            "",
            "> La similitud se calcula con embeddings de Ollama; si no están "
            "disponibles, cae a similitud léxica y el reporte lo indica en el "
            "detalle. El score compuesto usa los pesos de "
            "`config.training.evaluation`.",
            "",
        ]
    )

    if len(resultados) >= 2:
        ordenados = sorted(
            resultados,
            key=lambda item: float(item.metrics.get("composite_score") or 0),
            reverse=True,
        )
        mejor = ordenados[0]
        peor = ordenados[-1]
        diferencia = float(mejor.metrics.get("composite_score") or 0) - float(
            peor.metrics.get("composite_score") or 0
        )
        lineas.extend(
            [
                "## Comparación",
                "",
                f"- Mejor score: `{mejor.model}` "
                f"({_formatear_numero(mejor.metrics.get('composite_score'))}).",
                f"- Diferencia con `{peor.model}`: "
                f"{_formatear_numero(diferencia)} puntos.",
                "- Revisa el detalle: un score alto no compensa una emoción "
                "incorrecta ni una respuesta fuera de personaje.",
                "",
            ]
        )

    for resultado in resultados:
        lineas.extend(
            [
                f"## Detalle — `{resultado.model}`",
                "",
                f"- Duración: {resultado.duration_seconds:.1f} s",
                f"- Backend de similitud: `{resultado.metrics.get('similarity_backend', '—')}`",
                f"- Errores de red: {len(resultado.errors)}",
                "",
                "| Prompt | Esperada | Obtenida | JSON | Emoción | Similitud | Latencia | Respuesta |",
                "|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for item in resultado.prompts:
            lineas.append(
                "| "
                + " | ".join(
                    [
                        f"`{item.prompt_id}`",
                        f"`{item.expected_emotion}`",
                        f"`{item.emotion or '—'}`",
                        "sí" if item.json_valid else "no",
                        "sí" if item.emotion_correct else "no",
                        _formatear_numero(item.similarity),
                        _formatear_numero(item.latency_seconds, 2) + " s",
                        _recortar(item.response or item.error),
                    ]
                )
                + " |"
            )
        lineas.append("")
    return "\n".join(lineas).rstrip() + "\n"


def _resolver_modelos(
    config: dict[str, Any],
    modelos: Sequence[str] | None,
    *,
    compare_base: bool,
) -> list[str]:
    ollama_models = (config.get("ollama") or {}).get("models") or {}
    activo = str(ollama_models.get("active", "")).strip()
    base = str(ollama_models.get("base", "")).strip()
    elegidos: list[str] = []
    for modelo in modelos or ([activo] if activo else []):
        if modelo and modelo not in elegidos:
            elegidos.append(modelo)
    if compare_base and base and base not in elegidos:
        elegidos.append(base)
    if not elegidos:
        raise EvaluationError(
            "No hay modelo para evaluar: configura ollama.models.active."
        )
    return elegidos


def _guardar_resultados(
    resultados: Sequence[EvaluationResult],
    *,
    output: Path,
    prompts_path: str,
    judge: bool,
) -> tuple[Path, Path]:
    output = _ruta(output)
    if output.suffix.lower() != ".md":
        output = output / (
            "eval_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".md"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    reporte = generar_reporte(
        resultados, prompts_path=prompts_path, judge=judge
    )
    output.write_text(reporte, encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "prompts": prompts_path,
                "judge": judge,
                "results": [resultado.as_dict() for resultado in resultados],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output, json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evalúa y compara modelos de Nitshio."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Tags de Ollama a evaluar (default: ollama.models.active).",
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Añade ollama.models.base a la comparación.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Set YAML de evaluación (default: training.eval_prompts).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Reporte .md o carpeta (default: training.reports_dir).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument(
        "--record-metrics",
        action="store_true",
        help="Adjunta el resumen a models/versions.json.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = cargar_config(args.config)
        persona = cargar_persona(config)
        training_cfg = config.get("training") or {}
        prompts_path = _ruta(
            args.prompts or training_cfg.get("eval_prompts", "training/eval/prompts.yaml")
        )
        prompts = cargar_prompts(prompts_path, persona.emotions)
        modelos = _resolver_modelos(
            config, args.models, compare_base=args.compare_base
        )
        evaluacion_cfg = training_cfg.get("evaluation") or {}
        judge = bool(evaluacion_cfg.get("judge", False)) and not args.no_judge
        embedder = None
        if not args.no_embeddings:
            try:
                embedder = crear_embeddings(config)
            except Exception as exc:
                print(
                    f"AVISO: embeddings no disponibles ({exc}); se usará "
                    "similitud léxica.",
                    file=sys.stderr,
                )
    except (OSError, ValueError, EvaluationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    resultados: list[EvaluationResult] = []
    cantidad_prompts = (
        min(len(prompts), max(0, int(args.limit))) if args.limit else len(prompts)
    )
    for modelo in modelos:
        if not args.quiet:
            print(
                f"Evaluando {modelo} ({cantidad_prompts} prompts)...",
                file=sys.stderr,
            )
        try:
            resultado = evaluate_model(
                config,
                modelo,
                prompts,
                embedder=embedder,
                judge=judge,
                forzar_lexica=args.no_embeddings,
                limit=args.limit,
            )
        except (OllamaError, EvaluationError) as exc:
            print(f"ERROR al evaluar {modelo}: {exc}", file=sys.stderr)
            return 1
        resultados.append(resultado)
        if not args.quiet:
            print(
                f"  score={resultado.metrics.get('composite_score')} "
                f"json={resultado.metrics.get('json_valid_rate')} "
                f"emoción={resultado.metrics.get('emotion_accuracy')}",
                file=sys.stderr,
            )

    salida = args.output or _ruta(
        training_cfg.get("reports_dir", "training/eval/reports")
    )
    try:
        reporte, json_path = _guardar_resultados(
            resultados,
            output=salida,
            prompts_path=str(prompts_path),
            judge=judge,
        )
    except OSError as exc:
        print(f"ERROR al guardar el reporte: {exc}", file=sys.stderr)
        return 1

    if args.record_metrics:
        registro = load_registry(
            _ruta(training_cfg.get("versions_file", "models/versions.json"))
        )
        for resultado in resultados:
            if find_version(resultado.model, registro) is None:
                continue
            entrada = find_version(resultado.model, registro) or {}
            metricas = dict(entrada.get("metrics") or {})
            metricas["evaluation"] = resultado.metrics
            try:
                update_metrics(
                    resultado.model,
                    metricas,
                    registry_path=_ruta(
                        training_cfg.get("versions_file", "models/versions.json")
                    ),
                )
            except Exception as exc:
                print(
                    f"AVISO: no se pudieron registrar métricas de "
                    f"{resultado.model}: {exc}",
                    file=sys.stderr,
                )

    print(f"Reporte: {reporte}")
    print(f"JSON: {json_path}")
    for resultado in resultados:
        print(
            f"{resultado.model}: score "
            f"{resultado.metrics.get('composite_score')} | "
            f"JSON {resultado.metrics.get('json_valid_rate')} | "
            f"emoción {resultado.metrics.get('emotion_accuracy')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
