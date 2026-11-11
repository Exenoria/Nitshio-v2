#!/usr/bin/env python
"""Chat de texto mínimo para probar la personalidad de Nitshio.

Ejemplos:
    python scripts/chat_cli.py
    python scripts/chat_cli.py --once "Hola, Nitshio. ¿Cómo estás?"
    python scripts/chat_cli.py --json --once "Cuéntame algo"
    python scripts/chat_cli.py --model nitshio-v1 --debug
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permite ejecutar `python scripts/chat_cli.py` sin instalar el paquete.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.config import cargar_config, host_ollama
from brain.llm.ollama_client import OllamaClient, OllamaError
from brain.llm.parser import FALLBACK_WARNING
from brain.persona.loader import (
    PersonaError,
    cargar_persona,
    construir_mensajes,
)


def _configurar_salida() -> None:
    """Evita errores de encoding en consolas Windows viejas."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _recortar_historial(
    historial: list[dict[str, str]], turnos: int
) -> list[dict[str, str]]:
    max_mensajes = max(0, turnos) * 2
    return historial[-max_mensajes:] if max_mensajes else []


def main(argv: list[str] | None = None) -> int:
    _configurar_salida()
    parser = argparse.ArgumentParser(description="Chat de prueba de Nitshio.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--model",
        default=None,
        help="Modelo de Ollama. Por defecto, ollama.models.active.",
    )
    parser.add_argument("--once", default=None, help="Un solo mensaje y salir.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime solo el JSON de respuesta.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Muestra intentos, reparaciones y respuestas crudas.",
    )
    args = parser.parse_args(argv)

    try:
        config = cargar_config(args.config)
        persona = cargar_persona(config)
    except (OSError, ValueError, PersonaError) as exc:
        print(f"ERROR de configuración/personalidad: {exc}", file=sys.stderr)
        return 1

    modelo = args.model or str(config["ollama"]["models"]["active"])
    opciones = dict(config["ollama"].get("options", {}).get("primary", {}))
    retries = int(
        config["ollama"].get("structured_output", {}).get("retries", 2)
    )
    fallback_emotion = str(
        config["ollama"]
        .get("structured_output", {})
        .get("fallback_emotion", persona.default_emotion)
    )
    turnos = int(config.get("persona", {}).get("history_turns", 12))
    max_context = int(config.get("persona", {}).get("max_context_chars", 12000))

    cliente = OllamaClient(
        host_ollama(config),
        timeout=float(config["ollama"].get("request_timeout_seconds", 180)),
        keep_alive=str(config["ollama"].get("keep_alive", "30m")),
        think=bool(config["ollama"].get("think", False)),
    )

    try:
        cliente.health()
    except OllamaError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Abre Ollama o ejecuta `ollama serve` y vuelve a intentar.", file=sys.stderr)
        return 1

    if not args.json:
        print(f"Nitshio — modelo: {modelo}")
        print("Escribe /salir para terminar, /limpiar para borrar el historial.")
        print("-" * 60)

    historial: list[dict[str, str]] = []

    def responder(texto: str) -> bool:
        mensajes = construir_mensajes(
            persona,
            texto,
            history=historial,
            max_context_chars=max_context,
        )
        try:
            resultado = cliente.chat_structured(
                modelo,
                mensajes,
                allowed_emotions=persona.emotions,
                options=opciones,
                retries=retries,
                fallback_emotion=fallback_emotion,
            )
        except OllamaError as exc:
            print(f"ERROR de Ollama: {exc}", file=sys.stderr)
            return False

        parsed = resultado.parsed
        if parsed.fallback:
            print(FALLBACK_WARNING, file=sys.stderr)
        if args.json:
            print(
                json.dumps(
                    parsed.data.model_dump(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        else:
            print(f"Nitshio [{parsed.data.emotion}]: {parsed.data.response}")
        if args.debug:
            print(
                f"  [intentos={resultado.attempts} "
                f"reparado={parsed.repaired} fallback={parsed.fallback} "
                f"texto_extra={parsed.extra_text}]",
                file=sys.stderr,
            )
            for i, crudo in enumerate(resultado.raw_responses, start=1):
                print(f"  [crudo {i}] {crudo[:500]}", file=sys.stderr)

        historial.append({"role": "user", "content": texto})
        historial.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    parsed.data.model_dump(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
        del historial[: max(0, len(historial) - turnos * 2)]
        return True

    if args.once is not None:
        return 0 if responder(args.once) else 2

    while True:
        try:
            texto = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not texto:
            continue
        if texto.lower() in {"/salir", "/exit", "/quit"}:
            break
        if texto.lower() == "/limpiar":
            historial.clear()
            print("Historial borrado.")
            continue
        responder(texto)

    if not args.json:
        print("Nitshio: ...Hmph. Ya vete, pendejo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
