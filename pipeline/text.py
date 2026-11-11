"""Pipeline de texto interactivo para el CLI de Nitshio."""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

from brain.llm.orchestrator import Orchestrator, TurnResult
from brain.llm.parser import FALLBACK_WARNING


class TextPipeline:
    """Bucle de conversación por texto con comandos de depuración."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        *,
        debug: bool = False,
        on_turn: Callable[[TurnResult], None] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.debug = debug
        self.on_turn = on_turn

    def responder(self, texto: str, *, speak: bool = False) -> TurnResult:
        return self.orchestrator.process(texto, speak=speak)

    def imprimir(self, resultado: TurnResult) -> None:
        print(f"Nitshio [{resultado.emotion}]: {resultado.response}")
        if resultado.parsed.fallback:
            print(FALLBACK_WARNING, file=sys.stderr)
        if self.debug:
            print(
                f"  [intentos={resultado.attempts} "
                f"fallback={resultado.parsed.fallback} "
                f"reparado={resultado.parsed.repaired} "
                f"herramientas={[c.tool for c in resultado.tool_calls]}]",
                file=sys.stderr,
            )
            for indice, crudo in enumerate(resultado.raw_responses, start=1):
                print(f"  [crudo {indice}] {crudo[:500]}", file=sys.stderr)
            if resultado.memory_context:
                print(
                    f"  [memoria] {resultado.memory_context[:500]}",
                    file=sys.stderr,
                )
            for llamada, salida in zip(
                resultado.tool_calls, resultado.tool_results
            ):
                print(
                    f"  [tool {llamada.tool}] "
                    f"ok={salida.ok} {salida.content[:300] or salida.error}",
                    file=sys.stderr,
                )

    def run(self, *, speak: bool = False) -> int:
        print("Nitshio — chat de texto")
        print(
            "Comandos: /salir, /limpiar, /herramientas, /memoria <texto>, "
            "/emocion <nombre>"
        )
        print("-" * 60)
        while True:
            try:
                texto = input("Tú: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not texto:
                continue
            comando = texto.lower()
            if comando in {"/salir", "/exit", "/quit"}:
                break
            if comando == "/limpiar":
                self.orchestrator.reset_history()
                print("Historial borrado.")
                continue
            if comando == "/herramientas":
                registro = self.orchestrator.registry
                if registro is None:
                    print("Herramientas desactivadas.")
                else:
                    for herramienta in registro.catalog():
                        print(
                            f"- {herramienta['name']}: "
                            f"{herramienta['description']}"
                        )
                continue
            if comando.startswith("/memoria "):
                memoria = self.orchestrator.memory
                if memoria is None:
                    print("Memoria desactivada.")
                    continue
                consulta = texto[len("/memoria ") :].strip()
                try:
                    print(memoria.build_context(consulta) or "(sin recuerdos)")
                except Exception as exc:
                    print(f"Error de memoria: {exc}")
                continue
            if comando.startswith("/emocion "):
                nombre = texto[len("/emocion ") :].strip()
                if self.orchestrator.avatar is None:
                    print("Avatar desactivado.")
                else:
                    try:
                        self.orchestrator.avatar.set_emotion(nombre)
                        print(f"Emoción: {nombre}")
                    except Exception as exc:
                        print(f"Error de emoción: {exc}")
                continue
            try:
                resultado = self.responder(texto, speak=speak)
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                continue
            self.imprimir(resultado)
            if self.on_turn is not None:
                self.on_turn(resultado)
        print("Nitshio: ...Hmph. Ya vete, pendejo.")
        return 0
