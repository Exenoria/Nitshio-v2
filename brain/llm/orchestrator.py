"""Orquestador del pipeline: memoria, router, herramientas, LLM y avatar.

Es la pieza que une todo sin acoplar los módulos entre sí. El CLI y el modo
voz solo llaman a :meth:`Orchestrator.process`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from brain.llm.ollama_client import OllamaClient, OllamaError
from brain.llm.parser import ParsedResponse
from brain.persona.loader import Persona, construir_mensajes, respuesta_a_json
from brain.router.tool_router import ToolCall, ToolRouter, crear_router
from tools.base import ToolContext, ToolResult
from tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TurnResult:
    """Todo lo que produjo un turno de conversación."""

    user_text: str
    response: str
    emotion: str
    parsed: ParsedResponse
    attempts: int = 1
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    memory_context: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)


class Orchestrator:
    """Procesa un mensaje de texto y devuelve una respuesta en personaje."""

    def __init__(
        self,
        config: dict[str, Any],
        persona: Persona,
        client: OllamaClient,
        *,
        memory: Any = None,
        registry: ToolRegistry | None = None,
        router: ToolRouter | None = None,
        avatar: Any = None,
        tts: Any = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.persona = persona
        self.client = client
        self.memory = memory
        self.registry = registry
        self.router = router
        self.avatar = avatar
        self.tts = tts
        self.logger = logger or logging.getLogger("nitshio.orchestrator")

        ollama_cfg = config.get("ollama") or {}
        self.model = str((ollama_cfg.get("models") or {}).get("active", ""))
        if not self.model:
            raise ValueError("Falta ollama.models.active en la configuración.")
        self.options = dict(
            (ollama_cfg.get("options") or {}).get("primary") or {}
        )
        structured = ollama_cfg.get("structured_output") or {}
        self.retries = int(structured.get("retries", 2))
        self.fallback_emotion = str(
            structured.get("fallback_emotion", persona.default_emotion)
        )
        persona_cfg = config.get("persona") or {}
        self.history_turns = max(0, int(persona_cfg.get("history_turns", 12)))
        self.max_context_chars = int(
            persona_cfg.get("max_context_chars", 12000)
        )
        router_cfg = config.get("router") or {}
        self.router_enabled = bool(router_cfg.get("enabled", True))
        self.max_tool_rounds = max(0, int(router_cfg.get("max_tool_rounds", 1)))
        self._history: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Historial
    # ------------------------------------------------------------------
    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    def reset_history(self) -> None:
        self._history.clear()

    def _recortar_historial(self) -> None:
        max_mensajes = self.history_turns * 2
        if max_mensajes <= 0:
            self._history.clear()
        elif len(self._history) > max_mensajes:
            del self._history[: len(self._history) - max_mensajes]

    # ------------------------------------------------------------------
    # Contexto
    # ------------------------------------------------------------------
    def _contexto_memoria(self, texto: str) -> str:
        if self.memory is None:
            return ""
        try:
            return str(self.memory.build_context(texto) or "")
        except Exception as exc:
            self.logger.warning("Memoria no disponible: %s", exc)
            return ""

    def _contexto_herramienta(
        self, texto: str
    ) -> tuple[str, list[ToolCall], list[ToolResult]]:
        if (
            not self.router_enabled
            or self.router is None
            or self.registry is None
            or self.max_tool_rounds <= 0
        ):
            return "", [], []
        catalogo = self.registry.catalog()
        if not catalogo:
            return "", [], []
        try:
            llamada = self.router.route(texto, catalogo)
        except Exception as exc:
            self.logger.warning("El router falló: %s", exc)
            return "", [], []
        if not llamada.usa_herramienta:
            return "", [llamada], []
        resultado = self.registry.execute(llamada.tool, llamada.arguments)
        if resultado.ok:
            bloque = (
                f"## Resultado de la herramienta '{llamada.tool}'\n"
                f"{resultado.content[:4000]}\n"
                "Usa este resultado si responde a la petición; no inventes "
                "datos que no aparezcan aquí."
            )
        else:
            bloque = (
                f"## La herramienta '{llamada.tool}' falló\n"
                f"{resultado.error or 'Error desconocido'}\n"
                "Explícalo en personaje y ofrece una alternativa."
            )
        return bloque, [llamada], [resultado]

    # ------------------------------------------------------------------
    # Turno
    # ------------------------------------------------------------------
    def process(self, texto: str, *, speak: bool = False) -> TurnResult:
        """Procesa un mensaje y devuelve la respuesta validada."""
        user_text = " ".join(str(texto or "").split()).strip()
        if not user_text:
            raise ValueError("El mensaje está vacío.")

        contexto_memoria = self._contexto_memoria(user_text)
        contexto_herramienta, llamadas, resultados = self._contexto_herramienta(
            user_text
        )
        contexto = "\n\n".join(
            bloque
            for bloque in (contexto_memoria, contexto_herramienta)
            if bloque
        )

        mensajes = construir_mensajes(
            self.persona,
            user_text,
            history=self._history,
            contexto=contexto or None,
            max_context_chars=self.max_context_chars,
        )
        resultado = self.client.chat_structured(
            self.model,
            mensajes,
            allowed_emotions=self.persona.emotions,
            options=self.options,
            retries=self.retries,
            fallback_emotion=self.fallback_emotion,
        )
        parsed = resultado.parsed
        respuesta = parsed.data.response
        emocion = parsed.data.emotion

        if self.memory is not None:
            try:
                self.memory.save_conversation(
                    user_text, respuesta, emotion=emocion
                )
            except Exception as exc:
                self.logger.warning("No se pudo guardar la conversación: %s", exc)

        self._history.append({"role": "user", "content": user_text})
        self._history.append(
            {"role": "assistant", "content": respuesta_a_json(parsed.data)}
        )
        self._recortar_historial()

        if self.avatar is not None:
            try:
                self.avatar.set_emotion(emocion)
            except Exception as exc:
                self.logger.warning("No se pudo cambiar la emoción: %s", exc)

        if speak and self.tts is not None:
            try:
                self.tts.speak(respuesta)
            except Exception as exc:
                self.logger.warning("TTS falló: %s", exc)

        return TurnResult(
            user_text=user_text,
            response=respuesta,
            emotion=emocion,
            parsed=parsed,
            attempts=resultado.attempts,
            tool_calls=llamadas,
            tool_results=resultados,
            memory_context=contexto_memoria,
            history=self.history,
            raw_responses=list(resultado.raw_responses),
        )

    def process_with_retry(
        self, texto: str, *, speak: bool = False, intentos: int = 1
    ) -> TurnResult:
        """Reintenta el turno completo ante fallos de red de Ollama."""
        ultimo: Exception | None = None
        for _ in range(max(1, int(intentos))):
            try:
                return self.process(texto, speak=speak)
            except OllamaError as exc:
                ultimo = exc
        assert ultimo is not None
        raise ultimo


def crear_orquestador(
    config: dict[str, Any],
    persona: Persona,
    client: OllamaClient,
    *,
    con_memoria: bool = True,
    con_herramientas: bool = True,
    con_avatar: bool = False,
    con_tts: bool = False,
    confirm: Callable[[str], bool] | None = None,
    logger: logging.Logger | None = None,
) -> Orchestrator:
    """Construye el orquestador con degradación elegante de opcionales."""
    logger = logger or logging.getLogger("nitshio.orchestrator")
    memoria = None
    if con_memoria:
        try:
            from memory.store import MemoryStore

            memoria = MemoryStore(config)
        except Exception as exc:
            logger.warning("Memoria desactivada: %s", exc)

    registro = None
    if con_herramientas:
        try:
            contexto = ToolContext(
                root=ROOT,
                config=config,
                confirm=confirm,
                logger=logger,
                memory=memoria,
                ollama=client,
            )
            registro = ToolRegistry(contexto, logger=logger).discover()
            if registro.errors:
                logger.warning("Herramientas con errores: %s", registro.errors)
        except Exception as exc:
            logger.warning("Herramientas desactivadas: %s", exc)

    router = None
    try:
        router = crear_router(config, client)
    except Exception as exc:
        logger.warning("Router desactivado: %s", exc)

    avatar = None
    if con_avatar:
        try:
            from avatar.vmc import AvatarController

            avatar = AvatarController(config)
        except Exception as exc:
            logger.warning("Avatar desactivado: %s", exc)

    tts = None
    if con_tts:
        try:
            from tts.manager import TTSManager

            tts = TTSManager(config)
        except Exception as exc:
            logger.warning("TTS desactivado: %s", exc)

    return Orchestrator(
        config,
        persona,
        client,
        memory=memoria,
        registry=registro,
        router=router,
        avatar=avatar,
        tts=tts,
        logger=logger,
    )
