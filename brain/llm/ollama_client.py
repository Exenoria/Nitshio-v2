"""Cliente de Ollama con salida estructurada, reintento y fallback.

Toda llamada al modelo de lenguaje del proyecto debe pasar por este módulo.
El cliente no conoce nombres de modelo concretos: los recibe desde
``config/config.yaml`` a través del pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import requests

from brain.llm.parser import (
    ParsedResponse,
    ParseError,
    fallback_response,
    parse_response,
    response_schema,
)


class OllamaError(RuntimeError):
    """Error de red o de respuesta de la API de Ollama."""


@dataclass
class StructuredChatResult:
    """Resultado de una llamada con validación JSON y reintentos."""

    parsed: ParsedResponse
    attempts: int
    raw_responses: list[str] = field(default_factory=list)
    chat_data: dict[str, Any] | None = None


class OllamaClient:
    """Cliente HTTP mínimo para la API local de Ollama."""

    _CAMPOS_RAZONAMIENTO = ("thinking", "reasoning", "reasoning_content")

    def __init__(
        self,
        host: str = "http://localhost:11434",
        *,
        timeout: float = 180.0,
        keep_alive: str = "30m",
        think: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.think = bool(think)
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    # Bajo nivel
    # ------------------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.host}{path}"

    def health(self) -> dict[str, Any]:
        """Comprueba que Ollama responde y devuelve su versión."""
        try:
            resp = self.session.get(self._url("/api/version"), timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise OllamaError(
                f"No se pudo conectar con Ollama en {self.host}: {exc}"
            ) from exc

    def list_models(self) -> list[dict[str, Any]]:
        """Lista los modelos instalados en Ollama."""
        try:
            resp = self.session.get(self._url("/api/tags"), timeout=15)
            resp.raise_for_status()
            return list(resp.json().get("models", []))
        except requests.RequestException as exc:
            raise OllamaError(f"No se pudo listar modelos: {exc}") from exc

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        think: bool | None = None,
        keep_alive: str | None = None,
    ) -> dict[str, Any]:
        """Llama a ``/api/chat`` y devuelve la respuesta sin razonamiento.

        ``think`` por defecto usa ``ollama.think`` de la configuración. El
        campo se envía siempre explícitamente para que Qwen3.5 no mezcle su
        cadena de pensamiento con el JSON de personalidad.
        """
        pensar = self.think if think is None else bool(think)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": pensar,
            "keep_alive": keep_alive or self.keep_alive,
            "options": options or {},
        }
        if schema is not None:
            payload["format"] = schema

        try:
            resp = self.session.post(
                self._url("/api/chat"),
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            detalle = ""
            respuesta = getattr(exc, "response", None)
            if respuesta is not None:
                detalle = f" — {respuesta.text[:500]}"
            raise OllamaError(
                f"Fallo al llamar a Ollama ({model}): {exc}{detalle}"
            ) from exc

        if data.get("error"):
            raise OllamaError(f"Ollama devolvió un error: {data['error']}")
        return self._sin_razonamiento(data)

    @classmethod
    def _sin_razonamiento(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Copia la respuesta sin campos de pensamiento de Qwen3.5.

        Aunque ``think`` esté desactivado, algunas versiones de Ollama pueden
        devolver ``thinking``/``reasoning``. El pipeline solo debe ver el
        contenido final, así que esos campos se descartan antes de llegar al
        parser o a las métricas.
        """
        limpio = {
            clave: valor
            for clave, valor in data.items()
            if clave not in cls._CAMPOS_RAZONAMIENTO
        }
        mensaje = limpio.get("message")
        if isinstance(mensaje, dict):
            limpio["message"] = {
                clave: valor
                for clave, valor in mensaje.items()
                if clave not in cls._CAMPOS_RAZONAMIENTO
            }
        return limpio

    @staticmethod
    def _contenido(data: dict[str, Any]) -> str:
        """Extrae el texto final de una respuesta de chat o generate.

        Nunca usa campos de razonamiento como ``thinking`` o ``reasoning``:
        si el modelo solo devolvió pensamiento, el parser lo tratará como
        respuesta vacía y activará el fallback con aviso.
        """
        mensaje = data.get("message")
        if isinstance(mensaje, dict) and "content" in mensaje:
            return str(mensaje.get("content", ""))
        return str(data.get("response", ""))

    # ------------------------------------------------------------------
    # Alto nivel: JSON válido con reintentos
    # ------------------------------------------------------------------
    def chat_structured(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        allowed_emotions: Iterable[str],
        options: dict[str, Any] | None = None,
        retries: int = 2,
        fallback_emotion: str = "neutral",
        correction_prompt: str | None = None,
    ) -> StructuredChatResult:
        """Pide una respuesta, valida el JSON y reintenta si es inválido.

        - ``retries`` cuenta los reintentos *adicionales* (0 = un solo intento).
        - En cada reintento se conserva la respuesta inválida y se añade una
          instrucción correctiva explícita.
        - Si se agotan los reintentos, se devuelve un fallback seguro; nunca se
          propaga un JSON roto al pipeline.
        """
        emociones = list(allowed_emotions)
        schema = response_schema(emociones)
        historial = list(messages)
        crudos: list[str] = []
        ultimo_error: Exception | None = None
        ultimo_data: dict[str, Any] | None = None

        for intento in range(1, max(1, retries + 1) + 1):
            data = self.chat(
                model,
                historial,
                schema=schema,
                options=options,
                think=self.think,
            )
            ultimo_data = data
            crudo = self._contenido(data)
            crudos.append(crudo)
            try:
                parsed = parse_response(crudo, emociones)
                return StructuredChatResult(
                    parsed=parsed,
                    attempts=intento,
                    raw_responses=crudos,
                    chat_data=data,
                )
            except ParseError as exc:
                ultimo_error = exc
                if intento <= retries:
                    correccion = correction_prompt or (
                        "Tu respuesta anterior no cumple el formato. "
                        "Devuelve SOLO un objeto JSON válido, sin texto fuera "
                        "del JSON, con las claves exactas 'response' (texto) y "
                        "'emotion' (una de: "
                        + ", ".join(emociones)
                        + "). No agregues 'action' ni otras claves."
                    )
                    historial = list(messages) + [
                        {"role": "assistant", "content": crudo},
                        {"role": "user", "content": correccion},
                    ]

        # Último recurso: rescatar lo que se pueda y devolver emoción neutral.
        crudo_final = crudos[-1] if crudos else ""
        parsed = fallback_response(
            crudo_final,
            emociones,
            fallback_emotion=fallback_emotion,
        )
        parsed.errors.append(
            f"Se agotaron los reintentos: {ultimo_error}"
            if ultimo_error
            else "Se agotaron los reintentos."
        )
        return StructuredChatResult(
            parsed=parsed,
            attempts=max(1, len(crudos)),
            raw_responses=crudos,
            chat_data=ultimo_data,
        )


def extraer_json(texto: str) -> dict[str, Any]:
    """Utilidad para tests y herramientas: parsea un JSON sin validar emoción."""
    return json.loads(texto)
