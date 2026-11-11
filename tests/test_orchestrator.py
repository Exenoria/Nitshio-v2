"""Tests del orquestador: memoria, router, herramientas, historial y efectos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from brain.llm.ollama_client import OllamaError, StructuredChatResult
from brain.llm.orchestrator import Orchestrator
from brain.llm.parser import NitshioResponse, ParsedResponse
from brain.persona.loader import Persona
from brain.router.tool_router import ToolCall
from tools.base import Tool, ToolContext, ToolResult
from tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]


def _persona() -> Persona:
    return Persona(
        name="Nitshio",
        system_prompt="Eres Nitshio, una tsundere.",
        few_shots=(),
        emotions=("neutral", "fun", "angry", "joy", "sorrow", "surprised"),
        default_emotion="neutral",
        include_few_shots_in_runtime=False,
    )


def _config(**overrides: Any) -> dict[str, Any]:
    config = {
        "ollama": {
            "models": {"active": "nitshio-v1"},
            "options": {"primary": {"temperature": 0.7}},
            "structured_output": {
                "retries": 1,
                "fallback_emotion": "neutral",
            },
        },
        "persona": {"history_turns": 12, "max_context_chars": 4000},
        "router": {"enabled": False, "max_tool_rounds": 1},
    }
    config.update(overrides)
    return config


def _parsed(
    response: str = "Hmph.", emotion: str = "neutral"
) -> ParsedResponse:
    return ParsedResponse(
        data=NitshioResponse(response=response, emotion=emotion), raw="{}"
    )


class ClienteFalso:
    def __init__(self, respuestas: list[Any] | None = None) -> None:
        self.respuestas = list(respuestas or [])
        self.llamadas: list[dict[str, Any]] = []

    def chat_structured(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> StructuredChatResult:
        self.llamadas.append(
            {"model": model, "messages": messages, **kwargs}
        )
        if self.respuestas:
            respuesta = self.respuestas.pop(0)
            if isinstance(respuesta, Exception):
                raise respuesta
            if isinstance(respuesta, ParsedResponse):
                return StructuredChatResult(
                    parsed=respuesta, attempts=1, raw_responses=[respuesta.raw]
                )
            return StructuredChatResult(
                parsed=_parsed(**respuesta), attempts=1
            )
        return StructuredChatResult(parsed=_parsed(), attempts=1)


class MemoriaFalsa:
    def __init__(self, contexto: str = "") -> None:
        self.contexto = contexto
        self.guardados: list[tuple[str, str, str | None]] = []

    def build_context(self, texto: str) -> str:
        return self.contexto

    def save_conversation(
        self, user: str, response: str, emotion: str | None = None
    ) -> None:
        self.guardados.append((user, response, emotion))


class AvatarFalso:
    def __init__(self) -> None:
        self.emociones: list[str] = []

    def set_emotion(self, emocion: str) -> None:
        self.emociones.append(emocion)


class TTSFalso:
    def __init__(self) -> None:
        self.textos: list[str] = []

    def speak(self, texto: str) -> None:
        self.textos.append(texto)


class RouterFalso:
    def __init__(self, llamada: ToolCall) -> None:
        self.llamada = llamada
        self.consultas: list[str] = []

    def route(self, texto: str, catalogo, **kwargs: Any) -> ToolCall:
        self.consultas.append(texto)
        return self.llamada


class ToolFalsa(Tool):
    name = "falsa"
    description = "Herramienta de prueba"
    parameters = {
        "type": "object",
        "properties": {"texto": {"type": "string"}},
        "required": ["texto"],
        "additionalProperties": False,
    }

    def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        return ToolResult.exito(f"resultado:{arguments['texto']}")


def _registro(tmp_path: Path) -> ToolRegistry:
    contexto = ToolContext(
        root=tmp_path, config={"tools": {"enabled": ["falsa"]}}
    )
    registro = ToolRegistry(contexto)
    registro.register(ToolFalsa())
    return registro


# ---------------------------------------------------------------------------
# Turno básico
# ---------------------------------------------------------------------------
def test_turno_basico_actualiza_todo() -> None:
    cliente = ClienteFalso([{"response": "No me importa.", "emotion": "angry"}])
    memoria = MemoriaFalsa("Recuerdos relevantes: el usuario vive en México.")
    avatar = AvatarFalso()
    tts = TTSFalso()
    orquestador = Orchestrator(
        _config(),
        _persona(),
        cliente,
        memory=memoria,
        avatar=avatar,
        tts=tts,
    )
    resultado = orquestador.process("  hola   Nitshio ", speak=True)

    assert resultado.user_text == "hola Nitshio"
    assert resultado.response == "No me importa."
    assert resultado.emotion == "angry"
    assert resultado.attempts == 1
    assert resultado.memory_context.startswith("Recuerdos relevantes")
    assert memoria.guardados == [
        ("hola Nitshio", "No me importa.", "angry")
    ]
    assert avatar.emociones == ["angry"]
    assert tts.textos == ["No me importa."]
    assert orquestador.history == [
        {"role": "user", "content": "hola Nitshio"},
        {
            "role": "assistant",
            "content": '{"response":"No me importa.","emotion":"angry"}',
        },
    ]

    mensajes = cliente.llamadas[0]["messages"]
    assert mensajes[0]["role"] == "system"
    assert "Recuerdos relevantes" in mensajes[0]["content"]
    assert mensajes[-1] == {"role": "user", "content": "hola Nitshio"}
    assert cliente.llamadas[0]["allowed_emotions"] == _persona().emotions
    assert cliente.llamadas[0]["retries"] == 1


def test_turn_result_incluye_respuestas_crudas() -> None:
    crudo = '{"response": "No me importa.", "emotion": "angry"}'
    parsed = ParsedResponse(
        data=NitshioResponse(response="No me importa.", emotion="angry"),
        raw=crudo,
    )
    orquestador = Orchestrator(_config(), _persona(), ClienteFalso([parsed]))

    resultado = orquestador.process("hola")

    assert resultado.raw_responses == [crudo]


def test_mensaje_vacio_falla() -> None:
    orquestador = Orchestrator(_config(), _persona(), ClienteFalso())
    with pytest.raises(ValueError, match="vacío"):
        orquestador.process("   ")


def test_falta_modelo_activo() -> None:
    config = _config()
    config["ollama"]["models"] = {}
    with pytest.raises(ValueError, match="models.active"):
        Orchestrator(config, _persona(), ClienteFalso())


def test_memoria_rota_no_tumba_el_turno() -> None:
    class MemoriaRota(MemoriaFalsa):
        def build_context(self, texto: str) -> str:
            raise RuntimeError("chroma caído")

        def save_conversation(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("disco lleno")

    orquestador = Orchestrator(
        _config(), _persona(), ClienteFalso(), memory=MemoriaRota()
    )
    resultado = orquestador.process("hola")
    assert resultado.response
    assert resultado.memory_context == ""


def test_avatar_y_tts_rotos_no_tumban_el_turno() -> None:
    class AvatarRoto:
        def set_emotion(self, emocion: str) -> None:
            raise RuntimeError("vmc caído")

    class TTSRoto:
        def speak(self, texto: str) -> None:
            raise RuntimeError("audio caído")

    orquestador = Orchestrator(
        _config(),
        _persona(),
        ClienteFalso(),
        avatar=AvatarRoto(),
        tts=TTSRoto(),
    )
    resultado = orquestador.process("hola", speak=True)
    assert resultado.response


# ---------------------------------------------------------------------------
# Historial
# ---------------------------------------------------------------------------
def test_historial_se_recorta() -> None:
    config = _config()
    config["persona"]["history_turns"] = 1
    orquestador = Orchestrator(config, _persona(), ClienteFalso())
    orquestador.process("uno")
    orquestador.process("dos")
    assert len(orquestador.history) == 2
    assert orquestador.history[0]["content"] == "dos"


def test_reset_history() -> None:
    orquestador = Orchestrator(_config(), _persona(), ClienteFalso())
    orquestador.process("hola")
    orquestador.reset_history()
    assert orquestador.history == []


def test_historial_se_envia_al_modelo() -> None:
    cliente = ClienteFalso()
    orquestador = Orchestrator(_config(), _persona(), cliente)
    orquestador.process("primero")
    orquestador.process("segundo")
    mensajes = cliente.llamadas[1]["messages"]
    contenidos = [m["content"] for m in mensajes]
    assert "primero" in contenidos
    assert '{"response":"Hmph.","emotion":"neutral"}' in contenidos


# ---------------------------------------------------------------------------
# Router y herramientas
# ---------------------------------------------------------------------------
def test_router_ejecuta_herramienta(tmp_path: Path) -> None:
    cliente = ClienteFalso([{"response": "Ya lo hice.", "emotion": "fun"}])
    router = RouterFalso(
        ToolCall(
            tool="falsa",
            arguments={"texto": "dato"},
            confidence=0.9,
            reason="prueba",
        )
    )
    orquestador = Orchestrator(
        _config(router={"enabled": True, "max_tool_rounds": 1}),
        _persona(),
        cliente,
        registry=_registro(tmp_path),
        router=router,
    )
    resultado = orquestador.process("usa la herramienta")
    assert len(resultado.tool_calls) == 1
    assert resultado.tool_results[0].content == "resultado:dato"
    sistema = cliente.llamadas[0]["messages"][0]["content"]
    assert "Resultado de la herramienta 'falsa'" in sistema
    assert "resultado:dato" in sistema


def test_router_herramienta_fallida_se_explica(tmp_path: Path) -> None:
    cliente = ClienteFalso()
    router = RouterFalso(
        ToolCall(tool="inexistente", arguments={}, confidence=0.9)
    )
    orquestador = Orchestrator(
        _config(router={"enabled": True, "max_tool_rounds": 1}),
        _persona(),
        cliente,
        registry=_registro(tmp_path),
        router=router,
    )
    resultado = orquestador.process("usa algo")
    assert resultado.tool_results[0].ok is False
    sistema = cliente.llamadas[0]["messages"][0]["content"]
    assert "falló" in sistema


def test_router_deshabilitado_no_se_llama(tmp_path: Path) -> None:
    router = RouterFalso(
        ToolCall(tool="falsa", arguments={"texto": "x"}, confidence=0.9)
    )
    orquestador = Orchestrator(
        _config(),
        _persona(),
        ClienteFalso(),
        registry=_registro(tmp_path),
        router=router,
    )
    resultado = orquestador.process("hola")
    assert resultado.tool_calls == []
    assert router.consultas == []


def test_router_roto_no_tumba_el_turno(tmp_path: Path) -> None:
    class RouterRoto:
        def route(self, texto: str, catalogo, **kwargs: Any) -> ToolCall:
            raise RuntimeError("modelo caído")

    orquestador = Orchestrator(
        _config(router={"enabled": True, "max_tool_rounds": 1}),
        _persona(),
        ClienteFalso(),
        registry=_registro(tmp_path),
        router=RouterRoto(),
    )
    resultado = orquestador.process("hola")
    assert resultado.response
    assert resultado.tool_calls == []


def test_sin_registro_no_hay_herramientas() -> None:
    orquestador = Orchestrator(
        _config(router={"enabled": True, "max_tool_rounds": 1}),
        _persona(),
        ClienteFalso(),
        router=RouterFalso(ToolCall(tool="falsa", arguments={})),
    )
    resultado = orquestador.process("hola")
    assert resultado.tool_calls == []


# ---------------------------------------------------------------------------
# Reintentos de red
# ---------------------------------------------------------------------------
def test_process_with_retry_reintenta() -> None:
    cliente = ClienteFalso(
        [OllamaError("sin conexión"), {"response": "Ahora sí.", "emotion": "joy"}]
    )
    orquestador = Orchestrator(_config(), _persona(), cliente)
    resultado = orquestador.process_with_retry("hola", intentos=2)
    assert resultado.response == "Ahora sí."
    assert len(cliente.llamadas) == 2


def test_process_with_retry_agota_intentos() -> None:
    cliente = ClienteFalso([OllamaError("sin conexión")])
    orquestador = Orchestrator(_config(), _persona(), cliente)
    with pytest.raises(OllamaError):
        orquestador.process_with_retry("hola", intentos=1)
