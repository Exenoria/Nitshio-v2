"""Tests del reintento/fallback del cliente Ollama (sin red real)."""

from __future__ import annotations

from brain.llm.ollama_client import OllamaClient

EMOCIONES = ["neutral", "fun", "angry", "joy", "sorrow", "surprised"]


def _cliente_con_respuestas(respuestas: list[str]) -> OllamaClient:
    cliente = OllamaClient("http://localhost:11434")
    iterador = iter(respuestas)
    llamadas: list[dict] = []

    def fake_chat(model, messages, **kwargs):
        llamadas.append({"model": model, "messages": messages, **kwargs})
        return {"message": {"content": next(iterador)}}

    cliente.chat = fake_chat  # type: ignore[method-assign]
    cliente.llamadas = llamadas  # type: ignore[attr-defined]
    return cliente


def test_reintenta_y_valida() -> None:
    cliente = _cliente_con_respuestas(
        [
            "esto no es json",
            '{"response": "Ahora sí, pendejo.", "emotion": "angry"}',
        ]
    )
    resultado = cliente.chat_structured(
        "modelo",
        [{"role": "user", "content": "Hola"}],
        allowed_emotions=EMOCIONES,
        retries=2,
    )
    assert resultado.attempts == 2
    assert resultado.parsed.fallback is False
    assert resultado.parsed.data.emotion == "angry"
    assert len(resultado.raw_responses) == 2
    # El segundo intento debe incluir la respuesta inválida y la corrección.
    assert len(cliente.llamadas) == 2  # type: ignore[attr-defined]
    segundo = cliente.llamadas[1]["messages"]  # type: ignore[attr-defined]
    assert any("no cumple el formato" in m["content"] for m in segundo)


def test_fallback_tras_agotar_reintentos() -> None:
    cliente = _cliente_con_respuestas(["mal", "peor", "sigue mal"])
    resultado = cliente.chat_structured(
        "modelo",
        [{"role": "user", "content": "Hola"}],
        allowed_emotions=EMOCIONES,
        retries=2,
        fallback_emotion="neutral",
    )
    assert resultado.attempts == 3
    assert resultado.parsed.fallback is True
    assert resultado.parsed.data.emotion == "neutral"
    assert resultado.parsed.data.response


def test_reparacion_sin_reintento() -> None:
    cliente = _cliente_con_respuestas(
        ['{"response": "Hola", "emotion": "neutral",}']
    )
    resultado = cliente.chat_structured(
        "modelo",
        [{"role": "user", "content": "Hola"}],
        allowed_emotions=EMOCIONES,
        retries=2,
    )
    assert resultado.attempts == 1
    assert resultado.parsed.repaired is True
    assert resultado.parsed.data.response == "Hola"


def test_schema_enviado_a_ollama() -> None:
    cliente = _cliente_con_respuestas(
        ['{"response": "Hola", "emotion": "neutral"}']
    )
    cliente.chat_structured(
        "modelo",
        [{"role": "user", "content": "Hola"}],
        allowed_emotions=EMOCIONES,
        retries=0,
    )
    llamada = cliente.llamadas[0]  # type: ignore[attr-defined]
    assert llamada["schema"]["properties"]["emotion"]["enum"] == EMOCIONES


class _RespuestaFalsa:
    def __init__(self, data: dict) -> None:
        self._data = data
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._data


class _SesionFalsa:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.payload: dict | None = None

    def post(self, url: str, json: dict, timeout: float) -> _RespuestaFalsa:
        self.payload = json
        return _RespuestaFalsa(self.data)


def test_chat_envia_think_false_por_defecto() -> None:
    sesion = _SesionFalsa({"message": {"content": "Hola"}})
    cliente = OllamaClient("http://localhost:11434", session=sesion)  # type: ignore[arg-type]

    cliente.chat("modelo", [{"role": "user", "content": "Hola"}])

    assert sesion.payload is not None
    assert sesion.payload["think"] is False


def test_chat_envia_think_configurado() -> None:
    sesion = _SesionFalsa({"message": {"content": "Hola"}})
    cliente = OllamaClient(  # type: ignore[arg-type]
        "http://localhost:11434", think=True, session=sesion
    )

    cliente.chat("modelo", [{"role": "user", "content": "Hola"}])

    assert sesion.payload is not None
    assert sesion.payload["think"] is True


def test_chat_ignora_campos_de_razonamiento() -> None:
    sesion = _SesionFalsa(
        {
            "message": {
                "content": "Hola",
                "thinking": "cadena de pensamiento",
                "reasoning_content": "otra cadena",
            },
            "thinking": "nivel superior",
            "reasoning": "más razonamiento",
        }
    )
    cliente = OllamaClient("http://localhost:11434", session=sesion)  # type: ignore[arg-type]

    data = cliente.chat("modelo", [{"role": "user", "content": "Hola"}])

    assert data["message"]["content"] == "Hola"
    assert "thinking" not in data
    assert "reasoning" not in data
    assert "reasoning_content" not in data
    assert "thinking" not in data["message"]
    assert "reasoning_content" not in data["message"]


def test_contenido_no_usa_razonamiento_si_falta_content() -> None:
    assert (
        OllamaClient._contenido(
            {"message": {"thinking": "solo pensamiento"}}
        )
        == ""
    )


def test_chat_structured_propaga_think() -> None:
    cliente = _cliente_con_respuestas(
        ['{"response": "Hola", "emotion": "neutral"}']
    )
    cliente.think = True

    cliente.chat_structured(
        "modelo",
        [{"role": "user", "content": "Hola"}],
        allowed_emotions=EMOCIONES,
        retries=0,
    )

    assert cliente.llamadas[0]["think"] is True  # type: ignore[attr-defined]
