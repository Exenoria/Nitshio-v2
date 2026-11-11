"""Tests del aviso de fallback y de la salida cruda en el pipeline de texto."""

from __future__ import annotations

from brain.llm.orchestrator import TurnResult
from brain.llm.parser import fallback_response, parse_response
from pipeline.text import TextPipeline

EMOCIONES = ["neutral", "fun", "angry", "joy", "sorrow", "surprised"]


def _turno(raw: str) -> TurnResult:
    parsed = fallback_response(raw, EMOCIONES)
    return TurnResult(
        user_text="hola",
        response=parsed.data.response,
        emotion=parsed.data.emotion,
        parsed=parsed,
        raw_responses=[raw],
    )


def test_fallback_imprime_advertencia(capsys) -> None:
    pipeline = TextPipeline(None)  # type: ignore[arg-type]

    pipeline.imprimir(_turno("esto no es JSON"))

    _, err = capsys.readouterr()
    assert "ADVERTENCIA" in err
    assert "fallback" in err.lower()
    assert "[crudo" not in err


def test_debug_muestra_respuesta_cruda(capsys) -> None:
    pipeline = TextPipeline(None, debug=True)  # type: ignore[arg-type]

    pipeline.imprimir(_turno("esto no es JSON"))

    _, err = capsys.readouterr()
    assert "ADVERTENCIA" in err
    assert "[crudo 1] esto no es JSON" in err


def test_respuesta_valida_no_avisa(capsys) -> None:
    parsed = parse_response(
        '{"response": "Hmph.", "emotion": "neutral"}', EMOCIONES
    )
    turno = TurnResult(
        user_text="hola",
        response=parsed.data.response,
        emotion=parsed.data.emotion,
        parsed=parsed,
        raw_responses=[parsed.raw],
    )
    pipeline = TextPipeline(None)  # type: ignore[arg-type]

    pipeline.imprimir(turno)

    _, err = capsys.readouterr()
    assert "ADVERTENCIA" not in err
