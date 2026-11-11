"""Tests de carga de personalidad, few-shots y Modelfile."""

from __future__ import annotations

from pathlib import Path

from brain.config import ROOT, cargar_config
from brain.persona.loader import (
    cargar_persona,
    construir_mensajes,
    construir_system_prompt,
    validar_few_shots,
)
from scripts.build_model import render_modelfile


def test_persona_real_carga() -> None:
    config = cargar_config()
    persona = cargar_persona(config)
    assert persona.name == "Nitshio"
    assert "tsundere" in persona.system_prompt.lower()
    assert "doble fachada" in persona.system_prompt.lower()
    assert len(persona.few_shots) == 4
    assert "Ejemplos de tono" not in persona.system_prompt
    assert "angry" in persona.emotions
    assert persona.default_emotion == "neutral"


def test_few_shots_validos() -> None:
    config = cargar_config()
    persona = cargar_persona(config)
    assert validar_few_shots(persona) == []


def test_mensajes_incluyen_sistema_y_usuario() -> None:
    config = cargar_config()
    persona = cargar_persona(config)
    mensajes = construir_mensajes(persona, "Hola")
    assert mensajes[0]["role"] == "system"
    assert "Nitshio" in mensajes[0]["content"]
    assert mensajes[-1] == {"role": "user", "content": "Hola"}
    # Los few-shot se incluyen como pares user/assistant.
    roles = [m["role"] for m in mensajes]
    assert roles.count("assistant") >= len(persona.few_shots)


def test_mensajes_pueden_omitir_few_shots_si_se_configura() -> None:
    from dataclasses import replace

    config = cargar_config()
    persona = replace(cargar_persona(config), include_few_shots_in_runtime=False)
    mensajes = construir_mensajes(persona, "Hola")
    roles = [m["role"] for m in mensajes]
    assert roles.count("assistant") == 0


def test_contexto_se_inyecta_y_recorta() -> None:
    config = cargar_config()
    persona = cargar_persona(config)
    prompt = construir_system_prompt(
        persona, contexto="Recuerdo: al usuario le gusta el café.", max_context_chars=12000
    )
    assert "café" in prompt
    assert "Contexto de esta conversación" in prompt


def test_modelfile_render() -> None:
    config = cargar_config()
    persona = cargar_persona(config)
    modelfile = render_modelfile(config, persona)
    assert modelfile.startswith("#")
    assert f"FROM {config['ollama']['models']['base']}" in modelfile
    assert "SYSTEM \"\"\"" in modelfile
    assert "MESSAGE user" in modelfile
    assert "MESSAGE assistant" in modelfile
    assert "PARAMETER temperature" in modelfile
    assert "PARAMETER num_ctx" in modelfile


def test_emociones_config_coinciden_con_parser() -> None:
    config = cargar_config()
    persona = cargar_persona(config)
    assert set(persona.emotions) == {
        "neutral",
        "fun",
        "angry",
        "joy",
        "sorrow",
        "surprised",
    }


def test_archivos_de_personalidad_existen() -> None:
    assert (ROOT / "brain" / "persona" / "nitshio_short.md").exists()
    assert (ROOT / "brain" / "persona" / "few_shots_short.jsonl").exists()
    assert (ROOT / "brain" / "persona" / "nitshio.md").exists()
    assert (ROOT / "brain" / "persona" / "few_shots.jsonl").exists()
    assert (ROOT / "config" / "emotions.yaml").exists()


def test_persona_breve_en_rango_de_tokens() -> None:
    config = cargar_config()
    persona = cargar_persona(config)
    # Aproximación conservadora: ~4 caracteres por token en español.
    tokens_estimados = len(persona.system_prompt) / 4
    assert 200 <= tokens_estimados <= 300
