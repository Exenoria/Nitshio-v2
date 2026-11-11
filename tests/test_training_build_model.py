"""Tests de construcción de Modelfile, incluido el FROM de un GGUF local."""

from __future__ import annotations

import copy
from pathlib import Path

from brain.config import cargar_config
from brain.persona.loader import cargar_persona
from training.build_model import (
    familia_base,
    main,
    registrar_version,
    render_modelfile,
)
from training.versions import load_registry


def _gguf(tmp_path: Path, nombre: str = "nitshio-v2.gguf") -> Path:
    ruta = tmp_path / nombre
    ruta.write_bytes(b"GGUF")
    return ruta


def test_render_modelfile_gguf_usa_from_local(tmp_path: Path) -> None:
    config = cargar_config()
    persona = cargar_persona(config)
    gguf = _gguf(tmp_path)

    modelfile = render_modelfile(config, persona, base_override=str(gguf))

    assert f"FROM {gguf.resolve().as_posix()}" in modelfile
    assert 'TEMPLATE """' in modelfile
    assert "PARAMETER stop" in modelfile
    assert "MESSAGE user" in modelfile
    assert "MESSAGE assistant" in modelfile


def test_render_modelfile_gguf_con_espacios(tmp_path: Path) -> None:
    config = cargar_config()
    persona = cargar_persona(config)
    gguf = _gguf(tmp_path, "modelo con espacios.gguf")

    modelfile = render_modelfile(config, persona, base_override=str(gguf))

    assert f'FROM "{gguf.resolve().as_posix()}"' in modelfile


def test_registrar_version_gguf(tmp_path: Path) -> None:
    config = cargar_config()
    persona = cargar_persona(config)
    gguf = _gguf(tmp_path)
    versions = tmp_path / "versions.json"
    modelfile = render_modelfile(config, persona, base_override=str(gguf))

    entrada = registrar_version(
        "nitshio-v2",
        str(config["ollama"]["models"]["base"]),
        persona,
        modelfile,
        versions_file=versions,
        version_type="gguf",
        gguf_path=gguf,
    )

    assert entrada["type"] == "gguf"
    assert entrada["gguf_path"] == str(gguf)
    assert entrada["base"] == str(config["ollama"]["models"]["base"])
    data = load_registry(versions)
    assert data["versions"][0]["type"] == "gguf"


def test_main_gguf_print(tmp_path: Path, capsys) -> None:
    gguf = _gguf(tmp_path)

    codigo = main(["--gguf", str(gguf), "--print"])

    assert codigo == 0
    salida = capsys.readouterr().out
    assert f"FROM {gguf.resolve().as_posix()}" in salida
    assert 'TEMPLATE """' in salida


def test_main_gguf_inexistente(tmp_path: Path, capsys) -> None:
    codigo = main(["--gguf", str(tmp_path / "no-existe.gguf"), "--print"])

    assert codigo == 1
    assert "no existe el GGUF" in capsys.readouterr().err


def test_main_gguf_extension_incorrecta(tmp_path: Path, capsys) -> None:
    ruta = tmp_path / "modelo.bin"
    ruta.write_bytes(b"no es gguf")

    codigo = main(["--gguf", str(ruta), "--print"])

    assert codigo == 1
    assert "no es .gguf" in capsys.readouterr().err


def test_familia_base_qwen() -> None:
    config = cargar_config()
    assert familia_base(config) == "qwen"


def test_modelfile_qwen_omite_template_y_stops_phi() -> None:
    config = cargar_config()
    persona = cargar_persona(config)
    modelfile = render_modelfile(config, persona)

    assert f"FROM {config['ollama']['models']['base']}" in modelfile
    assert not any(
        linea.startswith("TEMPLATE") for linea in modelfile.splitlines()
    )
    assert not any(
        linea.startswith("PARAMETER stop") for linea in modelfile.splitlines()
    )
    assert "PHI_TEMPLATE" in modelfile  # solo en el comentario explicativo
    assert "MESSAGE user" in modelfile


def test_modelfile_phi_conserva_template_y_stops() -> None:
    config = copy.deepcopy(cargar_config())
    config["ollama"]["models"]["base_family"] = "phi"
    persona = cargar_persona(config)
    modelfile = render_modelfile(config, persona)

    assert any(
        linea.startswith('TEMPLATE """') for linea in modelfile.splitlines()
    )
    assert modelfile.count("PARAMETER stop") == 4
    assert "# >>> PHI_TEMPLATE" not in modelfile
    assert "# <<< PHI_TEMPLATE" not in modelfile


def test_main_output_no_sobrescribe_default(tmp_path: Path) -> None:
    salida = tmp_path / "Modelfile.qwen2b"
    versions = tmp_path / "versions.json"

    codigo = main(
        [
            "--version",
            "nitshio-test-output",
            "--output",
            str(salida),
            "--versions-file",
            str(versions),
        ]
    )

    assert codigo == 0
    assert salida.exists()
    contenido = salida.read_text(encoding="utf-8")
    assert "FROM qwen3.5:2b" in contenido
    assert not any(
        linea.startswith("TEMPLATE") for linea in contenido.splitlines()
    )
    data = load_registry(versions)
    assert data["versions"][0]["name"] == "nitshio-test-output"
