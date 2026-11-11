"""Tests unitarios de las funciones puras de scripts/check_models.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts import check_models


def test_tps_usa_eval_duration() -> None:
    data = {"eval_count": 100, "eval_duration": 2_000_000_000}
    assert check_models._tps(data, wall_seconds=99) == pytest.approx(50.0)


def test_tps_fallback_wall_time() -> None:
    data = {"eval_count": 20, "eval_duration": 0}
    assert check_models._tps(data, wall_seconds=4.0) == pytest.approx(5.0)


def test_json_valido_ok() -> None:
    ok, error = check_models._json_valido('{"response": "hola", "emotion": "joy"}')
    assert ok is True
    assert error is None


def test_json_valido_rechaza_texto_extra() -> None:
    ok, error = check_models._json_valido('{"response": "hola"} basura')
    assert ok is False
    assert error


def test_json_valido_rechaza_array() -> None:
    ok, error = check_models._json_valido("[]")
    assert ok is False
    assert "objeto" in (error or "")


def test_cargar_config_ok(tmp_path: Path) -> None:
    cfg = {
        "ollama": {"models": {"base": "modelo-base"}},
        "hardware": {"min_tokens_per_second": 8},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert check_models.cargar_config(path)["ollama"]["models"]["base"] == "modelo-base"


def test_cargar_config_falta_seccion(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("ollama:\n  models: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hardware"):
        check_models.cargar_config(path)


def test_veredicto_faltantes() -> None:
    veredicto, sugerencias = check_models.veredicto_y_sugerencias(
        faltantes=["modelo-x"],
        perfiles=[],
        embeddings=None,
        config={"hardware": {}},
        gpu=None,
        ram={"percent": 10},
        cargados=[],
    )
    assert veredicto.startswith("NO SOPORTADO")
    assert sugerencias


def test_veredicto_soportado() -> None:
    perfiles = [
        {
            "rol": "base",
            "modelo": "m",
            "tokens_por_segundo": 25,
            "latencia_s": 2,
            "json_valido": True,
        },
        {
            "rol": "router",
            "modelo": "r",
            "tokens_por_segundo": 80,
            "latencia_s": 1,
            "json_valido": True,
        },
    ]
    veredicto, _ = check_models.veredicto_y_sugerencias(
        faltantes=[],
        perfiles=perfiles,
        embeddings={"modelo": "e", "latencia_s": 0.2, "dimensiones": 768},
        config={
            "hardware": {
                "min_tokens_per_second": 8,
                "comfortable_tokens_per_second": 20,
                "min_router_tokens_per_second": 20,
            }
        },
        gpu={"name": "GPU", "vram_used_mb": 2000, "vram_total_mb": 4096, "util_percent": 10},
        ram={"percent": 50},
        cargados=[{"processor": "GPU"}],
    )
    assert veredicto.startswith("SOPORTADO")


def test_veredicto_ajustado() -> None:
    perfiles = [
        {
            "rol": "base",
            "modelo": "m",
            "tokens_por_segundo": 10,
            "latencia_s": 4,
            "json_valido": True,
        },
        {
            "rol": "router",
            "modelo": "r",
            "tokens_por_segundo": 80,
            "latencia_s": 1,
            "json_valido": True,
        },
    ]
    veredicto, _ = check_models.veredicto_y_sugerencias(
        faltantes=[],
        perfiles=perfiles,
        embeddings=None,
        config={
            "hardware": {
                "min_tokens_per_second": 8,
                "comfortable_tokens_per_second": 20,
                "min_router_tokens_per_second": 20,
            }
        },
        gpu={"name": "GPU", "vram_used_mb": 2000, "vram_total_mb": 4096, "util_percent": 10},
        ram={"percent": 50},
        cargados=[{"processor": "GPU"}],
    )
    assert veredicto.startswith("AJUSTADO")


def test_tag_latest_implicito() -> None:
    assert check_models._normalizar_tag("nitshio-v1") == "nitshio-v1:latest"
    assert check_models._normalizar_tag("phi3.5:q4") == "phi3.5:q4"
    assert check_models._esta_instalado(
        "nitshio-v1", ["nitshio-v1:latest", "otro:latest"]
    )
    assert not check_models._esta_instalado("nitshio-v2", ["nitshio-v1:latest"])


def test_config_real_tiene_modelos() -> None:
    root = Path(__file__).resolve().parents[1]
    config = check_models.cargar_config(root / "config" / "config.yaml")
    modelos = config["ollama"]["models"]
    for rol in ("base", "active", "router", "embeddings", "vision"):
        assert modelos.get(rol), f"Falta el modelo {rol}"


def test_json_salida_es_serializable() -> None:
    # Protege contra objetos no serializables en --json.
    data = {"a": 1, "b": [1, 2], "c": {"d": "ñ"}}
    assert json.loads(json.dumps(data, ensure_ascii=False)) == data


def test_parar_modelo_ejecuta_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    llamadas: list[list[str]] = []

    def fake_run(comando: list[str], **_: object) -> object:
        llamadas.append(comando)
        return object()

    monkeypatch.setattr(
        check_models.shutil, "which", lambda _nombre: "C:/ollama.exe"
    )
    monkeypatch.setattr(check_models.subprocess, "run", fake_run)

    check_models.parar_modelo("nitshio-v1")

    assert llamadas == [["C:/ollama.exe", "stop", "nitshio-v1"]]


def test_parar_modelo_sin_ollama_no_falla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(check_models.shutil, "which", lambda _nombre: None)
    check_models.parar_modelo("nitshio-v1")


def test_parar_modelo_ignora_errores(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_: object, **__: object) -> object:
        raise OSError("boom")

    monkeypatch.setattr(
        check_models.shutil, "which", lambda _nombre: "C:/ollama.exe"
    )
    monkeypatch.setattr(check_models.subprocess, "run", fake_run)

    check_models.parar_modelo("nitshio-v1")


class _OllamaResidencia:
    def __init__(
        self, cargados: list[dict], *, error: Exception | None = None
    ) -> None:
        self.cargados = cargados
        self.error = error
        self.llamadas: list[str] = []

    def generate(self, modelo: str, prompt: str, **kwargs: object) -> dict:
        self.llamadas.append(modelo)
        if self.error:
            raise self.error
        return {}

    def loaded(self) -> list[dict]:
        return self.cargados


def _gpu(total: int = 4096, used: int = 200) -> dict:
    return {
        "name": "T1200",
        "vram_total_mb": total,
        "vram_used_mb": used,
        "util_percent": 10,
    }


def _cargado(nombre: str, size_gb: float, vram_gb: float) -> dict:
    return {
        "name": nombre,
        "size": int(size_gb * 1024**3),
        "size_vram": int(vram_gb * 1024**3),
    }


def test_residencia_simultanea_caben_juntos() -> None:
    ollama = _OllamaResidencia(
        [
            _cargado("qwen3.5:2b", 2.0, 2.0),
            _cargado("qwen3.5:0.8b", 1.0, 1.0),
        ]
    )

    resultado = check_models.probar_residencia_simultanea(
        ollama,  # type: ignore[arg-type]
        {"base": "qwen3.5:2b", "router": "qwen3.5:0.8b"},
        keep_alive="5m",
        gpu=_gpu(),
    )

    assert resultado["cargados_juntos"] is True
    assert resultado["caben_juntos"] is True
    assert resultado["combinado_vram_mb"] == pytest.approx(3072.0)
    assert len(ollama.llamadas) == 2


def test_residencia_simultanea_router_en_cpu_no_cabe() -> None:
    ollama = _OllamaResidencia(
        [
            _cargado("qwen3.5:2b", 2.0, 2.0),
            _cargado("qwen3.5:0.8b", 1.0, 0.0),
        ]
    )

    resultado = check_models.probar_residencia_simultanea(
        ollama,  # type: ignore[arg-type]
        {"base": "qwen3.5:2b", "router": "qwen3.5:0.8b"},
        keep_alive="5m",
        gpu=_gpu(),
    )

    assert resultado["cargados_juntos"] is True
    assert resultado["caben_juntos"] is False
    assert "CPU" in resultado["motivo"]


def test_residencia_simultanea_ollama_expulsa_uno() -> None:
    ollama = _OllamaResidencia([_cargado("qwen3.5:2b", 2.0, 2.0)])

    resultado = check_models.probar_residencia_simultanea(
        ollama,  # type: ignore[arg-type]
        {"base": "qwen3.5:2b", "router": "qwen3.5:0.8b"},
        keep_alive="5m",
        gpu=_gpu(),
    )

    assert resultado["cargados_juntos"] is False
    assert resultado["caben_juntos"] is False
    assert "no mantuvo ambos" in resultado["motivo"]


def test_residencia_simultanea_sin_gpu() -> None:
    ollama = _OllamaResidencia(
        [
            _cargado("qwen3.5:2b", 2.0, 0.0),
            _cargado("qwen3.5:0.8b", 1.0, 0.0),
        ]
    )

    resultado = check_models.probar_residencia_simultanea(
        ollama,  # type: ignore[arg-type]
        {"base": "qwen3.5:2b", "router": "qwen3.5:0.8b"},
        keep_alive="5m",
        gpu=None,
    )

    assert resultado["caben_juntos"] is False
    assert "GPU" in resultado["motivo"]


def test_residencia_simultanea_error_de_carga() -> None:
    ollama = _OllamaResidencia([], error=check_models.OllamaError("boom"))

    resultado = check_models.probar_residencia_simultanea(
        ollama,  # type: ignore[arg-type]
        {"base": "qwen3.5:2b", "router": "qwen3.5:0.8b"},
        keep_alive="5m",
        gpu=_gpu(),
    )

    assert resultado["cargados_juntos"] is False
    assert "No se pudo cargar" in resultado["motivo"]


def test_veredicto_propone_router_en_cpu_y_mismo_modelo() -> None:
    perfiles = [
        {
            "rol": "base",
            "modelo": "qwen3.5:2b",
            "tokens_por_segundo": 25,
            "latencia_s": 2,
            "json_valido": True,
        },
        {
            "rol": "router",
            "modelo": "qwen3.5:0.8b",
            "tokens_por_segundo": 80,
            "latencia_s": 1,
            "json_valido": True,
        },
    ]
    residencia = {
        "roles": {"base": "qwen3.5:2b", "router": "qwen3.5:0.8b"},
        "modelos": [{"name": "qwen3.5:2b"}, {"name": "qwen3.5:0.8b"}],
        "cargados_juntos": True,
        "caben_juntos": False,
        "motivo": "La suma de VRAM supera el límite.",
    }

    veredicto, sugerencias = check_models.veredicto_y_sugerencias(
        faltantes=[],
        perfiles=perfiles,
        embeddings=None,
        config={
            "hardware": {
                "min_tokens_per_second": 8,
                "comfortable_tokens_per_second": 20,
                "min_router_tokens_per_second": 20,
            }
        },
        gpu=_gpu(),
        ram={"percent": 50},
        cargados=[],
        residencia=residencia,
    )

    assert veredicto.startswith("AJUSTADO")
    assert any("num_gpu: 0" in sugerencia for sugerencia in sugerencias)
    assert any("mismo modelo" in sugerencia for sugerencia in sugerencias)
