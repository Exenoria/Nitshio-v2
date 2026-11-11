#!/usr/bin/env python
"""Construye un Modelfile de Nitshio y, opcionalmente, un modelo en Ollama.

Ejemplos:
    python training/build_model.py --version nitshio-v1 --create --activate
    python training/build_model.py --version nitshio-qwen2b --output models/Modelfile.qwen2b
    python training/build_model.py --version nitshio-v2 --print
    python training/build_model.py --version nitshio-v2 --gguf models/nitshio-v2.gguf --create
    python training/build_model.py --version nitshio-v3 --create

El script valida los few-shots antes de compilar y no hardcodea modelos: el
modelo base, la familia y los parámetros se leen de ``config/config.yaml``.
Con ``--gguf`` el ``FROM`` apunta a un GGUF local, se conserva la plantilla de
Phi y la versión se registra con ``type=gguf``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.config import cargar_config
from brain.persona.loader import PersonaError, cargar_persona, validar_few_shots
from training.versions import (
    DEFAULT_VERSIONS,
    VersionError,
    activate_version,
    actualizar_config,
    register_version,
)

DEFAULT_TEMPLATE = ROOT / "models" / "Modelfile.template"
DEFAULT_OUTPUT = ROOT / "models" / "Modelfile"
VERSIONS_FILE = DEFAULT_VERSIONS


def _quoted(texto: str) -> str:
    """Devuelve un string entre comillas compatible con Modelfile."""
    return json.dumps(texto, ensure_ascii=False)


def _from_value(valor: str) -> str:
    """Normaliza el valor de ``FROM`` para un tag o un GGUF local.

    Los tags se dejan intactos. Las rutas ``.gguf`` se resuelven a absolutas y
    se escriben con barras normales. Si la ruta contiene espacios se
    entrecomilla, que es lo que acepta el parser de Modelfile de Ollama.
    """
    path = Path(valor).expanduser()
    if path.suffix.lower() != ".gguf":
        return valor
    ruta = path.resolve().as_posix()
    if any(caracter.isspace() for caracter in ruta):
        return f'"{ruta}"'
    return ruta


def familia_base(config: dict[str, Any]) -> str:
    """Devuelve la familia declarada en ``ollama.models.base_family``.

    Si la clave no existe se asume ``phi`` para no romper configuraciones
    antiguas. La familia decide si se conservan el TEMPLATE y los stop de Phi.
    """
    modelos = (config.get("ollama") or {}).get("models") or {}
    return str(modelos.get("base_family", "phi")).strip().lower()


def _resolver_bloque(render: str, nombre: str, *, incluir: bool) -> str:
    """Conserva o elimina un bloque del template delimitado por marcadores.

    Los marcadores tienen la forma ``# >>> NOMBRE`` / ``# <<< NOMBRE``. Al
    conservar el bloque se quitan los comentarios; al eliminarlo se quita todo
    el contenido, incluidos los marcadores.
    """
    if incluir:
        patron = re.compile(
            rf"(?m)^[ \t]*# (?:>>>|<<<) {re.escape(nombre)}[ \t]*\n"
        )
        return patron.sub("", render)
    patron = re.compile(
        rf"(?ms)^[ \t]*# >>> {re.escape(nombre)}[ \t]*\n"
        rf".*?^[ \t]*# <<< {re.escape(nombre)}[ \t]*(?:\n|$)"
    )
    return patron.sub("", render)


def _usar_plantilla_phi(config: dict[str, Any], base_override: str | None) -> bool:
    """Indica si el Modelfile debe llevar TEMPLATE/stop de Phi.

    Los GGUF locales conservan la plantilla Phi por compatibilidad con los
    modelos ya exportados. Para tags, la decisión depende de la familia
    configurada: solo ``phi*`` incluye la plantilla.
    """
    if base_override and Path(base_override).suffix.lower() == ".gguf":
        return True
    return familia_base(config).startswith("phi")


def render_modelfile(
    config: dict[str, Any],
    persona: Any,
    template_path: Path | None = None,
    *,
    base_override: str | None = None,
) -> str:
    """Renderiza el Modelfile completo a partir de config y persona.

    ``base_override`` permite sustituir el tag base por la ruta de un GGUF
    local. Los bloques PHI_TEMPLATE y PHI_STOPS solo se conservan si la familia
    configurada es ``phi`` o si ``base_override`` es un ``.gguf``.
    """
    ruta_template = Path(template_path) if template_path else DEFAULT_TEMPLATE
    if not ruta_template.exists():
        raise FileNotFoundError(f"No existe el template: {ruta_template}")
    template = ruta_template.read_text(encoding="utf-8")

    modelos = config["ollama"]["models"]
    base = _from_value(base_override) if base_override else str(modelos["base"])
    system = persona.system_prompt
    if '"""' in system:
        raise ValueError(
            "El prompt de personalidad contiene triple comilla; cambia el "
            "delimitador del SYSTEM en el template."
        )

    opciones = dict(config["ollama"].get("options", {}).get("primary", {}))
    lineas_parametros: list[str] = []
    for clave in ("temperature", "top_p", "repeat_penalty", "num_ctx", "num_predict"):
        if clave in opciones:
            lineas_parametros.append(f"PARAMETER {clave} {opciones[clave]}")

    mensajes: list[str] = []
    for ejemplo in persona.few_shots:
        mensajes.append(f"MESSAGE user {_quoted(ejemplo.user)}")
        mensajes.append(f"MESSAGE assistant {_quoted(ejemplo.assistant_json())}")

    render = template
    render = render.replace("__BASE_MODEL__", base)
    render = render.replace("__SYSTEM__", system)
    render = render.replace("__PARAMETERS__", "\n".join(lineas_parametros))
    render = render.replace("__MESSAGES__", "\n".join(mensajes))

    incluir_phi = _usar_plantilla_phi(config, base_override)
    for bloque in ("PHI_TEMPLATE", "PHI_STOPS"):
        render = _resolver_bloque(render, bloque, incluir=incluir_phi)
    return render


def registrar_version(
    version: str,
    base: str,
    persona: Any,
    modelfile: str,
    *,
    activate: bool = False,
    dataset_path: Path | str | None = None,
    notes: str = "",
    versions_file: Path | str = VERSIONS_FILE,
    version_type: str = "modelfile",
    gguf_path: Path | str | None = None,
) -> dict[str, Any]:
    """Registra la versión en ``models/versions.json``.

    ``version_type`` es ``"modelfile"`` para un modelo construido desde un tag
    de Ollama o ``"gguf"`` para uno construido desde un GGUF local.
    """
    if not notes:
        notes = (
            "Modelo GGUF local."
            if version_type == "gguf"
            else "Modelo construido desde Modelfile."
        )
    return register_version(
        version,
        base,
        type=version_type,
        persona=persona,
        modelfile=modelfile,
        dataset_path=dataset_path,
        gguf_path=str(gguf_path) if gguf_path else None,
        notes=notes,
        activate=activate,
        registry_path=versions_file,
    )


def activar_en_config(version: str, config_path: Path | None = None) -> None:
    """Cambia ``ollama.models.active`` sin reescribir todo el YAML."""
    actualizar_config(version, config_path or (ROOT / "config" / "config.yaml"))


def crear_en_ollama(version: str, modelfile_path: Path) -> None:
    """Ejecuta ``ollama create`` y muestra su salida."""
    if not modelfile_path.exists():
        raise FileNotFoundError(modelfile_path)
    comando = ["ollama", "create", version, "-f", str(modelfile_path)]
    print("Ejecutando:", " ".join(comando))
    try:
        subprocess.run(comando, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No se encontró el comando 'ollama'. Instala Ollama y asegúrate "
            "de que esté en PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ollama create falló con código {exc.returncode}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Construye nitshio-vN.")
    parser.add_argument(
        "--version",
        default="nitshio-v1",
        help="Nombre del modelo/versión (ej. nitshio-v1).",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Ejecuta `ollama create` después de escribir el Modelfile.",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Actualiza ollama.models.active en config/config.yaml.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Imprime el Modelfile en vez de escribirlo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Ruta del Modelfile generado (default: models/Modelfile). "
            "Úsala para no sobrescribir borradores locales."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Ruta alternativa a config.yaml.",
    )
    parser.add_argument(
        "--versions-file",
        type=Path,
        default=VERSIONS_FILE,
        help="Registro de versiones alternativo (útil para pruebas).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset usado; se guarda su hash en versions.json.",
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        default=None,
        help=(
            "GGUF local: usa FROM <ruta> con el TEMPLATE de Phi-3.5 y "
            "registra la versión con type=gguf."
        ),
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Nota libre para el registro de versión.",
    )
    args = parser.parse_args(argv)

    try:
        config = cargar_config(args.config)
        persona = cargar_persona(config)
    except (OSError, ValueError, PersonaError) as exc:
        print(f"ERROR de configuración/personalidad: {exc}", file=sys.stderr)
        return 1

    problemas = validar_few_shots(persona)
    if problemas:
        print("ERROR: los few-shots no son válidos:", file=sys.stderr)
        for problema in problemas:
            print(f"  - {problema}", file=sys.stderr)
        return 1

    gguf_path: Path | None = None
    version_type = "modelfile"
    base_override: str | None = None
    if args.gguf:
        gguf_path = args.gguf.expanduser().resolve()
        if not gguf_path.is_file():
            print(f"ERROR: no existe el GGUF: {gguf_path}", file=sys.stderr)
            return 1
        if gguf_path.suffix.lower() != ".gguf":
            print(f"ERROR: el archivo no es .gguf: {gguf_path}", file=sys.stderr)
            return 1
        version_type = "gguf"
        base_override = str(gguf_path)

    try:
        modelfile = render_modelfile(
            config, persona, base_override=base_override
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR al renderizar: {exc}", file=sys.stderr)
        return 1

    if args.print:
        print(modelfile)
        return 0

    output_path = (
        args.output
        if args.output.is_absolute()
        else ROOT / args.output
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(modelfile, encoding="utf-8")
    print(f"Modelfile escrito en {output_path}")

    try:
        registrar_version(
            args.version,
            str(config["ollama"]["models"]["base"]),
            persona,
            modelfile,
            activate=False,
            dataset_path=args.dataset,
            notes=args.notes,
            version_type=version_type,
            gguf_path=gguf_path,
            versions_file=args.versions_file,
        )
    except (OSError, VersionError) as exc:
        print(f"ERROR al registrar la versión: {exc}", file=sys.stderr)
        return 1
    print(f"Versión registrada: {args.version}")

    if args.create:
        try:
            crear_en_ollama(args.version, output_path)
        except (OSError, RuntimeError) as exc:
            print(f"ERROR al crear el modelo: {exc}", file=sys.stderr)
            return 1
        print(f"Modelo creado en Ollama: {args.version}")

    if args.activate:
        try:
            activate_version(
                args.version,
                config_path=args.config or (ROOT / "config" / "config.yaml"),
            )
        except (OSError, VersionError) as exc:
            print(f"ERROR al activar: {exc}", file=sys.stderr)
            return 1
        print(f"Modelo activo en config: {args.version}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
