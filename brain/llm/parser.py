"""Parser y validación de la salida JSON de Nitshio.

Responsabilidades:
- Extraer un objeto JSON aunque el modelo agregue texto o bloques de código.
- Reparar errores comunes (comas finales, comillas tipográficas, saltos de
  línea sin escapar).
- Validar con Pydantic que existan ``response`` y ``emotion``.
- Comprobar que la emoción esté declarada en ``config/emotions.yaml``.
- Producir un fallback seguro si todo lo anterior falla.

El parser no llama a Ollama; el reintento automático vive en
``brain/llm/ollama_client.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:  # json-repair es opcional: si no está, usamos la reparación propia.
    from json_repair import repair_json as _json_repair_externo
except ImportError:  # pragma: no cover - depende del entorno
    _json_repair_externo = None


class ParseError(ValueError):
    """La respuesta no se pudo interpretar como un NitshioResponse válido."""


FALLBACK_WARNING = (
    "ADVERTENCIA: el modelo no devolvió un JSON válido; se usó la respuesta "
    "de emergencia (fallback). Ejecuta con --debug para ver la respuesta cruda."
)


class NitshioResponse(BaseModel):
    """Formato exacto que debe devolver el modelo de personalidad."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    response: str = Field(min_length=1, max_length=4000)
    emotion: str = Field(min_length=1, max_length=32)


@dataclass
class ParsedResponse:
    """Resultado del parser con metadatos útiles para logs y evaluación."""

    data: NitshioResponse
    raw: str
    repaired: bool = False
    fallback: bool = False
    extra_text: bool = False
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL
)


def _quitar_cercas(texto: str) -> str:
    """Quita un bloque ```json ... ``` si el modelo lo agregó."""
    match = _FENCE_RE.match(texto)
    if match:
        return match.group(1).strip()
    return texto.strip()


def extract_json_object(texto: str) -> tuple[str, bool]:
    """Extrae el primer objeto JSON balanceado de un texto.

    Devuelve ``(objeto, habia_texto_extra)``. Respeta cadenas y escapes para no
    confundir una llave dentro de un string con el cierre del objeto.
    """
    if not isinstance(texto, str) or not texto.strip():
        raise ParseError("La respuesta está vacía.")

    limpio = _quitar_cercas(texto.lstrip("\ufeff").strip())
    inicio = limpio.find("{")
    if inicio < 0:
        raise ParseError("No se encontró ningún objeto JSON.")

    profundidad = 0
    en_string = False
    escape = False
    for indice in range(inicio, len(limpio)):
        caracter = limpio[indice]
        if en_string:
            if escape:
                escape = False
            elif caracter == "\\":
                escape = True
            elif caracter == '"':
                en_string = False
            continue
        if caracter == '"':
            en_string = True
        elif caracter == "{":
            profundidad += 1
        elif caracter == "}":
            profundidad -= 1
            if profundidad == 0:
                objeto = limpio[inicio : indice + 1]
                extra = limpio[:inicio].strip() != "" or limpio[indice + 1 :].strip() != ""
                return objeto, extra

    raise ParseError("El objeto JSON está incompleto (faltan llaves).")


# ---------------------------------------------------------------------------
# Reparación
# ---------------------------------------------------------------------------
def _normalizar_comillas(texto: str) -> str:
    """Cambia comillas tipográficas por comillas rectas."""
    return (
        texto.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("«", '"')
        .replace("»", '"')
    )


def _escapar_controles_en_strings(texto: str) -> str:
    """Escapa saltos de línea y tabs literales dentro de strings JSON."""
    salida: list[str] = []
    en_string = False
    escape = False
    for caracter in texto:
        if en_string:
            if escape:
                salida.append(caracter)
                escape = False
                continue
            if caracter == "\\":
                salida.append(caracter)
                escape = True
                continue
            if caracter == '"':
                salida.append(caracter)
                en_string = False
                continue
            if caracter == "\n":
                salida.append("\\n")
                continue
            if caracter == "\r":
                salida.append("\\r")
                continue
            if caracter == "\t":
                salida.append("\\t")
                continue
            if ord(caracter) < 0x20:
                salida.append(f"\\u{ord(caracter):04x}")
                continue
            salida.append(caracter)
        else:
            salida.append(caracter)
            if caracter == '"':
                en_string = True
    return "".join(salida)


def _quitar_comas_finales(texto: str) -> str:
    """Elimina comas antes de } o ] (error JSON muy común)."""
    return re.sub(r",\s*([}\]])", r"\1", texto)


def repair_json_text(texto: str) -> str:
    """Intenta convertir un texto casi-JSON en JSON parseable.

    Estrategia conservadora:
    1. Extraer el objeto.
    2. Normalizar comillas y comas finales.
    3. Escapar controles dentro de strings.
    4. Intentar ``json.loads``.
    5. Si existe ``json-repair``, usarlo como segunda pasada.
    """
    objeto, _ = extract_json_object(texto)
    candidato = _normalizar_comillas(objeto)
    candidato = _quitar_comas_finales(candidato)
    candidato = _escapar_controles_en_strings(candidato)

    try:
        json.loads(candidato)
        return candidato
    except json.JSONDecodeError:
        pass

    if _json_repair_externo is not None:
        try:
            reparado = _json_repair_externo(candidato, return_objects=False)
            if isinstance(reparado, str):
                json.loads(reparado)
                return reparado
        except Exception:
            pass

    return candidato


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------
def _validar_emocion(emocion: str, permitidas: Iterable[str]) -> str:
    """Normaliza y valida la emoción contra la lista configurada."""
    permitidas_lista = [str(e).strip().lower() for e in permitidas]
    normalizada = str(emocion).strip().lower()
    if normalizada not in permitidas_lista:
        raise ParseError(
            f"Emoción inválida: {emocion!r}. "
            f"Permitidas: {', '.join(permitidas_lista)}"
        )
    return normalizada


def parse_response(
    texto: str, allowed_emotions: Iterable[str]
) -> ParsedResponse:
    """Extrae, repara y valida una respuesta del modelo.

    Lanza ``ParseError`` si no se puede obtener un objeto con ``response`` y
    ``emotion`` válidos. El llamador decide si reintenta o usa fallback.
    """
    errores: list[str] = []
    try:
        objeto_texto, extra = extract_json_object(texto)
    except ParseError as exc:
        raise ParseError(str(exc)) from exc

    reparado = False
    try:
        datos = json.loads(objeto_texto)
    except json.JSONDecodeError as exc:
        errores.append(f"JSON inválido: {exc}")
        try:
            objeto_texto = repair_json_text(texto)
            datos = json.loads(objeto_texto)
            reparado = True
        except (json.JSONDecodeError, ParseError) as exc2:
            raise ParseError(f"No se pudo reparar el JSON: {exc2}") from exc2

    if not isinstance(datos, dict):
        raise ParseError("La raíz del JSON no es un objeto.")

    try:
        respuesta = NitshioResponse.model_validate(datos)
    except ValidationError as exc:
        raise ParseError(f"El JSON no cumple el esquema: {exc}") from exc

    emocion = _validar_emocion(respuesta.emotion, allowed_emotions)
    respuesta = respuesta.model_copy(update={"emotion": emocion})

    if not respuesta.response.strip():
        raise ParseError("El campo 'response' está vacío.")

    return ParsedResponse(
        data=respuesta,
        raw=texto,
        repaired=reparado,
        extra_text=extra,
        errors=errores,
    )


def fallback_response(
    texto: str,
    allowed_emotions: Iterable[str],
    fallback_emotion: str = "neutral",
) -> ParsedResponse:
    """Construye una respuesta segura cuando el parser no logra validar.

    Intenta rescatar el campo ``response`` aunque el resto del JSON esté roto.
    Si no hay nada rescatable, usa un mensaje corto en personaje.
    """
    permitidas = list(allowed_emotions)
    emocion = fallback_emotion if fallback_emotion in permitidas else permitidas[0]

    # 1. Intento normal.
    try:
        return parse_response(texto, permitidas)
    except ParseError:
        pass

    rescatado: str | None = None

    # 2. Intentar rescatar "response" con regex aunque el JSON esté roto.
    match = re.search(
        r'"response"\s*:\s*"((?:\\.|[^"\\])*)"', texto or "", re.DOTALL
    )
    if match:
        try:
            rescatado = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            rescatado = match.group(1)

    # 3. Si el texto no parece JSON, usar el texto limpio.
    if not rescatado:
        limpio = _quitar_cercas(texto or "")
        if limpio and not limpio.lstrip().startswith("{"):
            rescatado = limpio

    if not rescatado:
        rescatado = "Hmph... no tengo ganas de responder eso ahora."

    rescatado = " ".join(str(rescatado).split()).strip()[:1000]
    if not rescatado:
        rescatado = "Hmph... no tengo ganas de responder eso ahora."

    return ParsedResponse(
        data=NitshioResponse(response=rescatado, emotion=emocion),
        raw=texto or "",
        fallback=True,
        extra_text=True,
        errors=["Se usó el fallback seguro."],
    )


def response_schema(allowed_emotions: Iterable[str]) -> dict[str, Any]:
    """JSON Schema que se envía a Ollama con ``format``."""
    return {
        "type": "object",
        "properties": {
            "response": {"type": "string", "minLength": 1},
            "emotion": {
                "type": "string",
                "enum": [str(e).strip().lower() for e in allowed_emotions],
            },
        },
        "required": ["response", "emotion"],
        "additionalProperties": False,
    }
