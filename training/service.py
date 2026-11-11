"""Lógica de la UI de entrenamiento, separada de Gradio.

Así los tests pueden cubrir chat, feedback, correcciones, editor del dataset,
reconstrucción del Modelfile y versionado sin levantar una interfaz web.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from brain.config import cargar_config, host_ollama
from brain.llm.ollama_client import OllamaClient, OllamaError
from brain.persona.loader import Persona, cargar_persona, construir_mensajes
from training.build_model import (
    DEFAULT_OUTPUT,
    crear_en_ollama,
    registrar_version,
    render_modelfile,
)
from training.schema import (
    DatasetError,
    DatasetExample,
    load_dataset,
    parse_example,
    save_jsonl,
)
from training.versions import (
    DEFAULT_VERSIONS,
    VersionError,
    activate_version,
    find_version,
    list_versions,
    load_registry,
    rollback_version,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ChatTurn:
    """Un turno del chat de prueba de la UI."""

    user: str
    response: str
    emotion: str
    attempts: int = 1
    repaired: bool = False
    fallback: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class FeedbackResult:
    """Resultado de guardar un 👍/👎 o una corrección."""

    saved_memory: bool
    correction_path: str | None = None
    memory_id: str | None = None
    message: str = ""


class TrainingServiceError(RuntimeError):
    """La operación de la UI no se pudo completar."""


def _ruta(valor: str | Path) -> Path:
    path = Path(valor)
    return path if path.is_absolute() else ROOT / path


class TrainingService:
    """Servicio local que usa la UI y los tests."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        client: OllamaClient | None = None,
        memory: Any | None = None,
        logger: logging.Logger | None = None,
        config_path: Path | str | None = None,
    ) -> None:
        self.config = config or cargar_config()
        self.config_path = (
            _ruta(config_path) if config_path else ROOT / "config" / "config.yaml"
        )
        self.persona: Persona = cargar_persona(self.config)
        self.logger = logger or logging.getLogger("nitshio.training")
        ollama_cfg = self.config.get("ollama") or {}
        self.client = client or OllamaClient(
            host_ollama(self.config),
            timeout=float(ollama_cfg.get("request_timeout_seconds", 180)),
            keep_alive=str(ollama_cfg.get("keep_alive", "30m")),
            think=bool(ollama_cfg.get("think", False)),
        )
        self.model = str((ollama_cfg.get("models") or {}).get("active", ""))
        if not self.model:
            raise TrainingServiceError("Falta ollama.models.active.")
        self.options = dict((ollama_cfg.get("options") or {}).get("primary") or {})
        structured = ollama_cfg.get("structured_output") or {}
        self.retries = int(structured.get("retries", 2))
        self.fallback_emotion = str(
            structured.get("fallback_emotion", self.persona.default_emotion)
        )
        training_cfg = self.config.get("training") or {}
        self.dataset_dir = _ruta(
            training_cfg.get("dataset_dir", "training/datasets")
        )
        self.versions_file = _ruta(
            training_cfg.get("versions_file", str(DEFAULT_VERSIONS))
        )
        self._memory = memory
        self._memory_initialized = memory is not None

    # ------------------------------------------------------------------
    # Memoria
    # ------------------------------------------------------------------
    @property
    def memory(self) -> Any | None:
        if not self._memory_initialized:
            self._memory_initialized = True
            try:
                from memory.store import MemoryStore

                self._memory = MemoryStore(self.config)
            except Exception as exc:
                self.logger.warning("Memoria no disponible en UI: %s", exc)
                self._memory = None
        return self._memory

    def _guardar_memoria(
        self,
        texto: str,
        *,
        metadata: dict[str, Any],
        coleccion: str = "memories",
    ) -> str | None:
        memoria = self.memory
        if memoria is None:
            return None
        try:
            return str(memoria.add(coleccion, texto, metadata=metadata))
        except Exception as exc:
            self.logger.warning("No se pudo guardar feedback en memoria: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Chat de prueba
    # ------------------------------------------------------------------
    def chat(
        self,
        user_text: str,
        *,
        history: Iterable[dict[str, str]] | None = None,
    ) -> ChatTurn:
        """Genera una respuesta con el modelo activo y el mismo formato real."""
        texto = " ".join(str(user_text or "").split()).strip()
        if not texto:
            raise TrainingServiceError("El mensaje está vacío.")
        mensajes = construir_mensajes(
            self.persona,
            texto,
            history=list(history or []),
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
        turno = ChatTurn(
            user=texto,
            response=parsed.data.response,
            emotion=parsed.data.emotion,
            attempts=resultado.attempts,
            repaired=parsed.repaired,
            fallback=parsed.fallback,
            errors=list(parsed.errors),
        )
        self._guardar_memoria(
            f"Usuario: {texto}\nNitshio: {turno.response}",
            metadata={
                "type": "training_ui_turn",
                "emotion": turno.emotion,
                "model": self.model,
            },
            coleccion="conversations",
        )
        return turno

    # ------------------------------------------------------------------
    # Feedback y correcciones
    # ------------------------------------------------------------------
    def save_feedback(
        self,
        user_text: str,
        response: str,
        emotion: str,
        rating: str,
        *,
        corrected_response: str | None = None,
        corrected_emotion: str | None = None,
    ) -> FeedbackResult:
        """Guarda un voto y, si hay corrección, la añade al dataset."""
        if rating not in {"up", "down"}:
            raise TrainingServiceError("rating debe ser 'up' o 'down'.")
        ruta_correccion: Path | None = None
        memoria_id: str | None = None
        mensaje = "Feedback guardado."
        if corrected_response or corrected_emotion:
            respuesta = (corrected_response or response).strip()
            emocion = (corrected_emotion or emotion).strip()
            ejemplo = parse_example(
                {
                    "user": user_text,
                    "response": respuesta,
                    "emotion": emocion,
                },
                self.persona.emotions,
            )
            ruta_correccion = self.dataset_dir / "corrections.jsonl"
            ruta_correccion.parent.mkdir(parents=True, exist_ok=True)
            with ruta_correccion.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        ejemplo.to_record(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            memoria_id = self._guardar_memoria(
                (
                    f"Corrección de Nitshio para «{user_text}»: "
                    f"{ejemplo.response} (emoción {ejemplo.emotion})"
                ),
                metadata={
                    "type": "correction",
                    "rating": rating,
                    "emotion": ejemplo.emotion,
                    "source": "training_ui",
                },
            )
            mensaje = f"Corrección guardada en {ruta_correccion}."
        else:
            memoria_id = self._guardar_memoria(
                f"Feedback {rating} para «{user_text}»: {response}",
                metadata={
                    "type": "feedback",
                    "rating": rating,
                    "emotion": emotion,
                    "source": "training_ui",
                },
            )
        return FeedbackResult(
            saved_memory=memoria_id is not None,
            correction_path=str(ruta_correccion) if ruta_correccion else None,
            memory_id=memoria_id,
            message=mensaje,
        )

    def append_example(
        self,
        user_text: str,
        response: str,
        emotion: str,
        *,
        path: Path | str | None = None,
    ) -> Path:
        """Añade un ejemplo validado al JSONL indicado."""
        ejemplo = parse_example(
            {"user": user_text, "response": response, "emotion": emotion},
            self.persona.emotions,
        )
        destino = _ruta(path) if path else self.dataset_dir / "corrections.jsonl"
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    ejemplo.to_record(), ensure_ascii=False, separators=(",", ":")
                )
                + "\n"
            )
        return destino

    def load_corrections(self) -> list[DatasetExample]:
        ruta = self.dataset_dir / "corrections.jsonl"
        if not ruta.exists():
            return []
        resultado = load_dataset(ruta, self.persona.emotions)
        return resultado.examples

    # ------------------------------------------------------------------
    # Editor del dataset
    # ------------------------------------------------------------------
    def dataset_text(self, path: Path | str | None = None) -> str:
        ruta = _ruta(path) if path else self.dataset_dir / "seed.jsonl"
        if not ruta.exists():
            return ""
        return ruta.read_text(encoding="utf-8")

    def validate_dataset_text(self, texto: str) -> list[str]:
        """Devuelve una lista de errores; vacía si todo es válido."""
        errores: list[str] = []
        for numero, linea in enumerate(str(texto).splitlines(), start=1):
            if not linea.strip():
                continue
            try:
                parse_example(linea, self.persona.emotions, line=numero)
            except DatasetError as exc:
                errores.append(str(exc))
        return errores

    def save_dataset_text(
        self,
        texto: str,
        *,
        path: Path | str | None = None,
    ) -> Path:
        """Valida y guarda el texto editado; no escribe si hay errores."""
        errores = self.validate_dataset_text(texto)
        if errores:
            raise TrainingServiceError(
                "El dataset tiene errores:\n" + "\n".join(errores)
            )
        destino = _ruta(path) if path else self.dataset_dir / "seed.jsonl"
        lineas = [linea for linea in str(texto).splitlines() if linea.strip()]
        ejemplos = [
            parse_example(linea, self.persona.emotions) for linea in lineas
        ]
        save_jsonl(destino, ejemplos)
        return destino

    # ------------------------------------------------------------------
    # Modelo y versiones
    # ------------------------------------------------------------------
    def build_modelfile(
        self,
        version: str,
        *,
        create: bool = False,
        activate: bool = False,
        dataset_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Reconstruye el Modelfile, lo registra y opcionalmente lo activa."""
        modelfile = render_modelfile(self.config, self.persona)
        DEFAULT_OUTPUT.write_text(modelfile, encoding="utf-8")
        registrar_version(
            version,
            str(self.config["ollama"]["models"]["base"]),
            self.persona,
            modelfile,
            activate=False,
            dataset_path=dataset_path,
            notes="Modelo reconstruido desde la UI.",
            versions_file=self.versions_file,
        )
        if create:
            crear_en_ollama(version, DEFAULT_OUTPUT)
        if activate:
            activate_version(
                version,
                registry_path=self.versions_file,
                config_path=self.config_path,
            )
        return {
            "version": version,
            "modelfile": str(DEFAULT_OUTPUT),
            "created": create,
            "activated": activate,
        }

    def versions(self) -> dict[str, Any]:
        return load_registry(self.versions_file)

    def activate(self, version: str) -> dict[str, Any]:
        try:
            return activate_version(
                version,
                registry_path=self.versions_file,
                config_path=self.config_path,
            )
        except VersionError as exc:
            raise TrainingServiceError(str(exc)) from exc

    def rollback(self) -> dict[str, Any]:
        try:
            return rollback_version(
                registry_path=self.versions_file,
                config_path=self.config_path,
            )
        except VersionError as exc:
            raise TrainingServiceError(str(exc)) from exc

    def version_rows(self) -> list[list[str]]:
        data = self.versions()
        activa = data.get("active")
        filas: list[list[str]] = []
        for version in list_versions(data):
            filas.append(
                [
                    str(version.get("name", "")),
                    str(version.get("base", "")),
                    str(version.get("type", "")),
                    "ACTIVA" if version.get("name") == activa else str(
                        version.get("status", "")
                    ),
                    str(version.get("created_at", "")),
                ]
            )
        return filas

    def cloud_plan(self) -> dict[str, str]:
        """Devuelve el comando que se ejecuta en Kaggle/Colab.

        La UI local no lanza QLoRA en la T1200: no hay GPU suficiente ni
        PyTorch CUDA instalado. El botón muestra el comando y el notebook.
        """
        training_cfg = self.config.get("training") or {}
        notebook = str(
            training_cfg.get("cloud", {}).get(
                "notebook", "training/cloud/Nitshio_QLoRA.ipynb"
            )
        )
        dataset = str(
            (training_cfg.get("synthetic") or {}).get(
                "output", "training/datasets/synthetic.jsonl"
            )
        )
        comando = (
            "python training/finetune_qlora.py "
            f"--dataset {dataset} --output-dir outputs/nitshio-lora"
        )
        return {
            "notebook": notebook,
            "dataset": dataset,
            "command": comando,
            "message": (
                "El fine-tune se ejecuta en Kaggle/Colab. Sube el proyecto y "
                "el dataset, abre el notebook y ejecuta las celdas. Localmente "
                "solo se preparan datos y se evalúa el GGUF resultante."
            ),
        }
