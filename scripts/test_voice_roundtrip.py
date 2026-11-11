#!/usr/bin/env python
"""Prueba integral del pipeline de voz con audio Piper inyectado.

Cadena que se valida:

    texto -> Piper (voz del usuario) -> micrófono falso -> Moonshine real
    -> orquestador real (Ollama + memoria) -> TTS Piper -> CABLE Input
    -> avatar VMC (emoción)

El audio de respuesta se graba de ``CABLE Output`` para comprobar que llegó al
cable virtual que VSeeFace usa como micrófono. No sustituye la prueba humana de
``run.py --mode voice``, pero cierra la lógica del Paso 2 sin depender de que
una persona hable.

Uso:
    python scripts/test_voice_roundtrip.py
    python scripts/test_voice_roundtrip.py --text "Hola Nitshio, ¿cómo estás?"
    python scripts/test_voice_roundtrip.py --no-tools --no-memory
    python scripts/test_voice_roundtrip.py --no-speak --no-cable
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.config import cargar_config, host_ollama
from brain.llm.ollama_client import OllamaClient, OllamaError
from brain.llm.orchestrator import crear_orquestador
from brain.persona.loader import cargar_persona
from stt.base import crear_stt
from tts.base import AudioChunk
from tts.piper_provider import PiperProvider

SALIDA_AUDIO = ROOT / "data" / "cache" / "audio" / "voice_roundtrip_cable.wav"
FRECUENCIA_STT = 16000


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def leer_wav(datos: bytes) -> tuple[np.ndarray, int]:
    """Convierte un WAV PCM 16-bit en un array mono float32."""
    with wave.open(io.BytesIO(datos), "rb") as wav:
        canales = wav.getnchannels()
        ancho = wav.getsampwidth()
        frecuencia = wav.getframerate()
        crudo = wav.readframes(wav.getnframes())
    if ancho != 2:
        raise ValueError(f"Solo se admite PCM 16-bit; llegó {ancho * 8}-bit.")
    audio = np.frombuffer(crudo, dtype=np.int16).astype(np.float32) / 32768.0
    if canales > 1:
        audio = audio.reshape(-1, canales).mean(axis=1)
    return audio.astype(np.float32), int(frecuencia)


def remuestrear(
    audio: np.ndarray, origen: int, destino: int
) -> np.ndarray:
    """Remuestreo lineal suficiente para alimentar al STT."""
    if origen == destino or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    muestras = int(round(audio.size * destino / origen))
    if muestras <= 1:
        return np.zeros(0, dtype=np.float32)
    x_viejo = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    x_nuevo = np.linspace(0.0, 1.0, muestras, endpoint=False)
    return np.interp(x_nuevo, x_viejo, audio).astype(np.float32)


def guardar_wav(audio: np.ndarray, frecuencia: int, ruta: Path) -> None:
    """Guarda la evidencia de audio sin depender de scipy."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos = np.clip(audio, -1.0, 1.0)
    pcm = (datos * 32767.0).astype(np.int16)
    with wave.open(str(ruta), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(frecuencia)
        wav.writeframes(pcm.tobytes())


def rms_ventanas(
    audio: np.ndarray, frecuencia: int, ventana_s: float = 0.25
) -> tuple[float, float]:
    """Devuelve RMS global y el máximo RMS por ventana."""
    if audio.size == 0:
        return 0.0, 0.0
    total = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    tam = max(1, int(frecuencia * ventana_s))
    maximo = 0.0
    for inicio in range(0, audio.size, tam):
        bloque = audio[inicio : inicio + tam]
        if bloque.size:
            valor = float(np.sqrt(np.mean(np.square(bloque, dtype=np.float64))))
            maximo = max(maximo, valor)
    return total, maximo


# ---------------------------------------------------------------------------
# Micrófono falso y grabador de CABLE Output
# ---------------------------------------------------------------------------
class MicrofonoFalso:
    """Entrega una única utterance ya sintetizada y luego se calla."""

    def __init__(self, audio: np.ndarray, sample_rate: int = FRECUENCIA_STT):
        self.audio = audio
        self.sample_rate = int(sample_rate)
        self.entregado = False
        self.cerrado = 0

    def listen_once(self, timeout_s: float | None = None) -> np.ndarray | None:
        if self.entregado:
            return None
        self.entregado = True
        return self.audio

    def utterances(self, stop_event: threading.Event | None = None):
        if not self.entregado:
            self.entregado = True
            yield self.audio

    def close(self) -> None:
        self.cerrado += 1


def buscar_entrada(nombre: str) -> int | None:
    """Busca el primer dispositivo de entrada cuyo nombre contenga ``nombre``."""
    import sounddevice as sd

    for indice, dispositivo in enumerate(sd.query_devices()):
        if int(dispositivo.get("max_input_channels", 0)) <= 0:
            continue
        if nombre.lower() in str(dispositivo.get("name", "")).lower():
            return indice
    return None


class GrabadorCable:
    """Graba CABLE Output en un hilo para verificar la salida del TTS."""

    def __init__(
        self,
        nombre: str = "CABLE Output",
        frecuencia: int = 48000,
        canales: int = 2,
    ) -> None:
        self.nombre = nombre
        self.frecuencia = frecuencia
        self.canales = canales
        self.dispositivo: int | None = None
        self.error: Exception | None = None
        self._bloques: list[np.ndarray] = []
        self._stop = threading.Event()
        self._hilo: threading.Thread | None = None

    def start(self) -> None:
        self._hilo = threading.Thread(
            target=self._run, name="nitshio-cable-grab", daemon=True
        )
        self._hilo.start()

    def _run(self) -> None:
        try:
            import sounddevice as sd

            self.dispositivo = buscar_entrada(self.nombre)
            if self.dispositivo is None:
                raise RuntimeError(
                    f"No se encontró el dispositivo de entrada {self.nombre!r}."
                )
            with sd.InputStream(
                device=self.dispositivo,
                samplerate=self.frecuencia,
                channels=self.canales,
                dtype="float32",
                blocksize=1024,
            ) as stream:
                while not self._stop.is_set():
                    datos, _ = stream.read(1024)
                    self._bloques.append(
                        np.asarray(datos, dtype=np.float32).copy()
                    )
        except Exception as exc:  # pragma: no cover - depende del equipo
            self.error = exc

    def stop(self) -> None:
        self._stop.set()
        if self._hilo is not None:
            self._hilo.join(timeout=5)
            self._hilo = None

    def audio(self) -> np.ndarray:
        if not self._bloques:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._bloques)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------
def vseeface_activo() -> bool:
    """Indica si el proceso de VSeeFace está corriendo."""
    try:
        import psutil

        for proceso in psutil.process_iter(["name"]):
            nombre = str(proceso.info.get("name") or "").lower()
            if "vseeface" in nombre:
                return True
    except Exception:
        pass
    return False


def _crear_cliente(config: dict[str, Any]) -> OllamaClient:
    ollama_cfg = config.get("ollama") or {}
    return OllamaClient(
        host_ollama(config),
        timeout=float(ollama_cfg.get("request_timeout_seconds", 180)),
        keep_alive=str(ollama_cfg.get("keep_alive", "30m")),
        think=bool(ollama_cfg.get("think", False)),
    )


def _imprimir_resumen(
    *,
    texto_usuario: str,
    transcript: Any,
    turn: Any,
    chunks: list[AudioChunk],
    emociones: list[str],
    cable_rms: tuple[float, float] | None,
    ruta_audio: Path | None,
) -> dict[str, Any]:
    duracion_tts = sum(chunk.duration_s for chunk in chunks)
    resumen = {
        "texto_usuario": texto_usuario,
        "transcripcion": transcript.text,
        "stt": transcript.provider,
        "duracion_stt_s": transcript.duration_s,
        "respuesta": turn.response,
        "emocion": turn.emotion,
        "intentos": turn.attempts,
        "herramientas": [llamada.tool for llamada in turn.tool_calls],
        "chunks_tts": len(chunks),
        "duracion_tts_s": round(duracion_tts, 2),
        "emociones_enviadas": emociones,
        "cable_rms": round(cable_rms[0], 6) if cable_rms else None,
        "cable_rms_max": round(cable_rms[1], 6) if cable_rms else None,
        "audio_guardado": str(ruta_audio) if ruta_audio else None,
    }

    print("\n" + "=" * 64)
    print("  RESULTADO DE LA PRUEBA INTEGRAL DE VOZ")
    print("=" * 64)
    print(f"Texto inyectado:  {texto_usuario}")
    print(f"Transcripción:    {transcript.text!r} ({transcript.provider})")
    print(f"Duración STT:     {transcript.duration_s:.2f} s")
    print(f"Respuesta:        {turn.response}")
    print(f"Emoción:          {turn.emotion}")
    print(f"Intentos JSON:    {turn.attempts}")
    if turn.tool_calls:
        print(
            "Herramientas:     "
            + ", ".join(llamada.tool for llamada in turn.tool_calls)
        )
    print(f"Chunks TTS:       {len(chunks)} ({duracion_tts:.2f} s)")
    print(f"Emociones VMC:    {emociones}")
    if cable_rms is not None:
        print(
            f"CABLE Output:     RMS {cable_rms[0]:.6f} | "
            f"máx {cable_rms[1]:.6f}"
        )
    if ruta_audio is not None:
        print(f"Audio guardado:   {ruta_audio}")
    print("=" * 64)
    return resumen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prueba integral de voz con audio Piper inyectado."
    )
    parser.add_argument(
        "--text",
        default="Hola Nitshio, ¿cómo estás?",
        help="Frase que se sintetiza como si la dijera el usuario.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--no-avatar", action="store_true")
    parser.add_argument("--no-speak", action="store_true")
    parser.add_argument(
        "--no-cable",
        action="store_true",
        help="No graba CABLE Output (útil si VB-Cable no está instalado).",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("nitshio.test_voice")

    try:
        config = cargar_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"ERROR de configuración: {exc}", file=sys.stderr)
        return 1
    if args.model:
        config["ollama"]["models"]["active"] = args.model

    try:
        persona = cargar_persona(config)
    except Exception as exc:
        print(f"ERROR de personalidad: {exc}", file=sys.stderr)
        return 1

    cliente = _crear_cliente(config)
    try:
        version = cliente.health().get("version", "?")
        logger.info("Ollama OK (v%s), modelo %s", version, config["ollama"]["models"].get("active"))
    except OllamaError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Abre Ollama o ejecuta `ollama serve`.", file=sys.stderr)
        return 1

    piper = PiperProvider(config)
    orquestador = None
    pipeline = None
    stt = None
    grabador: GrabadorCable | None = None
    try:
        # 1. Sintetizar la frase del usuario con la misma voz local.
        logger.info("Sintetizando la frase del usuario con Piper...")
        chunk_usuario = piper.synthesize(args.text)
        audio_usuario, frecuencia_usuario = leer_wav(chunk_usuario.data)
        audio_usuario = remuestrear(
            audio_usuario, frecuencia_usuario, FRECUENCIA_STT
        )
        silencio = np.zeros(int(FRECUENCIA_STT * 0.2), dtype=np.float32)
        audio_usuario = np.concatenate(
            [silencio, audio_usuario, silencio]
        ).astype(np.float32)
        logger.info(
            "Audio inyectado: %.2f s a %d Hz",
            audio_usuario.size / FRECUENCIA_STT,
            FRECUENCIA_STT,
        )

        # 2. STT real y orquestador real.
        stt = crear_stt(config)
        orquestador = crear_orquestador(
            config,
            persona,
            cliente,
            con_memoria=not args.no_memory,
            con_herramientas=not args.no_tools,
            con_avatar=not args.no_avatar,
            con_tts=not args.no_speak,
            logger=logger,
        )
        if not args.no_avatar and orquestador.avatar is None:
            print("ERROR: el avatar no se pudo crear.", file=sys.stderr)
            return 1
        if not args.no_speak and orquestador.tts is None:
            print("ERROR: el TTS no se pudo crear.", file=sys.stderr)
            return 1

        emociones_enviadas: list[str] = []
        if orquestador.avatar is not None:
            original_set = orquestador.avatar.set_emotion

            def set_emocion(nombre: str) -> None:
                emociones_enviadas.append(nombre)
                original_set(nombre)

            orquestador.avatar.set_emotion = set_emocion  # type: ignore[method-assign]
            orquestador.avatar.start()

        chunks_tts: list[AudioChunk] = []
        if orquestador.tts is not None:
            original_speak = orquestador.tts.speak

            def speak_con_registro(
                texto: str, **kwargs: Any
            ) -> list[AudioChunk]:
                anterior = kwargs.get("on_chunk")

                def on_chunk(fragmento: str, audio: AudioChunk) -> None:
                    chunks_tts.append(audio)
                    if anterior is not None:
                        anterior(fragmento, audio)

                kwargs["on_chunk"] = on_chunk
                return original_speak(texto, **kwargs)

            orquestador.tts.speak = speak_con_registro  # type: ignore[method-assign]

        if not args.no_cable:
            grabador = GrabadorCable()
            grabador.start()
            time.sleep(0.3)

        from pipeline.voice import VoicePipeline

        pipeline = VoicePipeline(
            config,
            orquestador,
            stt=stt,
            tts=orquestador.tts,
            microphone=MicrofonoFalso(audio_usuario),
        )

        logger.info("Ejecutando VoicePipeline.run_once(speak=%s)...", not args.no_speak)
        evento = pipeline.run_once(speak=not args.no_speak)
        if evento is None:
            print("ERROR: el pipeline no devolvió ningún evento.", file=sys.stderr)
            return 1

        # Dar tiempo a que el último chunk llegue al cable y al avatar.
        time.sleep(0.8)
        if grabador is not None:
            grabador.stop()

        cable_rms = None
        ruta_audio = None
        if grabador is not None:
            audio_cable = grabador.audio()
            if grabador.error is not None:
                logger.warning("No se pudo grabar CABLE Output: %s", grabador.error)
            elif audio_cable.size:
                cable_rms = rms_ventanas(audio_cable, grabador.frecuencia)
                ruta_audio = SALIDA_AUDIO
                guardar_wav(audio_cable, grabador.frecuencia, ruta_audio)
            else:
                logger.warning("CABLE Output no devolvió muestras.")

        resumen = _imprimir_resumen(
            texto_usuario=args.text,
            transcript=evento.transcript,
            turn=evento.turn,
            chunks=chunks_tts,
            emociones=emociones_enviadas,
            cable_rms=cable_rms,
            ruta_audio=ruta_audio,
        )

        # 3. Comprobaciones mínimas.
        fallos: list[str] = []
        if not evento.transcript.text.strip():
            fallos.append("la transcripción quedó vacía")
        if not evento.turn.response.strip():
            fallos.append("la respuesta quedó vacía")
        if evento.turn.emotion not in persona.emotions:
            fallos.append(f"emoción fuera de persona: {evento.turn.emotion}")
        if not args.no_speak and not chunks_tts:
            fallos.append("el TTS no generó chunks")
        if not args.no_avatar and evento.turn.emotion not in emociones_enviadas:
            fallos.append("el avatar no recibió la emoción del turno")
        if cable_rms is not None and cable_rms[1] < 0.001:
            fallos.append(
                "CABLE Output no registró audio del TTS "
                f"(RMS máx {cable_rms[1]:.6f})"
            )
        if not args.no_avatar and not vseeface_activo():
            logger.warning(
                "VSeeFace no está corriendo; la emoción se envió por UDP "
                "pero no se pudo ver visualmente."
            )

        if args.json:
            print(json.dumps(resumen, ensure_ascii=False, indent=2))

        if fallos:
            print("\nFALLOS:", file=sys.stderr)
            for fallo in fallos:
                print(f"  - {fallo}", file=sys.stderr)
            return 1
        print("\nPrueba integral OK.")
        return 0
    except Exception as exc:
        logger.exception("La prueba integral falló")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if grabador is not None:
            grabador.stop()
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception:
                pass
        if stt is not None:
            try:
                stt.close()
            except Exception:
                pass
        if orquestador is not None:
            if orquestador.avatar is not None:
                try:
                    orquestador.avatar.close()
                except Exception:
                    pass
            if orquestador.tts is not None:
                try:
                    orquestador.tts.close()
                except Exception:
                    pass
        try:
            piper.close()
        except Exception:
            pass
        try:
            cliente.session.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
