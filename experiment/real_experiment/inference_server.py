import argparse
import base64
import binascii
import hashlib
import io
import json
import os
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import get_window, resample_poly

SAMPLE_RATE = 48000
MODEL_SAMPLE_RATE = 16000
MODEL_SAMPLE_RATIO = SAMPLE_RATE // MODEL_SAMPLE_RATE
IMAGE_SIZE = 128
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
MAX_AUDIO_SECONDS = 10
AUDIO_BATCH_STEPS = 30
SERVER_PACKAGE = Path(__file__).resolve().parent
SERVER_ROOT = SERVER_PACKAGE.parent
CLIENT_PACKAGE = SERVER_ROOT.parent
CHECKPOINT_DIR = SERVER_PACKAGE / "checkpoints"
DEFAULT_RECORD_DIR = SERVER_PACKAGE / "inference_records"

# Use the same packaged RGB+audio policy for Python and HTTP inference.
sys.path.insert(0, str(SERVER_ROOT))
from real_experiment.policy import AVNavRealPolicy, DEFAULT_FOUNDATION, DEFAULT_POLICY  # noqa: E402


class AVNavInferencePolicy(AVNavRealPolicy):
    """HTTP-compatible name for the validation-selected RGB+audio policy."""


def stereo_pcm_to_spectrogram(audio_pcm, sample_rate=SAMPLE_RATE):
    """Create the [65, 26, 2] SoundSpaces feature from one stereo audio second.

    SoundSpaces computes a centered 512-point STFT at 16 kHz, averages each
    4-by-4 block of magnitudes, and applies log1p. The robot sends 48 kHz int16
    PCM, so this function first resamples it to the model's 16 kHz rate.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError("This service expects audio sampled at {} Hz".format(SAMPLE_RATE))

    audio = np.asarray(audio_pcm)
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("audio must have shape [samples, 2]")
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32, copy=False)

    # The model input remains one second: use the latest second and left-pad
    # startup buffers shorter than one second.
    window_samples = SAMPLE_RATE
    if audio.shape[0] > window_samples:
        audio = audio[-window_samples:]
    elif audio.shape[0] < window_samples:
        audio = np.pad(audio, ((window_samples - audio.shape[0], 0), (0, 0)))
    audio = resample_poly(audio, up=1, down=MODEL_SAMPLE_RATIO, axis=0).astype(
        np.float32, copy=False
    )

    n_fft = 512
    hop_length = 160
    win_length = 400
    window = get_window("hann", win_length, fftbins=True).astype(np.float32)
    window = np.pad(window, ((n_fft - win_length) // 2,
                             n_fft - win_length - (n_fft - win_length) // 2))
    features = []
    for channel in audio.T:
        centered = np.pad(channel, (n_fft // 2, n_fft // 2), mode="constant")
        frames = np.lib.stride_tricks.sliding_window_view(centered, n_fft)[::hop_length]
        magnitude = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).T

        # Match skimage.measure.block_reduce(..., block_size=(4, 4), func=np.mean),
        # including its zero padding at the bottom/right edges.
        rows, columns = magnitude.shape
        padded = np.pad(
            magnitude,
            ((0, (-rows) % 4), (0, (-columns) % 4)),
            mode="constant",
        )
        reduced = padded.reshape(padded.shape[0] // 4, 4,
                                 padded.shape[1] // 4, 4).mean(axis=(1, 3))
        features.append(np.log1p(reduced).astype(np.float32))

    spectrogram = np.stack(features, axis=-1)
    if spectrogram.shape != (65, 26, 2):
        raise ValueError("preprocessing produced unexpected spectrogram shape {}".format(
            spectrogram.shape))
    return spectrogram


def prepare_observation(audio_pcm, rgb, depth=None, sample_rate=SAMPLE_RATE):
    """Prepare robot sensor inputs in the AVNav model's observation format."""
    spectrogram = stereo_pcm_to_spectrogram(audio_pcm, sample_rate=sample_rate)
    rgb_model_input = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return {"spectrogram": spectrogram, "rgb": rgb_model_input}



def decode_base64(value, field):
    if not isinstance(value, str):
        raise ValueError("{} must be a base64 string".format(field))
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("{} is not valid base64".format(field)) from exc


def decode_observation(payload):
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    session_id = payload.get("session_id")
    request_id = payload.get("request_id")
    sequence = payload.get("sequence")
    if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128:
        raise ValueError("session_id must be a string of 1 to 128 characters")
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 160:
        raise ValueError("request_id must be a string of 1 to 160 characters")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    if payload.get("sample_rate") != SAMPLE_RATE:
        raise ValueError("sample_rate must be {}".format(SAMPLE_RATE))
    if payload.get("audio_dtype") != "int16-le":
        raise ValueError("audio_dtype must be int16-le")
    audio_shape = payload.get("audio_shape")
    if (not isinstance(audio_shape, list) or len(audio_shape) != 2 or
            any(isinstance(d, bool) or not isinstance(d, int) for d in audio_shape) or
            audio_shape[0] < 1 or audio_shape[0] > SAMPLE_RATE * MAX_AUDIO_SECONDS or
            audio_shape[1] != 2):
        raise ValueError("audio_shape must be [samples, 2] with at most 10 seconds")
    audio_bytes = decode_base64(payload.get("audio_base64"), "audio_base64")
    if len(audio_bytes) != audio_shape[0] * 2 * 2:
        raise ValueError("audio payload byte length does not match audio_shape")
    audio = np.frombuffer(audio_bytes, dtype="<i2").reshape(audio_shape).copy()

    if payload.get("rgb_format") != "png":
        raise ValueError("rgb_format must be png")
    rgb_bytes = decode_base64(payload.get("rgb_base64"), "rgb_base64")
    bgr = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode RGB PNG")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max(rgb.shape[:2]) > MAX_IMAGE_SIDE:
        raise ValueError("RGB image dimensions exceed {} pixels".format(MAX_IMAGE_SIDE))
    rgb_shape = payload.get("rgb_shape")
    if rgb_shape != list(rgb.shape):
        raise ValueError("rgb_shape does not match decoded RGB image")

    depth = None
    if any(field in payload for field in ("depth_shape", "depth_dtype", "depth_base64")):
        depth_shape = payload.get("depth_shape")
        if depth_shape != list(rgb.shape[:2]):
            raise ValueError("depth_shape must match the RGB image height and width")
        if payload.get("depth_dtype") != "float32-le":
            raise ValueError("depth_dtype must be float32-le")
        # Validate legacy client payloads; depth never enters the RGB+audio model.
        depth_bytes = decode_base64(payload.get("depth_base64"), "depth_base64")
        expected_depth_bytes = rgb.shape[0] * rgb.shape[1] * np.dtype("<f4").itemsize
        if len(depth_bytes) != expected_depth_bytes:
            raise ValueError("depth payload byte length does not match depth_shape")
        depth = np.frombuffer(depth_bytes, dtype="<f4").reshape(depth_shape).copy()
    return session_id, sequence, request_id, audio, rgb, depth


class ObservationArchive:
    """Save each accepted observation and the policy action returned for it."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.images_dir = self.directory / "images"
        self.audio_dir = self.directory / "audio"
        self.audio_batches_dir = self.directory / "audio_batches"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.audio_batches_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save_observation(self, session_id, sequence, request_id, audio, rgb, action_id):
        session_key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
        request_key = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:12]
        stem = "{}_s{}_q{}_{}_a{}".format(
            time.time_ns(), session_key, sequence, request_key, int(action_id)
        )

        success, encoded_rgb = cv2.imencode(
            ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
        if not success:
            raise ValueError("could not encode received RGB image for archiving")
        image_path = self.images_dir / (stem + ".png")

        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(np.asarray(audio, dtype="<i2").tobytes())
        audio_path = self.audio_dir / (stem + ".wav")

        # Write through temporary files so a partially written sensor file is
        # never mistaken for a complete observation after a server interruption.
        self._write_atomic(image_path, encoded_rgb.tobytes())
        self._write_atomic(audio_path, audio_buffer.getvalue())

        record = {
            "timestamp_ns": time.time_ns(),
            "session_id": session_id,
            "sequence": sequence,
            "request_id": request_id,
            "action_id": int(action_id),
            "image": str(image_path.relative_to(self.directory)),
            "audio": str(audio_path.relative_to(self.directory)),
        }
        self._append_jsonl(self.directory / "requests.jsonl", record)
        return record

    def save_audio_batch(self, session_id, step_observations):
        """Save concatenated PCM and the original per-step archive references."""
        if not step_observations:
            raise ValueError("audio batch must contain at least one step")
        sequences = [int(item[0]) for item in step_observations]
        if any(right != left + 1 for left, right in zip(sequences, sequences[1:])):
            raise ValueError("audio batch sequences must be consecutive")

        combined_audio = np.concatenate(
            [item[1] for item in step_observations], axis=0
        )
        session_key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
        start_sequence, end_sequence = sequences[0], sequences[-1]
        stem = "s{}_q{:06d}-{:06d}_{}".format(
            session_key, start_sequence, end_sequence, time.time_ns()
        )
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(np.asarray(combined_audio, dtype="<i2").tobytes())
        audio_path = self.audio_batches_dir / (stem + ".wav")
        self._write_atomic(audio_path, audio_buffer.getvalue())

        record = {
            "timestamp_ns": time.time_ns(),
            "session_id": session_id,
            "start_sequence": start_sequence,
            "end_sequence": end_sequence,
            "sequences": sequences,
            "audio": str(audio_path.relative_to(self.directory)),
            "samples": int(combined_audio.shape[0]),
            "source_steps": [
                {
                    "sequence": int(sequence),
                    "request_id": observation["request_id"],
                    "action_id": int(observation["action_id"]),
                    "audio": observation["audio"],
                    "samples": int(raw_audio.shape[0]),
                }
                for sequence, raw_audio, observation in step_observations
            ],
        }
        self._append_jsonl(self.directory / "audio_batches.jsonl", record)
        return record

    def record_command(self, observation_record, action_id):
        record = dict(observation_record)
        record["action_id"] = int(action_id)
        self._append_jsonl(self.directory / "commands.jsonl", record)

    def _append_jsonl(self, path, record):
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8") as output_file:
                output_file.write(line)
                output_file.flush()

    @staticmethod
    def _write_atomic(path, contents):
        temporary_path = path.with_name(path.name + ".tmp")
        try:
            temporary_path.write_bytes(contents)
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


class StatefulInferenceHandler(BaseHTTPRequestHandler):
    server_version = "AVNavRealExperiment/1.0"

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ready", "policy": "AVNavInferencePolicy",
                                 "candidate": "corrective_epoch_015", "modalities": ["audio", "rgb"]})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/step":
            self.send_json(404, {"error": "not found"})
            return
        expected_token = self.server.auth_token
        if expected_token and self.headers.get("Authorization", "") != "Bearer " + expected_token:
            self.send_json(401, {"error": "unauthorized"})
            return
        try:
            body_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "invalid Content-Length"})
            return
        if body_length <= 0 or body_length > MAX_BODY_BYTES:
            self.send_json(413, {"error": "request body is empty or too large"})
            return
        try:
            body = self.rfile.read(body_length)
            if len(body) != body_length:
                raise ValueError("request body ended early")
            payload = json.loads(body.decode("utf-8"))
            session_id, sequence, request_id, audio, rgb, depth = decode_observation(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            self.send_json(400, {"error": str(exc)})
            return

        # The policy keeps previous RGB frames and a five-step LSTM history.
        # Audio batching is archival only; inference uses the current step's audio.
        # One lock preserves that ordering and this service is scoped to one robot.
        with self.server.policy_lock:
            if session_id == self.server.session_id and sequence == self.server.last_sequence:
                if request_id == self.server.last_request_id:
                    self.send_json(200, self.server.last_response)
                else:
                    self.send_json(409, {"error": "sequence already used by another request"})
                return
            if session_id != self.server.session_id:
                if sequence != 0:
                    self.send_json(409, {"error": "a new session must start at sequence 0"})
                    return
                self.server.policy.reset()
                self.server.audio_batch.clear()
                self.server.session_id = session_id
                self.server.last_sequence = -1
                self.server.last_request_id = None
            elif sequence != self.server.last_sequence + 1:
                self.send_json(409, {"error": "out-of-order sequence; expected {}".format(
                    self.server.last_sequence + 1)})
                return

            try:
                observation = self.server.frontend(audio, rgb, depth, SAMPLE_RATE)
                if not isinstance(observation, dict):
                    raise TypeError("frontend must return a dict with rgb and spectrogram")
                for field in ("rgb", "spectrogram"):
                    if field not in observation:
                        raise ValueError("frontend result is missing {}".format(field))
                action_id = int(self.server.policy.act(
                    observation["spectrogram"], observation["rgb"]))
                if action_id not in (0, 1, 2, 3):
                    raise ValueError("policy returned action outside 0..3")
                archived_observation = self.server.observation_archive.save_observation(
                    session_id, sequence, request_id, audio, rgb, action_id
                )
                self.server.audio_batch.append((sequence, audio, archived_observation))
                if len(self.server.audio_batch) == AUDIO_BATCH_STEPS:
                    self.server.observation_archive.save_audio_batch(
                        session_id, self.server.audio_batch
                    )
                    self.server.audio_batch.clear()
                self.server.observation_archive.record_command(archived_observation, action_id)
            except Exception as exc:
                # Inference may have advanced recurrent state. Require a new
                # session before accepting another observation after a failure.
                self.server.policy.reset()
                self.server.audio_batch.clear()
                self.server.session_id = None
                self.server.last_sequence = -1
                self.server.last_request_id = None
                self.server.log_message("inference failed: %s", exc)
                self.send_json(500, {"error": "inference failed"})
                return

            response = {
                "session_id": session_id,
                "sequence": sequence,
                "request_id": request_id,
                "action_id": action_id,
            }
            self.server.last_sequence = sequence
            self.server.last_request_id = request_id
            self.server.last_response = response
        self.send_json(200, response)

    def send_json(self, status, value):
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-checkpoint", default=str(DEFAULT_FOUNDATION))
    parser.add_argument("--policy-checkpoint", default=str(DEFAULT_POLICY))
    parser.add_argument("--device", default=None, help="cpu, cuda, or defaults to available CUDA")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--token", default=os.environ.get("REMOTE_INFERENCE_TOKEN"))
    parser.add_argument("--record-dir", default=str(DEFAULT_RECORD_DIR),
                        help="directory for received RGB/audio and policy action records")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        policy = AVNavInferencePolicy(args.foundation_checkpoint, args.policy_checkpoint,
                                      device=args.device, deterministic=True)
    except Exception as exc:
        parser.error("could not initialize policy: {}".format(exc))

    server = ThreadingHTTPServer((args.host, args.port), StatefulInferenceHandler)
    server.frontend = prepare_observation
    server.policy = policy
    server.observation_archive = ObservationArchive(args.record_dir)
    server.audio_batch = []
    server.policy_lock = threading.Lock()
    server.session_id = None
    server.last_sequence = -1
    server.last_request_id = None
    server.last_response = None
    server.auth_token = args.token
    server.daemon_threads = True
    print("AVNav inference API listening at http://{}:{}/v1/step".format(
        args.host, args.port), flush=True)
    print("Policy history is retained per session; action IDs are returned unchanged.", flush=True)
    print("Audio archives are concatenated in batches of {} steps; model input remains per-step.".format(
        AUDIO_BATCH_STEPS), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down inference service.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
