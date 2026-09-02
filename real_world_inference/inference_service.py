from __future__ import annotations

"""Unitree-compatible HTTP transport for LingBot VLA V2 inference."""

import base64
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

try:
    from .policy_adapter import (
        VIRTUAL_MODEL_ACTION_DIM,
        WIRE_ARM_ACTION_DIM,
        LingBotMobileTCP23Adapter,
        PolicyInputs,
        validate_mobile_state26,
    )
except ImportError:  # Direct script execution.
    from policy_adapter import (
        VIRTUAL_MODEL_ACTION_DIM,
        WIRE_ARM_ACTION_DIM,
        LingBotMobileTCP23Adapter,
        PolicyInputs,
        validate_mobile_state26,
    )


NAME_RE = re.compile(r'(?:^|;)\s*name="([^"]+)"')


def _multipart_boundary(content_type: str) -> bytes:
    for item in str(content_type or "").split(";"):
        key, separator, value = item.strip().partition("=")
        if separator and key.lower() == "boundary":
            return value.strip().strip('"').encode("utf-8")
    raise ValueError("multipart boundary missing")


def _multipart_part_name(header_blob: bytes) -> Optional[str]:
    for line in header_blob.decode("latin1", errors="replace").split("\r\n"):
        if line.lower().startswith("content-disposition:"):
            match = NAME_RE.search(line)
            return match.group(1) if match else None
    return None


def parse_multipart_payload(raw: bytes, content_type: str) -> dict:
    boundary = _multipart_boundary(content_type)
    data: dict = {}
    image_parts: dict[str, bytes] = {}
    for part in raw.split(b"--" + boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].rstrip(b"\r\n")
        header_blob, separator, body = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        if body.endswith(b"\r\n"):
            body = body[:-2]
        name = _multipart_part_name(header_blob)
        if name == "meta":
            data = json.loads(body.decode("utf-8")) if body else {}
        elif name and name.startswith("image_"):
            image_parts[name[len("image_") :]] = body
    images = data.setdefault("images", {})
    if not isinstance(images, dict):
        raise ValueError("meta.images must be an object")
    images.update(image_parts)
    return data


class LingBotInferenceService:
    def __init__(self, cfg: dict, adapter=None):
        self.cfg = cfg
        self.server_cfg = cfg["server"]
        self.policy_cfg = cfg["policy"]
        self.adapter = adapter or LingBotMobileTCP23Adapter(self.policy_cfg)
        raw_mapping = self.policy_cfg.get("image_inputs", {})
        if not isinstance(raw_mapping, dict) or not raw_mapping:
            raise ValueError("policy.image_inputs must be a non-empty object")
        self.image_inputs = {str(key): str(value) for key, value in raw_mapping.items()}
        expected_wire_roles = {"head_fpv", "left_hand", "right_hand"}
        if set(self.image_inputs) != expected_wire_roles:
            raise ValueError(
                "policy.image_inputs must preserve the Unitree wire roles "
                f"{sorted(expected_wire_roles)}, got {sorted(self.image_inputs)}"
            )
        self.lingbot_image_inputs = {
            "observation.images.cam0": self.image_inputs["head_fpv"],
            "observation.images.cam1": self.image_inputs["left_hand"],
            "observation.images.cam2": self.image_inputs["right_hand"],
        }

    @staticmethod
    def wire_capabilities() -> dict:
        return {
            "wire_format": "mobile_tcp23_pose20_base4",
            "action_dim": WIRE_ARM_ACTION_DIM,
            "model_action_dim": VIRTUAL_MODEL_ACTION_DIM,
            "wire_arm_action_dim": WIRE_ARM_ACTION_DIM,
            "base_action_dim": 4,
        }

    @staticmethod
    def _decode_image(payload: object, role: str) -> np.ndarray:
        if isinstance(payload, list) and payload:
            payload = payload[-1]
        if isinstance(payload, (bytes, bytearray)):
            raw = bytes(payload)
        elif isinstance(payload, str):
            try:
                raw = base64.b64decode(payload)
            except Exception as error:
                raise ValueError(f"invalid base64 image: {role}") from error
        else:
            raise ValueError(f"image {role!r} must be JPEG bytes or base64 text")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"JPEG decode failed: {role}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _decode_images(self, raw_images: object) -> dict[str, np.ndarray]:
        if not isinstance(raw_images, dict):
            raise ValueError("images must be an object")
        decoded: dict[str, np.ndarray] = {}
        for lingbot_key, request_role in self.lingbot_image_inputs.items():
            if request_role not in raw_images or not raw_images[request_role]:
                raise KeyError(
                    f"missing images[{request_role!r}] for LingBot field {lingbot_key!r}"
                )
            decoded[lingbot_key] = self._decode_image(raw_images[request_role], request_role)
        return decoded

    def handle_observation(self, data: dict, upstream_timestamp_end: float) -> tuple[dict, float, float]:
        handle_start = time.perf_counter()
        if "mobile_state" not in data:
            raise KeyError("mobile_tcp23 inference requires mobile_state[26]")
        state = validate_mobile_state26(data["mobile_state"])
        images = self._decode_images(data.get("images", {}))
        prompt = str(data.get("prompt") or self.policy_cfg["task_prompt"])
        actions23, infer_ms = self.adapter.infer_actions23(
            PolicyInputs(mobile_state26=state, images=images, prompt=prompt)
        )
        actions20 = actions23[:, :20].astype(np.float32, copy=False)
        base_action4 = np.column_stack(
            (
                actions23[:, 20],
                np.zeros(len(actions23), dtype=np.float32),
                actions23[:, 21],
                actions23[:, 22],
            )
        ).astype(np.float32)
        downstream_timestamp_start = float(time.time())
        response = {
            "type": "action_sequence",
            "actions": actions20.tolist(),
            "base_action": base_action4.tolist(),
            "upstream_timestamp_end": float(upstream_timestamp_end),
            "downstream_timestamp_start": downstream_timestamp_start,
            "policy_inference_format": "mobile_tcp23",
            **self.wire_capabilities(),
        }
        if data.get("upstream_timestamp_start") is not None:
            response["upstream_timestamp_start"] = float(data["upstream_timestamp_start"])
        handle_ms = (time.perf_counter() - handle_start) * 1000.0
        return response, infer_ms, handle_ms

    def build_http_server(self, host: str, port: int) -> ThreadingHTTPServer:
        owner = self
        infer_lock = threading.Lock()
        session_id = str(uuid.uuid4())

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send_json(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_payload(self) -> dict:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as error:
                    raise ValueError("invalid Content-Length") from error
                raw = self.rfile.read(length) if length > 0 else b""
                content_type = str(self.headers.get("Content-Type", ""))
                if content_type.startswith("multipart/form-data"):
                    return parse_multipart_payload(raw, content_type)
                return json.loads(raw.decode("utf-8")) if raw else {}

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    self._send_json(
                        {
                            "ok": True,
                            "type": "health",
                            "backend": "lingbot-vla-v2",
                            "policy_inference_format": "mobile_tcp23",
                            "image_inputs": dict(owner.image_inputs),
                            **owner.wire_capabilities(),
                        }
                    )
                    return
                self._send_json({"ok": False, "type": "error", "message": "not found"}, 404)

            def do_POST(self) -> None:  # noqa: N802
                request_start = time.perf_counter()
                try:
                    data = self._read_payload()
                    decode_done = time.perf_counter()
                    if self.path == "/handshake":
                        reset_pose_history = getattr(owner.adapter, "reset_pose_history", None)
                        if callable(reset_pose_history):
                            reset_pose_history()
                        self._send_json(
                            {
                                "ok": True,
                                "type": "handshake_ack",
                                "session_id": session_id,
                                "server_time": float(time.time()),
                                "backend": "lingbot-vla-v2",
                                "policy_inference_format": "mobile_tcp23",
                                "image_inputs": dict(owner.image_inputs),
                                **owner.wire_capabilities(),
                            }
                        )
                        return
                    if self.path != "/infer":
                        self._send_json({"ok": False, "type": "error", "message": "not found"}, 404)
                        return
                    if not infer_lock.acquire(blocking=False):
                        self._send_json(
                            {"ok": False, "type": "busy", "message": "inference already running"},
                            429,
                        )
                        return
                    try:
                        response, infer_ms, handle_ms = owner.handle_observation(
                            data, upstream_timestamp_end=float(time.time())
                        )
                        total_sec = max(0.0, time.perf_counter() - request_start)
                        decode_sec = max(0.0, decode_done - request_start)
                        response["server_total_sec"] = total_sec
                        response["server_decode_sec"] = decode_sec
                        response["server_timing"] = {
                            "total_sec": total_sec,
                            "decode_sec": decode_sec,
                            "infer_sec": max(0.0, infer_ms * 1e-3),
                            "handle_sec": max(0.0, handle_ms * 1e-3),
                        }
                        if data.get("seq") is not None:
                            response["seq"] = int(data["seq"])
                        response["session_id"] = str(data.get("session_id") or session_id)
                        self._send_json(response)
                    finally:
                        infer_lock.release()
                except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
                    self._send_json({"ok": False, "type": "invalid_request", "message": str(error)}, 400)
                except Exception as error:
                    self._send_json({"ok": False, "type": "error", "message": str(error)}, 500)

            def log_message(self, _format: str, *args) -> None:
                return

        return ThreadingHTTPServer((host, int(port)), Handler)

    def serve_http(self, host: str, port: int) -> None:
        server = self.build_http_server(host, port)
        print(f"[LingBot HTTP] listening on http://{host}:{port} (/health, /handshake, /infer)")
        server.serve_forever()
