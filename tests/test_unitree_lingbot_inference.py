from __future__ import annotations

import base64
import http.client
import json
import threading
import unittest

import cv2
import numpy as np

from real_world_inference.inference_service import LingBotInferenceService
from real_world_inference.policy_adapter import (
    LingBotMobileTCP23Adapter,
    PolicyInputs,
    lingbot_actions_to_mobile23,
    mobile_state26_to_lingbot,
)
from real_world_inference.pose_transforms import (
    rot6d_to_matrix,
    rot6d_to_xyzw,
    xyzw_to_matrix,
    xyzw_to_rot6d,
)


IDENTITY_ROT6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
IMAGE_KEYS = (
    "observation.images.cam0",
    "observation.images.cam1",
    "observation.images.cam2",
)


def make_state() -> np.ndarray:
    state = np.zeros(26, dtype=np.float32)
    state[0:3] = [1, 2, 3]
    state[3:9] = IDENTITY_ROT6D
    state[9] = 4
    state[10:13] = [5, 6, 7]
    state[13:19] = IDENTITY_ROT6D
    state[19] = 8
    state[20:23] = [101, 102, 103]
    state[23:26] = [9, 10, 11]
    return state


def make_policy_result(horizon: int = 10) -> dict[str, np.ndarray]:
    end = np.zeros((horizon, 14), dtype=np.float32)
    end[:, 0:3] = [1, 2, 3]
    end[:, 3:7] = [0, 0, 0, 1]
    end[:, 7:10] = [4, 5, 6]
    end[:, 10:14] = [0, 0, 0, 1]
    effector = np.tile([0.25, 0.75], (horizon, 1)).astype(np.float32)
    base = np.tile([0.1, -0.2, 0.3], (horizon, 1)).astype(np.float32)
    return {
        "action.end.position": end,
        "action.effector.position": effector,
        "action.base.position": base,
    }


def make_cfg() -> dict:
    return {
        "server": {"host": "127.0.0.1", "port": 0},
        "policy": {
            "inference_format": "mobile_tcp23",
            "action_dim": 23,
            "task_prompt": "test task",
            "target_chunk_size": 10,
            "image_inputs": {
                "head_fpv": "head_fpv",
                "left_hand": "left_hand",
                "right_hand": "right_hand",
            },
        },
    }


class FakePolicy:
    def __init__(self):
        self.observations: list[dict] = []

    def infer(self, observation: dict) -> dict[str, np.ndarray]:
        self.observations.append(observation)
        return make_policy_result()


class FakeAdapter:
    def __init__(self):
        self.inputs: list[PolicyInputs] = []
        self.reset_count = 0

    def reset_pose_history(self) -> None:
        self.reset_count += 1

    def infer_actions23(self, inputs: PolicyInputs) -> tuple[np.ndarray, float]:
        self.inputs.append(inputs)
        return lingbot_actions_to_mobile23(make_policy_result()), 12.5


class PoseTransformTests(unittest.TestCase):
    def test_random_rotation_round_trip(self):
        rng = np.random.default_rng(20260812)
        quaternions = rng.normal(size=(2000, 4))
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
        matrices = xyzw_to_matrix(quaternions)
        reconstructed = rot6d_to_matrix(xyzw_to_rot6d(quaternions))
        np.testing.assert_allclose(reconstructed, matrices, atol=2e-6)

    def test_mixed_axis_180_degree_round_trip(self):
        axis = np.array([1.0, -1.0, 0.5])
        axis /= np.linalg.norm(axis)
        quaternion = np.r_[axis, 0.0]
        reconstructed = xyzw_to_matrix(rot6d_to_xyzw(xyzw_to_rot6d(quaternion)))
        np.testing.assert_allclose(reconstructed, xyzw_to_matrix(quaternion), atol=2e-6)

    def test_degenerate_rot6d_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "degenerate"):
            rot6d_to_xyzw([1, 0, 0, 2, 0, 0])


class MappingTests(unittest.TestCase):
    def test_mobile_state_slots(self):
        mapped = mobile_state26_to_lingbot(make_state())
        np.testing.assert_allclose(
            mapped["observation.state.end.position"],
            [1, 2, 3, 0, 0, 0, 1, 5, 6, 7, 0, 0, 0, 1],
        )
        np.testing.assert_array_equal(mapped["observation.state.effector.position"], [4, 8])
        np.testing.assert_array_equal(mapped["observation.state.base.position"], [9, 10, 11])

    def test_lingbot_action_slots(self):
        actions = lingbot_actions_to_mobile23(make_policy_result(2))
        self.assertEqual(actions.shape, (2, 23))
        np.testing.assert_allclose(actions[0, :10], [1, 2, 3, 1, 0, 0, 0, 1, 0, 0.25])
        np.testing.assert_allclose(actions[0, 10:20], [4, 5, 6, 1, 0, 0, 0, 1, 0, 0.75])
        np.testing.assert_allclose(actions[0, 20:23], [0.1, -0.2, 0.3])

    def test_adapter_builds_exact_lingbot_observation(self):
        fake_policy = FakePolicy()
        adapter = LingBotMobileTCP23Adapter(make_cfg()["policy"], policy=fake_policy)
        images = {key: np.zeros((8, 9, 3), dtype=np.uint8) for key in IMAGE_KEYS}
        actions, _ = adapter.infer_actions23(
            PolicyInputs(make_state(), images, "test task")
        )
        self.assertEqual(actions.shape, (10, 23))
        observation = fake_policy.observations[-1]
        self.assertEqual(set(observation), {
            "observation.state.end.position",
            "observation.state.effector.position",
            "observation.state.base.position",
            *IMAGE_KEYS,
            "task",
        })

    def test_online_quaternion_sign_is_continuous(self):
        fake_policy = FakePolicy()
        adapter = LingBotMobileTCP23Adapter(make_cfg()["policy"], policy=fake_policy)
        images = {key: np.zeros((4, 4, 3), dtype=np.uint8) for key in IMAGE_KEYS}
        first = make_state()
        second = make_state()
        observations = []
        for state, angle in ((first, np.deg2rad(179.0)), (second, np.deg2rad(181.0))):
            quaternion = np.array([np.sin(angle / 2), 0, 0, np.cos(angle / 2)])
            state[3:9] = xyzw_to_rot6d(quaternion)
            observations.append(adapter.build_observation(PolicyInputs(state, images, "test task")))
        q1 = observations[0]["observation.state.end.position"][3:7]
        q2 = observations[1]["observation.state.end.position"][3:7]
        self.assertGreater(float(np.dot(q1, q2)), 0.99)


class HttpContractTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeAdapter()
        self.service = LingBotInferenceService(make_cfg(), adapter=self.adapter)
        self.server = self.service.build_http_server("127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method: str, path: str, body: bytes | None = None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = json.loads(response.read().decode())
        connection.close()
        return response.status, payload

    @staticmethod
    def jpeg(color_bgr: tuple[int, int, int]) -> bytes:
        image = np.full((12, 16, 3), color_bgr, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("test JPEG encoding failed")
        return encoded.tobytes()

    def test_health_and_handshake_match_mobile_tcp23_contract(self):
        status, health = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["wire_format"], "mobile_tcp23_pose20_base4")
        self.assertEqual(health["action_dim"], 20)
        self.assertEqual(health["model_action_dim"], 23)
        self.assertEqual(health["base_action_dim"], 4)
        self.assertEqual(health["image_inputs"], {
            "head_fpv": "head_fpv",
            "left_hand": "left_hand",
            "right_hand": "right_hand",
        })
        status, handshake = self.request("POST", "/handshake", b"{}", {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(handshake["type"], "handshake_ack")
        self.assertEqual(self.adapter.reset_count, 1)

    def test_json_infer_response_and_rgb_decode(self):
        image_bytes = self.jpeg((10, 20, 30))
        encoded = base64.b64encode(image_bytes).decode()
        body = json.dumps({
            "seq": 7,
            "mobile_state": make_state().tolist(),
            "images": {role: encoded for role in ("head_fpv", "left_hand", "right_hand")},
        }).encode()
        status, response = self.request("POST", "/infer", body, {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(response["seq"], 7)
        self.assertEqual(np.asarray(response["actions"]).shape, (10, 20))
        base = np.asarray(response["base_action"])
        self.assertEqual(base.shape, (10, 4))
        np.testing.assert_allclose(base[0], [0.1, 0, -0.2, 0.3])
        decoded = self.adapter.inputs[-1].images["observation.images.cam0"]
        self.assertGreater(float(decoded[..., 0].mean()), float(decoded[..., 2].mean()))

    def test_missing_image_returns_400(self):
        body = json.dumps({"mobile_state": make_state().tolist(), "images": {}}).encode()
        status, response = self.request("POST", "/infer", body, {"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertEqual(response["type"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
