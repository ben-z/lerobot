# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:
```shell
python src/lerobot/scripts/server/policy_server.py \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue

import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class
from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
    receive_bytes_in_chunks,
)
from lerobot.transport import (
    async_inference_pb2,  # type: ignore
    async_inference_pb2_grpc,  # type: ignore
)


class PolicyServer(async_inference_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self._running_event = threading.Event()

        # FPS measurement


        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None

    @property
    def running(self):
        return self._running_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self._running_event.clear()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self._running_event.set()

        return async_inference_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return async_inference_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)
        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return async_inference_pb2.Empty()

    def RunInference(self, request_iterator, context):  # noqa: N802
        """Receive a stream of observations, run inference on the first, and return an action chunk."""
        client_id = context.peer()
        self.logger.debug(f"Running inference for client {client_id}")

        try:
            # The client sends a stream, but for this synchronous-style inference,
            # we only expect one observation at a time.
            first_observation_request = next(request_iterator)

            # Deserialize observation
            timed_observation = pickle.loads(first_observation_request.data)  # nosec

            obs_timestep = timed_observation.get_timestep()
            self.logger.info(f"Received observation #{obs_timestep}")

            # Predict action chunk
            action_chunk = self._predict_action_chunk(timed_observation)

            # Serialize action chunk
            actions_bytes = pickle.dumps(action_chunk)  # nosec

            self.logger.info(f"Sending action chunk for step #{obs_timestep}")
            return async_inference_pb2.Actions(data=actions_bytes)

        except Exception as e:
            self.logger.error(f"Error in RunInference: {e}")
            return async_inference_pb2.Actions(data=b'')

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _prepare_observation(self, observation_t: TimedObservation) -> Observation:
        """
        Prepare observation, ready for policy inference.
        E.g.: To keep observation sampling rate high (and network packet tiny) we send int8 [0,255] images from the
        client and then convert them to float32 [0,1] images here, before running inference.
        """
        # RawObservation from robot.get_observation() - wrong keys, wrong dtype, wrong image shape
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            self.device,
        )
        # processed Observation - right keys, right dtype, right image shape

        return observation

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation"""
        inference_starts = time.perf_counter()

        """1. Prepare observation"""
        start_time = time.perf_counter()
        observation = self._prepare_observation(observation_t)
        preprocessing_time = time.perf_counter() - start_time

        self.last_processed_obs: TimedObservation = observation_t

        """2. Get action chunk"""
        start_time = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_time

        """3. Post-inference processing"""
        start_time = time.perf_counter()
        # Move to CPU before serializing
        action_tensor = action_tensor.cpu().squeeze(0)

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocessing_time = time.perf_counter() - start_time
        inference_stops = time.perf_counter()

        self.logger.info(
            f"Observation {observation_t.get_timestep()} |"
            f"Inference time: {1000 * (inference_stops - inference_starts):.2f}ms"
        )

        # full-process latency breakdown for debugging purposes
        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Preprocessing time: {1000 * (preprocessing_time - inference_starts):.2f}ms | "
            f"Inference time: {1000 * (inference_time - preprocessing_time):.2f}ms | "
            f"Postprocessing time: {1000 * (postprocessing_time - inference_time):.2f}ms | "
            f"Total time: {1000 * (postprocessing_time - inference_starts):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    async_inference_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
