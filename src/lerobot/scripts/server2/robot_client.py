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
Example command:
```shell
python src/lerobot/scripts/server/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import queue
import threading
import time
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any, Callable, Optional

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    send_bytes_in_chunks,
    validate_robot_cameras_for_policy,
    visualize_action_queue_size,
)
from lerobot.transport import (
    async_inference_pb2,  # type: ignore
    async_inference_pb2_grpc,  # type: ignore
)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        if config.verify_robot_cameras:
            # Load policy config for validation
            policy_config = PreTrainedConfig.from_pretrained(config.pretrained_name_or_path)
            policy_image_features = policy_config.image_features

            # The cameras specified for inference must match the one supported by the policy chosen
            validate_robot_cameras_for_policy(lerobot_features, policy_image_features)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = async_inference_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self._running_event = threading.Event()
        self.action_queue = Queue()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

    @property
    def running(self):
        return self._running_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server."""
        try:
            # Client-server handshake
            self.stub.Ready(async_inference_pb2.Empty())
            self.logger.info("Connected to policy server.")

            # Send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)  # nosec
            policy_setup = async_inference_pb2.PolicySetup(data=policy_config_bytes)
            self.stub.SendPolicyInstructions(policy_setup)
            self.logger.info("Policy instructions sent to server.")

            self._running_event.set()

            # Start the threads
            self.control_thread = threading.Thread(target=self._control_loop)
            self.policy_thread = threading.Thread(target=self._policy_client_loop)

            self.control_thread.start()
            self.policy_thread.start()

            self.logger.info("Control and policy threads started.")
            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client."""
        self._running_event.clear()

        # Wait for threads to finish
        if hasattr(self, "control_thread") and self.control_thread.is_alive():
            self.control_thread.join()
        if hasattr(self, "policy_thread") and self.policy_thread.is_alive():
            self.policy_thread.join()

        self.robot.disconnect()
        self.logger.info("Robot disconnected.")

        self.channel.close()
        self.logger.info("Client stopped, channel closed.")

    def _run_inference_on_server(self, obs: TimedObservation) -> Optional[list[TimedAction]]:
        """Send an observation to the server and get an action chunk back."""
        try:
            # Serialize observation
            observation_bytes = pickle.dumps(obs)  # nosec

            def observation_iterator():
                yield async_inference_pb2.Observation(data=observation_bytes)

            actions_chunk = self.stub.RunInference(observation_iterator())

            if not actions_chunk.data:
                return None

            # Deserialize actions
            timed_actions = pickle.loads(actions_chunk.data)  # nosec
            return timed_actions

        except grpc.RpcError as e:
            self.logger.error(f"Error during inference RPC: {e}")
            return None

    def _control_loop(self):
        """Continuously fetch actions from the queue and execute them on the robot."""
        self.logger.info("Control loop started.")
        while self.running:
            control_loop_start = time.perf_counter()
            try:
                # Get an action from the queue (blocks until an item is available)
                timed_action = self.action_queue.get(timeout=1.0)  # Timeout to allow checking self.running

                action_dict = {key: timed_action.get_action()[i].item() for i, key in enumerate(self.robot.action_features)}
                self.robot.send_action(action_dict)
                self.logger.debug(f"Action #{timed_action.get_timestep()} executed.")
                time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

            except queue.Empty:
                self.logger.debug("Action queue is empty, waiting for actions")
                time.sleep(1)
                continue  # Continue to check self.running

    def _policy_client_loop(self):
        """Continuously get observations, send them to the server, and queue the actions."""
        self.logger.info("Policy client loop started.")
        timestep = 0
        while self.running:
            # Get observation from the robot
            raw_obs = self.robot.get_observation()
            obs = Observation(raw_obs)
            timed_obs = TimedObservation(
                observation=obs,
                timestamp=time.perf_counter(),
                timestep=timestep,
            )

            # Run inference on the server
            timed_actions = self._run_inference_on_server(timed_obs)

            if timed_actions:
                # Add actions to the queue
                for action in timed_actions:
                    self.action_queue.put(action)
                self.logger.info(f"Queued {len(timed_actions)} actions for timestep #{timestep}.")

            # Sleep to maintain target FPS
            time.sleep(1)
            timestep += 1

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                if self.inference_semaphore.acquire(blocking=False):
                    _captured_observation = self.control_loop_observation(task, verbose)
                else:
                    self.logger.debug("Inference already in progress, skipping observation")

            self.logger.info(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        try:
            # Keep the main thread alive while the client is running
            while client.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            client.logger.info("Keyboard interrupt received.")
        finally:
            client.stop()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
