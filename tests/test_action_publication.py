import importlib
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np


def load_cli(module_name):
    # Load the CLI without starting ROS or requiring the camera and model stack.
    imports = (
        "scipy", "scipy.interpolate", "matplotlib", "matplotlib.pyplot",
        "websocket_client_policy", "polars", "rich", "tqdm", "tyro", "rclpy",
        "rclpy.node", "rclpy.qos", "sensor_msgs", "sensor_msgs.msg", "cv_bridge", "cv2",
    )
    with patch.dict(sys.modules, {name: MagicMock() for name in imports}):
        return importlib.import_module(module_name)


class ActionPublicationTest(unittest.TestCase):
    def test_finish_interrupt_and_inference_error(self):
        for direction in ("up", "down"):
            controller = load_cli(f"main_dns_1128_offset_from_real_{direction}")
            for publish in (False, True):
                for exit_mode in ("finish", "interrupt", "inference_error"):
                    with self.subTest(direction=direction, publish=publish, exit_mode=exit_mode):
                        self.check_run(controller, publish, exit_mode)

    def check_run(self, controller, publish, exit_mode):
        args = controller.Args(output_dir=Mock(), num_steps=2)
        if not publish:
            args.publish_actions = False

        node = Mock()
        policy_client = Mock()
        infer = policy_client.WebsocketClientPolicy.return_value.infer
        infer.return_value = {"actions": np.zeros((3, 7))}
        clock = Mock()
        clock.time.return_value = 0.0
        clock.monotonic.return_value = 0.0
        if exit_mode == "interrupt":
            clock.sleep.side_effect = lambda _: controller._signal_handler(None, None)
        elif exit_mode == "inference_error":
            infer.side_effect = RuntimeError("inference unavailable")

        with patch.multiple(
            controller, _init_ros_node=Mock(), _observation_thu_vlna=Mock(),
            _interpolate_actions=lambda actions, **kwargs: actions,
            ros_node=node, rclpy=Mock(), JointState=Mock(), stop_requested=False,
            latest_joint_state_time=0.0, latest_image_time=0.0, latest_wrist_image_time=0.0,
            _websocket_client_policy=policy_client, signal=Mock(), time=clock,
            threading=MagicMock(), tqdm=MagicMock(), TimingRecorder=Mock(),
        ):
            if exit_mode == "inference_error":
                with self.assertRaisesRegex(RuntimeError, "inference unavailable"):
                    controller.main(args)
            else:
                controller.main(args)

        steps = {"finish": 2, "interrupt": 1, "inference_error": 0}[exit_mode]
        self.assertEqual(node.publish_action.call_count, steps if publish else 0)
        if publish and exit_mode == "interrupt":
            node.joint_cmd_pub.publish.assert_called_once()
            (message,) = node.joint_cmd_pub.publish.call_args.args
            self.assertEqual(message.position, [0.0, 0.0, 0.0, 0.0, -0.3, 0.0, 0.0])
        else:
            node.joint_cmd_pub.publish.assert_not_called()


class ActionRefillTest(unittest.TestCase):
    def test_request_on_exhaustion_or_regular_prefetch(self):
        for name in ("main_1124", "main_dns_1128_offset_from_real_up",
                     "main_dns_1128_offset_from_real_down"):
            for short_horizon in (True, False):
                with self.subTest(name=name, short_horizon=short_horizon):
                    self.check_refill(load_cli(name), short_horizon)

    def check_refill(self, controller, short_horizon):
        legacy = controller.__name__ == "main_1124"
        # Two model steps yield six playback steps in the interpolating clients.
        steps = 2 if legacy else 6
        args = controller.Args(num_steps=steps + 1)
        if legacy:
            args.actions_file = None
        else:
            args.output_dir = Mock()
        if not short_horizon:
            args.actions_per_request = steps - 1

        node = Mock()
        client = Mock()
        infer = client.WebsocketClientPolicy.return_value.infer
        first = {"actions": np.zeros((2, 7))}
        next_chunk = {"actions": np.ones((2, 7))}
        infer.side_effect = [first, first, first, next_chunk]
        workers = []
        threads = MagicMock()

        def make_thread(target, args=(), **kwargs):
            thread = Mock()
            if args:  # The keyboard listener receives no request ID.
                thread.start.side_effect = lambda: workers.append(
                    (clock.sleep.call_count + 2, target, args))
            return thread

        def tick(_):
            if clock.sleep.call_count > steps + 4:
                self.fail(f"Playback stalled after {node.publish_action.call_count} commands")
            if workers and clock.sleep.call_count >= workers[0][0]:
                _, target, arguments = workers.pop(0)
                controller.latest_joint_state = np.ones(7)
                target(*arguments)

        threads.Thread.side_effect = make_thread
        clock = Mock()
        clock.monotonic.return_value = clock.time.return_value = 0
        clock.sleep.side_effect = tick
        patches = dict(_init_ros_node=Mock(), _observation_thu_vlna=Mock(),
                       latest_joint_state=np.zeros(7), latest_joint_state_time=0.0,
                       latest_image_time=0.0, latest_wrist_image_time=0.0,
                       _websocket_client_policy=client, ros_node=node, rclpy=Mock(),
                       time=clock, threading=threads, tqdm=MagicMock(), TimingRecorder=Mock())
        if not legacy:
            # A two-knot cubic spline is linear; run the production resampling loop.
            patches.update(CubicSpline=lambda x, y: lambda t: np.interp(t, x, y),
                           signal=Mock(), stop_requested=False)
        with patch.multiple(controller, **patches):
            controller.main(args)

        commands = [call.args[0][0] for call in node.publish_action.call_args_list]
        self.assertEqual(commands, [0.0] * steps + [1.0])
        self.assertEqual(infer.call_count, 4)
        self.assertFalse(workers)


if __name__ == "__main__":
    unittest.main()
