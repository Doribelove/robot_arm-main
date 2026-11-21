import importlib
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np


def load_controller(direction):
    # Load the CLI without starting ROS or requiring the camera and model stack.
    imports = (
        "scipy", "scipy.interpolate", "matplotlib", "matplotlib.pyplot",
        "websocket_client_policy", "polars", "rich", "tqdm", "tyro", "rclpy",
        "rclpy.node", "rclpy.qos", "sensor_msgs", "sensor_msgs.msg", "cv_bridge", "cv2",
    )
    with patch.dict(sys.modules, {name: MagicMock() for name in imports}):
        return importlib.import_module(f"main_dns_1128_offset_from_real_{direction}")


class ActionPublicationTest(unittest.TestCase):
    def test_finish_interrupt_and_inference_error(self):
        for direction in ("up", "down"):
            controller = load_controller(direction)
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
        if exit_mode == "interrupt":
            clock.sleep.side_effect = lambda _: controller._signal_handler(None, None)
        elif exit_mode == "inference_error":
            infer.side_effect = RuntimeError("inference unavailable")

        with patch.multiple(
            controller, _init_ros_node=Mock(), _observation_thu_vlna=Mock(),
            _interpolate_actions=lambda actions, **kwargs: actions,
            ros_node=node, rclpy=Mock(), JointState=Mock(), stop_requested=False,
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


if __name__ == "__main__":
    unittest.main()
