import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from test_action_publication import load_controller


SENSOR_TIMES = (
    ("latest_joint_state_time", "关节状态"),
    ("latest_image_time", "主相机"),
    ("latest_wrist_image_time", "腕部相机"),
)


class LiveObservationTests(unittest.TestCase):
    def check_run(self, direction, stale_field, sensor, phase):
        controller = load_controller(direction)
        args = controller.Args(num_steps=3, output_dir=Mock())
        args.max_observation_age = 0.5
        node = Mock()
        policy = Mock()
        infer = policy.WebsocketClientPolicy.return_value.infer
        infer.return_value = {"actions": np.zeros((4, 7))}
        clock = Mock()
        clock.monotonic.return_value = clock.time.return_value = 100.0
        timestamps = dict.fromkeys((field for field, _ in SENSOR_TIMES), 100.0)

        def advance_clock(_):
            clock.monotonic.return_value += 0.6
            clock.time.return_value = clock.monotonic.return_value
            for field, _ in SENSOR_TIMES:
                if field != stale_field:
                    setattr(controller, field, clock.monotonic.return_value)

        if phase in ("moving", "healthy"):
            clock.sleep.side_effect = advance_clock
        elif phase == "startup":
            timestamps[stale_field] = 99.4

        with patch.multiple(
            controller, **timestamps, latest_joint_state=np.zeros(7),
            latest_image=np.zeros((1, 1, 3)), latest_wrist_image=np.zeros((1, 1, 3)),
            _init_ros_node=Mock(), _interpolate_actions=lambda actions, **kwargs: actions,
            ros_node=node, ros_spin_thread=Mock(), ros_spin_active=True, stop_requested=False,
            _websocket_client_policy=policy, signal=Mock(), time=clock, rclpy=Mock(),
            threading=MagicMock(), tqdm=MagicMock(), TimingRecorder=Mock(),
        ):
            if phase == "healthy":
                controller.main(args)
            else:
                with self.assertRaisesRegex(RuntimeError, sensor):
                    controller.main(args)
            node.destroy_node.assert_called_once()
            controller.rclpy.shutdown.assert_called_once()
            self.assertFalse(controller.ros_spin_active)
        self.assertEqual(infer.call_count, 0 if phase == "startup" else 3)
        self.assertEqual(node.publish_action.call_count, {"startup": 0, "moving": 1, "healthy": 3}[phase])
        node.joint_cmd_pub.publish.assert_not_called()

    def test_reject_expired_inputs_before_first_inference(self):
        for direction in ("up", "down"):
            for field, sensor in SENSOR_TIMES:
                with self.subTest(direction=direction, sensor=sensor):
                    self.check_run(direction, field, sensor, "startup")

    def test_stop_buffered_commands_when_one_input_stalls(self):
        for direction in ("up", "down"):
            for field, sensor in SENSOR_TIMES:
                with self.subTest(direction=direction, sensor=sensor):
                    self.check_run(direction, field, sensor, "moving")

    def test_fresh_feedback_allows_normal_run(self):
        for direction in ("up", "down"):
            with self.subTest(direction=direction):
                self.check_run(direction, None, None, "healthy")
