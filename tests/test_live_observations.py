import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from test_action_publication import load_cli


SENSOR_TIMES = (
    ("latest_joint_state_time", "关节状态"),
    ("latest_image_time", "主相机"),
    ("latest_wrist_image_time", "腕部相机"),
)
CONTROLLERS = ("main_dns_1128_offset_from_real_up", "main_dns_1128_offset_from_real_down",
               "main_1124")


class LiveObservationTests(unittest.TestCase):
    def check_run(self, module_name, stale_field, sensor, phase):
        controller = load_cli(module_name)
        args = controller.Args(num_steps=3)
        args.max_observation_age = 0.5
        if module_name == "main_1124":
            args.actions_file = None
            cli_mocks = {}
        else:
            args.output_dir = Mock()
            cli_mocks = dict(_interpolate_actions=lambda actions, **kwargs: actions,
                             signal=Mock(), stop_requested=False)
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
            _init_ros_node=Mock(), **cli_mocks,
            ros_node=node, ros_spin_thread=Mock(), ros_spin_active=True,
            _websocket_client_policy=policy, time=clock, rclpy=Mock(),
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
        for module_name in CONTROLLERS:
            for field, sensor in SENSOR_TIMES:
                with self.subTest(controller=module_name, sensor=sensor):
                    self.check_run(module_name, field, sensor, "startup")

    def test_stop_buffered_commands_when_one_input_stalls(self):
        for module_name in CONTROLLERS:
            for field, sensor in SENSOR_TIMES:
                with self.subTest(controller=module_name, sensor=sensor):
                    self.check_run(module_name, field, sensor, "moving")

    def test_fresh_feedback_allows_normal_run(self):
        for module_name in CONTROLLERS:
            with self.subTest(controller=module_name):
                self.check_run(module_name, None, None, "healthy")
