import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


def load_driver():
    # Exercise the publisher thread without ROS, CAN or a physical arm.
    imports = {
        "rclpy": Mock(), "rclpy.node": SimpleNamespace(Node=object),
        "sensor_msgs.msg": Mock(), "std_msgs.msg": Mock(),
        "piper_sdk": SimpleNamespace(C_PiperInterface=object),
        "piper_msgs.msg": Mock(), "piper_msgs.srv": Mock(),
        "geometry_msgs.msg": Mock(), "scipy.spatial.transform": Mock(),
    }
    with patch.dict(sys.modules, imports):
        return importlib.import_module("piper.piper_ctrl_single_node")


class AutoEnableTest(unittest.TestCase):
    def run_publisher(self, auto_enable, gripper, feedback):
        driver = load_driver()
        arm = driver.PiperRosNode.__new__(driver.PiperRosNode)
        arm._PiperRosNode__enable_flag = False
        arm.auto_enable, arm.gripper_exist = auto_enable, gripper
        arm.piper = Mock()
        arm.get_logger = Mock()
        arm.create_rate = Mock()
        publishers = [
            "PublishArmState", "PublishArmJointAndGripper",
            "PublishArmCtrlAndGripper", "PublishArmEndPose",
        ]
        for name in publishers:
            setattr(arm, name, Mock())

        elapsed = 0

        def advance(delay):
            nonlocal elapsed
            elapsed += delay

        clock = Mock()
        clock.time.side_effect = clock.monotonic.side_effect = lambda: elapsed
        clock.sleep.side_effect = advance
        arm.piper.GetArmEnableStatus.side_effect = lambda: feedback(elapsed)
        arm.piper.GetArmLowSpdInfoMsgs.side_effect = lambda: SimpleNamespace(**{
            f"motor_{index}": SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=enabled))
            for index, enabled in enumerate(feedback(elapsed), 1)
        })
        with patch.object(driver, "time", clock):
            driver.rclpy.ok.side_effect = lambda: arm.create_rate.return_value.sleep.call_count < 2
            arm.publish_thread()

        for name in publishers:
            self.assertEqual(getattr(arm, name).call_count, 2, name)
        return arm, driver

    def test_timeout_leaves_feedback_running_and_arm_gated(self):
        ready = False
        arm, driver = self.run_publisher(True, False, lambda _: [True] * 5 + [ready])
        self.assertFalse(arm.GetEnableFlag())
        self.assertGreater(arm.piper.EnableArm.call_count, 0)
        ready = True
        with patch.object(driver.time, "sleep"):
            response = arm.handle_enable_service(SimpleNamespace(enable_request=True), SimpleNamespace())
        self.assertTrue(response.enable_response)
        self.assertTrue(arm.GetEnableFlag())
        arm.piper.GripperCtrl.assert_not_called()

    def test_delayed_acknowledgement_with_gripper(self):
        arm, _ = self.run_publisher(True, True, lambda seconds: [seconds >= 2] * 6)
        self.assertTrue(arm.GetEnableFlag())
        arm.piper.GripperCtrl.assert_called_with(0, 1000, 0x01, 0)

    def test_manual_mode_publishes_without_enabling(self):
        arm, _ = self.run_publisher(False, True, lambda _: [False] * 6)
        self.assertFalse(arm.GetEnableFlag())
        arm.piper.EnableArm.assert_not_called()
        arm.piper.GripperCtrl.assert_not_called()
