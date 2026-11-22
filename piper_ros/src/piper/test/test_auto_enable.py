import importlib
import itertools
import sys
import threading
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
    def run_publisher(self, auto_enable, gripper, feedback, during_retry=None):
        driver = load_driver()
        arm = driver.PiperRosNode.__new__(driver.PiperRosNode)
        arm._PiperRosNode__enable_flag = False
        arm._auto_enable_lock = threading.Lock()
        arm._auto_enable_cancelled = False
        arm.gripper_val_mutiple = 1
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
            if during_retry is not None:
                during_retry(arm, elapsed)

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

    def call_service(self, driver, arm, enable):
        clock = Mock()
        clock.monotonic.side_effect = clock.time.side_effect = itertools.count(step=0.5).__next__
        with patch.object(driver, "time", clock), patch.object(driver.rclpy, "ok", return_value=True):
            response = arm.handle_enable_service(SimpleNamespace(enable_request=enable), SimpleNamespace())
        return response.enable_response

    def assert_motion(self, arm, allowed):
        names = [f"joint{i}" for i in range(1, 7)]
        joint = SimpleNamespace(name=names, position=[0.0] * 6, velocity=[], effort=[])
        pose = SimpleNamespace(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, gripper=0, mode1=0, mode2=0)
        arm.piper.reset_mock()
        arm.joint_callback(joint)
        arm.pos_callback(pose)
        self.assertEqual(arm.piper.JointCtrl.call_count, int(allowed))
        self.assertEqual(arm.piper.EndPoseCtrl.call_count, int(allowed))
        self.assertEqual(arm.piper.MotionCtrl_2.call_count, 2 if allowed else 0)

    def test_timeout_leaves_feedback_running_and_arm_gated(self):
        ready = False
        arm, driver = self.run_publisher(True, False, lambda _: [True] * 5 + [ready])
        self.assertFalse(arm.GetEnableFlag())
        self.assertGreater(arm.piper.EnableArm.call_count, 0)
        ready = True
        self.assertTrue(self.call_service(driver, arm, True))
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

    def test_topic_enable_waits_for_all_motors(self):
        states = [True] * 5 + [False]
        arm, _ = self.run_publisher(False, False, lambda _: states)
        arm.enable_callback(SimpleNamespace(data=True))
        self.assert_motion(arm, False)
        states[-1] = True
        self.assert_motion(arm, True)
        states[0] = False
        self.assert_motion(arm, False)

    def test_failed_disable_keeps_joint_and_pose_commands_blocked(self):
        states = [True] * 6
        arm, driver = self.run_publisher(False, False, lambda _: states)
        self.assertTrue(self.call_service(driver, arm, True))
        self.assert_motion(arm, True)
        states[:5] = [False] * 5
        self.assertFalse(self.call_service(driver, arm, False))
        self.assert_motion(arm, False)
        states[:] = [True] * 6
        self.assert_motion(arm, False)
        states[:] = [False] * 6
        self.assertTrue(self.call_service(driver, arm, False))

    def test_manual_disable_cancels_pending_startup_retries(self):
        states = [False] * 6

        def disable(arm, elapsed):
            if elapsed == 1:
                arm.enable_callback(SimpleNamespace(data=False))
                states[:] = [True] * 6  # Delayed acknowledgement of the first enable.

        arm, _ = self.run_publisher(True, False, lambda _: states, during_retry=disable)
        arm.piper.EnableArm.assert_called_once_with(7)
        arm.piper.DisableArm.assert_called_once_with(7)
        self.assert_motion(arm, False)
