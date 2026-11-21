import unittest
from collections import deque
from types import SimpleNamespace

from recording_frames import take_synchronized_frame


def message(nanoseconds):
    stamp = SimpleNamespace(sec=1_700_000_000, nanosec=nanoseconds)
    return SimpleNamespace(header=SimpleNamespace(stamp=stamp))


class RecordingFrameTests(unittest.TestCase):
    def test_paused_camera_does_not_repeat_frames(self):
        joints, color, miivii = (deque([message(10)]) for _ in range(3))
        self.assertIsNotNone(take_synchronized_frame(joints, color, miivii))
        self.assertIsNone(take_synchronized_frame(joints, color, miivii))

        fresh_joints, fresh_color, fresh_miivii = (message(20) for _ in range(3))
        joints.append(fresh_joints)
        color.append(fresh_color)
        self.assertIsNone(take_synchronized_frame(joints, color, miivii))
        miivii.append(fresh_miivii)
        self.assertEqual(
            take_synchronized_frame(joints, color, miivii),
            (fresh_joints, fresh_color, fresh_miivii, None),
        )
        self.assertIsNone(take_synchronized_frame(joints, color, miivii))

    def test_alignment_discards_older_samples_but_keeps_future_ones(self):
        joints = deque(message(t) for t in (10, 30, 70))
        color = deque(message(t) for t in (10, 20, 40))
        miivii = deque(message(t) for t in (10, 30))
        self.assertEqual(
            take_synchronized_frame(joints, color, miivii),
            (message(30), message(40), message(30), None),
        )
        self.assertEqual(list(joints), [message(70)])

    def test_depth_is_required_and_consumed_when_enabled(self):
        joints, color, miivii = (deque([message(10)]) for _ in range(3))
        depth = deque()
        self.assertIsNone(take_synchronized_frame(joints, color, miivii, depth))
        depth.append(message(10))
        self.assertEqual(
            take_synchronized_frame(joints, color, miivii, depth),
            (message(10), message(10), message(10), message(10)),
        )
        self.assertFalse(any((joints, color, miivii, depth)))
