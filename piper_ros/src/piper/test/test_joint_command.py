import unittest
from array import array
from types import SimpleNamespace

from piper.joint_command import ARM_JOINTS, parse_joint_command


def command(names, positions, velocity=(), effort=()):
    return SimpleNamespace(name=names, position=array('d', positions),
                           velocity=array('d', velocity), effort=array('d', effort))


class JointCommandTest(unittest.TestCase):
    def test_reordered_joints_and_gripper_aliases(self):
        for gripper in ('joint7', 'gripper'):
            with self.subTest(gripper=gripper):
                msg = command([gripper, *reversed(ARM_JOINTS)],
                              [0.03, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                              [15, 0, 0, 0, 0, 0, 0],
                              [1.5, 0, 0, 0, 0, 0, 0])
                arm, speed, grip = parse_joint_command(msg, 2)
                self.assertEqual(arm, [5732, 11465, 17197, 22930, 28662, 34395])
                self.assertEqual((speed, grip), (15, (60000, 1500)))

    def test_arm_only_and_mirrored_finger(self):
        for names in (ARM_JOINTS, (*ARM_JOINTS, 'joint8')):
            with self.subTest(names=names):
                msg = command(names, [0.1] * len(names))
                self.assertEqual(parse_joint_command(msg, 1)[1:], (30, None))

        msg = command(['joint8', *ARM_JOINTS, 'joint7'], [-0.03, *([0.1] * 6), 0.03])
        self.assertEqual(parse_joint_command(msg, 2)[1:], (30, (60000, 1000)))

    def test_invalid_frames(self):
        names = [*ARM_JOINTS, 'joint7']
        for invalid in (command(names[:-2], [0] * 5),
                        command(names[:-1] + ['joint1'], [0] * 7),
                        command(names + ['gripper'], [0] * 8),
                        command(names, [0] * 6),
                        command(names, [0] * 7, velocity=[0] * 4),
                        command(names, [0] * 7, effort=[0] * 4),
                        command(names, [float('nan')] + [0] * 6)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_joint_command(invalid, 1)

    def test_deployment_commands(self):
        # README and deployment replay messages populate all seven array entries.
        names = [*ARM_JOINTS, 'joint7']
        for velocity, effort, expected in [
            ([0] * 7, [0] * 7, (30, (10000, 500))),
            ([0] * 6 + [10], [0] * 6 + [0.5], (10, (10000, 500))),
            ([0] * 6 + [150], [0] * 6 + [5], (100, (10000, 3000))),
            ([2] + [0] * 6, [0] * 6 + [float('nan')], (1, (10000, 1000))),
            ([0.2] * 6, [0] * 6 + [1.5], (30, (10000, 1500))),
        ]:
            with self.subTest(velocity=velocity, effort=effort):
                msg = command(names, [0.01] * 7, velocity, effort)
                self.assertEqual(parse_joint_command(msg, 1)[1:], expected)
