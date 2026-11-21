"""Decode named ROS JointState commands for the PiPER CAN interface."""

import math


ARM_JOINTS = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')


def parse_joint_command(msg, gripper_multiplier):
    """Return six arm targets, the speed percentage, and an optional gripper command."""
    count = len(msg.name)
    indices = {name: i for i, name in enumerate(msg.name)}
    if len(indices) != count or not all(name in indices for name in ARM_JOINTS):
        raise ValueError("joint command requires six named arm joints without duplicates")
    if 'joint7' in indices and 'gripper' in indices:
        raise ValueError("joint7 and gripper name the same actuator")
    gripper_index = indices.get('joint7', indices.get('gripper'))
    # joint_states_single recordings have seven positions but only six motor velocities.
    arm_velocity_only = count == 7 and gripper_index is not None and len(msg.velocity) == 6
    if len(msg.position) != count or len(msg.effort) not in (0, count):
        raise ValueError("joint command arrays do not match the joint names")
    if len(msg.velocity) not in (0, count) and not arm_velocity_only:
        raise ValueError("unexpected number of joint velocities")
    if not all(math.isfinite(value) for value in msg.position):
        raise ValueError("joint positions must be finite")

    # Keep the same radian-to-millidegree conversion as existing command recordings.
    arm_targets = [round(msg.position[indices[name]] * 57324.840764) for name in ARM_JOINTS]
    speed = 30
    if gripper_index is None:
        return arm_targets, speed, None

    # Seven-joint command publishers use the gripper velocity field for arm speed.
    if len(msg.velocity) == 7 and any(msg.velocity):
        speed = max(1, min(100, round(msg.velocity[gripper_index])))

    opening = abs(round(msg.position[gripper_index] * 1_000_000)) * gripper_multiplier
    effort = 1000
    if msg.effort and not math.isnan(msg.effort[gripper_index]):
        effort = round(max(0.5, min(3.0, msg.effort[gripper_index])) * 1000)
    return arm_targets, speed, (opening, effort)
