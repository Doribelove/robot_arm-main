def take_synchronized_frame(joints, color, miivii, depth=None):
    """Consume one aligned set of messages; return None if a stream is empty."""
    queues = [joints, color, miivii]
    if depth is not None:
        queues.append(depth)
    if not all(queues):
        return None

    def stamp_ns(message):
        stamp = message.header.stamp
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    # Match the slowest stream's newest sample. Every queue contains at
    # least one message at or after this timestamp.
    frame_time = min(stamp_ns(queue[-1]) for queue in queues)
    for queue in queues:
        while stamp_ns(queue[0]) < frame_time:
            queue.popleft()

    return (
        joints.popleft(),
        color.popleft(),
        miivii.popleft(),
        depth.popleft() if depth is not None else None,
    )
