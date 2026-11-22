"""Check local receive times before using live feedback to drive the arm."""


def live_input_ages(now, max_age, joint_time, image_time, wrist_time):
    ages = []
    for name, received in (
        ("关节状态", joint_time),
        ("主相机", image_time),
        ("腕部相机", wrist_time),
    ):
        if received is None:
            raise RuntimeError(f"缺少实时输入: {name}")
        age = now - received
        if age > max_age:
            raise RuntimeError(f"{name}数据超时: {age:.2f} 秒（上限 {max_age:.2f} 秒），停止执行")
        ages.append(age)
    return ages
