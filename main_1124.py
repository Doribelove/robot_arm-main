import dataclasses
import enum
import logging
import pathlib
import time
import threading

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
# 使用本地文件，不依赖openpi_client
import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import tyro

# ROS2导入
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
import cv2
from live_observations import live_input_ages

logger = logging.getLogger(__name__)

# 全局ROS节点和数据
ros_node = None
latest_joint_state = None
latest_image = None
latest_wrist_image = None
# 记录数据时间戳（用于监控数据新鲜度）
latest_joint_state_time = None
latest_image_time = None
latest_wrist_image_time = None
cv_bridge = CvBridge()
ros_spin_thread = None
ros_spin_active = False


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    THU_VLNA = "thu_vlna"


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server (通过SSH隧道)
    host: str = "127.0.0.1"  # 连接本地，通过SSH隧道转发
    # Port to connect to the server. If None, the server will use the default port.
    port: int | None = 6006  # 本地转发端口
    # API key to use for the server.
    api_key: str | None = None
    # Number of steps to run the policy for.
    num_steps: int = 30
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: pathlib.Path | None = None
    # Path to save the actions to a pickle file. (e.g., actions_output.pkl)
    actions_file: pathlib.Path | None = pathlib.Path("actions_output.pkl")
    # Environment to run the policy in.
    env: EnvMode = EnvMode.THU_VLNA  # 使用THU_VLNA获取ROS数据
    # Whether to publish actions to /joint_states topic
    publish_actions: bool = True  # 是否发布action到ROS话题（改为False默认关闭）
    # Action execution frequency (Hz)
    data_freq: float = 5.0  # 数据频率，默认15Hz
    # Number of actions to execute per request
    actions_per_request: int = 15  # 每次请求后执行多少个action
    max_observation_age: float = 1.0  # 最大实时输入接收间隔（秒）


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }

    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True, width=20)

        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True, width=8)

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:>7.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def _plot_actions(actions: np.ndarray, save_path: pathlib.Path) -> None:
    """绘制action曲线图
    
    Args:
        actions: shape=(num_steps, num_joints), 每步的关节动作
        save_path: PKL文件保存路径，图片将保存为同名png文件
    """
    num_steps, num_joints = actions.shape
    
    # 创建图形
    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 2 * num_joints), sharex=True)
    if num_joints == 1:
        axes = [axes]
    
    # 关节名称
    joint_names = [f'Joint {i+1}' for i in range(num_joints)]
    if num_joints == 7:
        joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Gripper']
    
    # 时间轴（步数）
    time_steps = np.arange(num_steps)
    
    # 为每个关节绘制曲线
    for i, (ax, name) in enumerate(zip(axes, joint_names)):
        ax.plot(time_steps, actions[:, i], 'b-', linewidth=2, label=name)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 添加统计信息
        mean_val = actions[:, i].mean()
        std_val = actions[:, i].std()
        min_val = actions[:, i].min()
        max_val = actions[:, i].max()
        ax.text(0.02, 0.98, f'μ={mean_val:.3f}, σ={std_val:.3f}\nmin={min_val:.3f}, max={max_val:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)
    
    # 设置x轴标签
    axes[-1].set_xlabel('Step', fontsize=11)
    
    # 总标题
    fig.suptitle(f'Executed Action Trajectories (Total Steps: {num_steps})', fontsize=14, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plot_path = save_path.with_suffix('.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"✅ 已保存action曲线图到 {plot_path}")


def main(args: Args) -> None:
    global ros_spin_active
    if args.env == EnvMode.THU_VLNA and not 0 < args.max_observation_age < float("inf"):
        raise ValueError("max_observation_age must be finite and positive")
    try:
        _run_controller(args)
    finally:
        if ros_node is not None:
            ros_spin_active = False
            if ros_spin_thread is not None:
                ros_spin_thread.join(timeout=1.0)
            ros_node.destroy_node()
            rclpy.shutdown()
            logger.info("ROS节点已关闭")


def _run_controller(args: Args) -> None:
    # 如果使用THU_VLNA环境，初始化ROS节点
    if args.env == EnvMode.THU_VLNA:
        logger.info("初始化ROS节点获取真实数据...")
        _init_ros_node()
    
    obs_fn = {
        EnvMode.ALOHA: _random_observation_aloha,
        EnvMode.ALOHA_SIM: _random_observation_aloha,
        EnvMode.DROID: _random_observation_droid,
        EnvMode.LIBERO: _random_observation_libero,
        EnvMode.THU_VLNA: lambda: _observation_thu_vlna(args.max_observation_age),
    }[args.env]

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    # Send a few observations to make sure the model is loaded.
    for _ in range(2):
        policy.infer(obs_fn())

    timing_recorder = TimingRecorder()
    executed_actions_list = []  # 保存所有实际执行的动作
    action_published = False  # 标记是否已打印发布信息
    
    # 计算每个action的时间间隔（秒）
    action_interval = 1.0 / args.data_freq  # 15Hz -> 0.0667s
    logger.info(f"⚙️  执行参数: 频率={args.data_freq}Hz, 间隔={action_interval*1000:.1f}ms, 每{args.actions_per_request}个action后请求新序列")
    
    # Action缓冲区
    action_buffer = None  # 存储当前的action序列 (50, 7)
    action_index = 0  # 当前执行到第几个action
    total_executed = 0  # 总共执行的action数
    request_count = 0  # 请求次数
    actions_executed_since_request = 0  # 从上次请求开始已经执行了多少个action
    
    # 异步请求相关变量
    next_action_buffer = None  # 存储异步请求返回的新action序列
    request_thread = None  # 请求线程
    request_active = False  # 是否有请求正在进行
    request_lock = threading.Lock()  # 线程锁，保护共享变量
    
    def async_request_worker(request_id):
        """后台线程：异步请求新的action序列"""
        nonlocal next_action_buffer, request_active
        
        try:
            logger.info(f"🔄 [请求 {request_id}] 后台线程开始请求...")
            start = time.time()
            
            # 发起请求
            action_response = policy.infer(obs_fn())
            
            end = time.time()
            inference_time = end - start
            inference_time_ms = inference_time * 1000
            
            # 记录timing
            timing_recorder.record("client_infer_ms", inference_time_ms)
            for key, value in action_response.get("server_timing", {}).items():
                timing_recorder.record(f"server_{key}", value)
            for key, value in action_response.get("policy_timing", {}).items():
                timing_recorder.record(f"policy_{key}", value)
            
            # 提取action序列
            if 'actions' in action_response:
                actions = np.array(action_response['actions'])
                with request_lock:
                    next_action_buffer = {
                        'actions': actions,
                        'inference_time': inference_time,
                        'request_id': request_id
                    }
                logger.info(f"✅ [请求 {request_id}] 后台接收完成: shape={actions.shape}, 耗时={inference_time_ms:.1f}ms")
            else:
                logger.error(f"❌ [请求 {request_id}] 返回数据中没有'actions'字段")
        
        except Exception as e:
            logger.error(f"❌ [请求 {request_id}] 异步请求出错: {e}")
        
        finally:
            with request_lock:
                request_active = False
    
    # 首次同步请求（确保有初始数据）
    logger.info("🔄 [初始化] 同步请求首个action序列...")
    first_response = policy.infer(obs_fn())
    if 'actions' in first_response:
        action_buffer = np.array(first_response['actions'])
        action_index = 0
        logger.info(f"✅ [初始化] 收到首个序列: shape={action_buffer.shape}")
    else:
        logger.error("❌ 初始化失败：无法获取首个action序列")
        return
    
    with tqdm.tqdm(total=args.num_steps, desc="Executing actions") as pbar:
        while total_executed < args.num_steps:
            if args.env == EnvMode.THU_VLNA:
                _check_live_inputs(args.max_observation_age)
            # 执行一个action
            action_start = time.time()
            # 检查是否需要触发新的异步请求
            refill_due = (actions_executed_since_request >= args.actions_per_request
                          or action_index >= len(action_buffer))
            with request_lock:
                start_request = refill_due and not request_active and next_action_buffer is None
                if start_request:
                    request_active = True
            if start_request:
                request_count += 1
                
                # 启动后台请求线程
                request_thread = threading.Thread(
                    target=async_request_worker, 
                    args=(request_count,),
                    daemon=True
                )
                request_thread.start()
            
            # 检查是否有新的action序列到达
            with request_lock:
                if next_action_buffer is not None:
                    new_buffer = next_action_buffer
                    next_action_buffer = None  # 清空
                    
                    # 计算延迟补偿
                    inference_time = new_buffer['inference_time']
                    inference_time_ms = inference_time * 1000
                    actions_to_skip = int(inference_time / action_interval)
                    
                    # 切换到新buffer
                    action_buffer = new_buffer['actions']
                    action_index = min(actions_to_skip, len(action_buffer) - 1) if actions_to_skip > 0 else 0
                    actions_executed_since_request = 0
                    
                    logger.info(f"🔄 [切换Buffer] 请求{new_buffer['request_id']} | 延迟={inference_time_ms:.1f}ms, 跳过{actions_to_skip}个action, 从[{action_index}]开始")
                    timing_recorder.record("actions_skipped", actions_to_skip)
            
            # 检查buffer是否还有可用的action
            if action_index >= len(action_buffer):
                logger.warning(f"⚠️  Buffer已用完 (index={action_index}, len={len(action_buffer)})，等待新序列...")
                time.sleep(action_interval)
                continue
            

            
            current_action = action_buffer[action_index]
            action_index += 1
            total_executed += 1
            actions_executed_since_request += 1
            
            # 保存执行的action
            executed_actions_list.append(current_action)
            
            # 格式化输出（每5个打印一次，避免刷屏）
            if total_executed % 5 == 1 or total_executed <= 3:
                action_str = ', '.join([f"{val:+.6f}" for val in current_action])
                logger.info(f"  Step {total_executed:3d} | Buffer[{action_index-1}] | Action: [{action_str}]")
            
            # 发布action到/joint_states话题
            if args.publish_actions and ros_node is not None:
                if not action_published:
                    logger.info("开始发布action到 /joint_states 话题...")
                    action_published = True
                if args.env == EnvMode.THU_VLNA:
                    _check_live_inputs(args.max_observation_age)
                ros_node.publish_action(current_action)
            
            pbar.update(1)
            
            # 精确控制频率：减去action执行耗时
            action_elapsed = time.time() - action_start
            sleep_time = max(0, action_interval - action_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    logger.info(f"\n✅ 执行完成! 总共执行 {total_executed} 个action, 请求 {request_count} 次")
    timing_recorder.print_all_stats()

    if args.timing_file is not None:
        timing_recorder.write_parquet(args.timing_file)
    
    # 保存所有实际执行的action数据
    if len(executed_actions_list) > 0 and args.actions_file is not None:
        import pickle
        with open(args.actions_file, 'wb') as f:
            pickle.dump(executed_actions_list, f)
        logger.info(f"已保存 {len(executed_actions_list)} 个action到 {args.actions_file}")
        
        # 打印action统计信息并绘制曲线图
        action_array = np.array(executed_actions_list)
        logger.info(f"Action数组形状: {action_array.shape}")
        logger.info(f"Action范围: min={action_array.min():.3f}, max={action_array.max():.3f}, mean={action_array.mean():.3f}")
        
        # 绘制action曲线图
        _plot_actions(action_array, args.actions_file)
    

def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }

def _init_ros_node():
    """初始化ROS节点并订阅数据"""
    global ros_node
    
    class ROSDataCollector(Node):
        def __init__(self):
            super().__init__('openpi_data_collector')
            
            # 订阅关节状态
            self.joint_sub = self.create_subscription(
                JointState,
                '/joint_states_single',
                self.joint_callback,
                10
            )
            
            # 订阅主相机图像（MIIVII）
            self.image_sub = self.create_subscription(
                Image,
                '/miivii_gmsl/image0',
                self.image_callback,
                10
            )
            
            # 订阅腕部相机（RealSense）
            self.wrist_image_sub = self.create_subscription(
                Image,
                '/camera/camera/color/image_raw',
                self.wrist_image_callback,
                10
            )
            
            # 创建Publisher，发布控制指令到/joint_states
            self.joint_cmd_pub = self.create_publisher(
                JointState,
                '/joint_states',
                10
            )
            
            self.get_logger().info("ROS节点已初始化，等待数据...")
        
        def joint_callback(self, msg):
            global latest_joint_state, latest_joint_state_time
            # 只取前7个关节（如果有的话）
            latest_joint_state = np.array(msg.position[:7])
            latest_joint_state_time = time.monotonic()
        
        def image_callback(self, msg):
            global latest_image, latest_image_time
            # MIIVII主相机 -> observation/image
            cv_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            latest_image = cv2.resize(cv_image, (224, 224))
            latest_image_time = time.monotonic()
        
        def wrist_image_callback(self, msg):
            global latest_wrist_image, latest_wrist_image_time
            # RealSense腕部相机 -> observation/wrist_image
            cv_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            latest_wrist_image = cv2.resize(cv_image, (224, 224))
            latest_wrist_image_time = time.monotonic()
        
        def publish_action(self, action_positions):
            """发布控制指令到/joint_states话题"""
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'piper_single'
            msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
            msg.position = action_positions.tolist() if isinstance(action_positions, np.ndarray) else action_positions
            msg.velocity = [0.0] * 7  # 可以根据需要设置速度
            msg.effort = [0.0] * 7    # 可以根据需要设置力矩
            
            self.joint_cmd_pub.publish(msg)
    
    rclpy.init()
    ros_node = ROSDataCollector()
    
    # 启动后台线程持续接收ROS数据
    global ros_spin_thread, ros_spin_active
    ros_spin_active = True
    
    def ros_spin_worker():
        """后台线程持续spin ROS节点"""
        while ros_spin_active:
            rclpy.spin_once(ros_node, timeout_sec=0.01)
    
    ros_spin_thread = threading.Thread(target=ros_spin_worker, daemon=True)
    ros_spin_thread.start()
    
    # 等待首帧数据
    logger.info("等待首帧ROS数据...")
    time.sleep(0.5)  # 等500ms让数据到达
    
    logger.info("ROS节点初始化完成")
# def _random_observation_thu_vlna() -> dict:
#     return {
#         "observation/state": np.random.rand(7),
#         "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#         "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#         "prompt": "do something",
#     }

def _check_live_inputs(max_age):
    return live_input_ages(time.monotonic(), max_age, latest_joint_state_time,
                           latest_image_time, latest_wrist_image_time)


def _observation_thu_vlna(max_age: float) -> dict:
    """从ROS节点获取真实观测数据"""
    global latest_joint_state, latest_image, latest_wrist_image
    global latest_joint_state_time, latest_image_time, latest_wrist_image_time
    
    # 后台线程已经在持续更新数据，直接读取最新值即可
    # 如果还没有数据，等待数据就绪
    max_wait = 10.0  # 最多等待10秒
    wait_start = time.monotonic()
    while latest_joint_state_time is None or latest_image_time is None or latest_wrist_image_time is None:
        missing = []
        if latest_joint_state_time is None:
            missing.append("关节状态")
        if latest_image_time is None:
            missing.append("主相机(MIIVII)")
        if latest_wrist_image_time is None:
            missing.append("腕部相机(RealSense)")
        
        elapsed = time.monotonic() - wait_start
        if elapsed > max_wait:
            raise RuntimeError(f"等待ROS数据超时({max_wait}秒): {', '.join(missing)}")
        
        logger.warning(f"⏳ 等待ROS数据 ({elapsed:.1f}s): {', '.join(missing)}")
        time.sleep(0.1)
    
    joint_age, image_age, wrist_age = _check_live_inputs(max_age)
    
    # 每10步打印一次数据新鲜度（避免刷屏）
    if hasattr(_observation_thu_vlna, 'call_count'):
        _observation_thu_vlna.call_count += 1
    else:
        _observation_thu_vlna.call_count = 0
    
    if _observation_thu_vlna.call_count % 10 == 0:
        logger.info(f"📊 数据年龄 | 关节: {joint_age * 1000:.0f}ms, 主相机: {image_age * 1000:.0f}ms, 腕相机: {wrist_age * 1000:.0f}ms")
    
    return {
        "observation/state": latest_joint_state,
        "observation/image": latest_image,
        "observation/wrist_image": latest_wrist_image,
        "prompt": "Move the robotic arm, and actuate the end-effector to press the elevator's UP call button.",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
