import os
from habitat.utils.visualizations import maps
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
IMAGE_DIR = os.path.join("examples", "images")
MAP_THICKNESS_SCALAR: int = 128

def get_topdown_map(sim):
    top_down_map = maps.get_topdown_map_from_sim(
        sim, map_resolution=1024
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    return top_down_map

def draw_path(top_down_map , path_point):
    return maps.draw_path(top_down_map , path_point , color = maps.MAP_SHORTEST_PATH_COLOR)

def draw_points_on_map(topdown_map, points=None):
    img = topdown_map.copy()
    
    if points is not None and len(points) > 0:
        for pt in points:
            cv2.circle(img, tuple(np.array(pt, dtype=np.int32)), radius=5, color=(255, 0, 0), thickness=-1)  # 红色填充点

    return img
def draw_point(sim , position , top_down_map):
    t_x, t_y = maps.to_grid(
        position[2],
        position[0],
        (top_down_map.shape[0], top_down_map.shape[1]),
        sim=sim,
    )
    point_padding = 2 * int(
            np.ceil(512 / MAP_THICKNESS_SCALAR)
    )
    top_down_map[
        t_x - point_padding : t_x + point_padding + 1,
        t_y - point_padding : t_y + point_padding + 1,
    ] = (255, 0, 0)

def draw_sound(sim , agent_pos , sound_points , top_down_map ,point_type):
    for index , position in enumerate(sound_points):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            (top_down_map.shape[0], top_down_map.shape[1]),
            sim=sim,
        )
        point_padding = 2 * int(
                np.ceil(512 / MAP_THICKNESS_SCALAR)
        )
        if index == 100:
            top_down_map[
                t_x - point_padding : t_x + point_padding + 1,
                t_y - point_padding : t_y + point_padding + 1,
            ] = (255, 0, 0)
        elif index == len(sound_points) -1:
            top_down_map[
                t_x - point_padding : t_x + point_padding + 1,
                t_y - point_padding : t_y + point_padding + 1,
            ] = (0, 0, 255)
        else:
            top_down_map[
                t_x - point_padding : t_x + point_padding + 1,
                t_y - point_padding : t_y + point_padding + 1,
            ] = point_type
    t_x, t_y = maps.to_grid(
            agent_pos[2],
            agent_pos[0],
            (top_down_map.shape[0], top_down_map.shape[1]),
            sim=sim,
        )

    top_down_map[
        t_x - point_padding : t_x + point_padding + 1,
        t_y - point_padding : t_y + point_padding + 1,
    ] = (0, 255, 0)

def draw_agent(
    image: np.ndarray,
    agent_center_coord: Tuple[int, int],
    agent_rotation: float,
    agent_radius_px: int = 5,
    ):
    return maps.draw_agent(image ,agent_center_coord , agent_rotation , agent_radius_px)

def real_point_to_grid(sim , path_points , top_down_map):
    grid_point = []
    for point in path_points:
        grid_point.append(maps.to_grid(point[2] , point[0]  , (top_down_map.shape[0] , top_down_map.shape[1]) , sim  , sim.pathfinder))
    return grid_point

def draw_map(env, path_points , agent_pos , sound_points):
        import math
        from scipy.spatial.transform import Rotation as R
        sim = env._env._sim
        agent_rot = sim.get_agent_state().rotation
        top_down_map = get_topdown_map(sim)
        path_points = real_point_to_grid(sim , path_points , top_down_map)
        quat = [agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w]
        r = R.from_quat(quat)
        euler = r.as_euler('zyx', degrees=True)
        yaw = euler[0]   # z 轴旋转角度
        
        # 保证在 0~360 之间
        yaw = (yaw + 360) % 360
        draw_sound(sim , agent_pos , sound_points,top_down_map , maps.MAP_VIEW_POINT_INDICATOR)
        maps.draw_path(top_down_map, path_points)
        top_down_map = maps.draw_agent(
            top_down_map, path_points[-1], yaw, agent_radius_px=8
        )
        return top_down_map
def plot_top_down_map_waypoint(info, dataset='mp3d'):
    """Render an AV-WaN snapshot as an RGB top-down map.

    The start is a green circle, action-selected waypoints are numbered red
    circles, sound sources are blue squares, and optional collision locations
    are magenta crosses. A cyan line shows the recorded agent path, while a
    gold line connects the start and waypoints in selection order. Coordinates
    in ``top_down_map`` are (row, column) in its unrotated map. Repeated
    waypoints show only the latest action index (including STOP). This also
    works from a saved info snapshot after the environment is closed.
    """
    map_info = info["top_down_map"]
    if "rgb_map" in map_info:
        # Also accept a previously rendered Habitat map. This makes it
        # possible to add the same annotated markers to cached map snapshots.
        top_down_map = np.asarray(map_info["rgb_map"]).copy()
        if top_down_map.ndim != 3 or top_down_map.shape[-1] not in (3, 4):
            raise ValueError("top_down_map.rgb_map must be an RGB or RGBA image")
        top_down_map = top_down_map[..., :3]
    else:
        map_values = map_info["map"].copy()
        # Remove source/target markers already baked into Habitat's map as well.
        map_values[np.isin(map_values, [maps.MAP_SOURCE_POINT_INDICATOR,
                                       maps.MAP_TARGET_POINT_INDICATOR])] = maps.MAP_VALID_POINT
        top_down_map = maps.colorize_topdown_map(
            map_values, map_info.get("fog_of_war_mask")
        ).copy()
    if map_info.get("agent_map_coord") is not None:
        radius = max(1, top_down_map.shape[0] // (16 if dataset == 'replica' else 50))
        top_down_map = maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_info["agent_map_coord"],
            agent_rotation=map_info.get("agent_angle", 0.0),
            agent_radius_px=radius,
        )
    # Rotate the map and the annotations together, but keep text upright.
    original_width = top_down_map.shape[1]
    rotate = top_down_map.shape[0] > original_width
    if rotate:
        top_down_map = np.rot90(top_down_map)
    top_down_map = np.ascontiguousarray(top_down_map)
    height, width = top_down_map.shape[:2]

    def image_point(coord):
        row, col = (int(v) for v in coord)
        if rotate:
            row, col = original_width - 1 - col, row
        return col, row

    def label(point, text, color):
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x = max(0, min(point[0] + 7, width - text_width - 1))
        y = point[1] - 7
        y = max(text_height + 1, min(y, height - 3))
        cv2.putText(top_down_map, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(top_down_map, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)

    # Overwrite earlier indices when several actions select the same location.
    waypoint_labels = {}
    for index, coord in enumerate(map_info["waypoint_map_coords"], start=1):
        point = image_point(coord)
        waypoint_labels[point] = str(index)

    start_coord = map_info.get('start_map_coord')
    start_point = image_point(start_coord) if start_coord is not None else None
    path_coords = [image_point(coord)
                   for coord in map_info.get('agent_path_map_coords', [])]
    if len(path_coords) > 1:
        cv2.polylines(top_down_map,
                      [np.asarray(path_coords, dtype=np.int32).reshape(-1, 1, 2)],
                      False, (30, 150, 165), 2, cv2.LINE_AA)
    waypoint_points = list(waypoint_labels)
    sequence = ([start_point] if start_point is not None else []) + waypoint_points
    if len(sequence) > 1:
        cv2.polylines(top_down_map,
                      [np.asarray(sequence, dtype=np.int32).reshape(-1, 1, 2)],
                      False, (255, 190, 0), 2, cv2.LINE_AA)

    red = (255, 0, 0)
    for point in waypoint_labels:
        cv2.circle(top_down_map, point, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(top_down_map, point, 3, red, -1, cv2.LINE_AA)
    # Draw sound markers after waypoints so they remain visible on overlap.
    blue = (0, 0, 255)
    sound_points = []
    for index, coord in enumerate(map_info.get('sound_source_map_coords', []), start=1):
        x, y = image_point(coord)
        sound_points.append((x, y))
        cv2.rectangle(top_down_map, (x - 6, y - 6), (x + 6, y + 6),
                      (255, 255, 255), -1)
        cv2.rectangle(top_down_map, (x - 4, y - 4), (x + 4, y + 4), blue, -1)
        label((x, y), f'T{index}', blue)

    # Optional collision locations from a replay/diagnostic snapshot.
    for coord in map_info.get('collision_map_coords', []):
        x, y = image_point(coord)
        cv2.drawMarker(top_down_map, (x, y), (210, 35, 160),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=13,
                       thickness=2, line_type=cv2.LINE_AA)

    if start_point is not None:
        green = (0, 255, 0)
        cv2.circle(top_down_map, start_point, 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(top_down_map, start_point, 6, green, -1, cv2.LINE_AA)
        label(start_point, 'START', green)
    for point, index in waypoint_labels.items():
        label(point, index, red)
    return top_down_map
# def display_path_map(topdown_map, key_points=None, path=None, image_filename=None):
#     output_dir = './output_images'
#     os.makedirs(output_dir, exist_ok=True)

#     fig, ax = plt.subplots(figsize=(12, 8))
#     canvas = FigureCanvas(fig)
#     ax.axis("off")
#     ax.imshow(topdown_map)

#     if path is not None:
#         path_x = [point[0] for point in path]
#         path_y = [point[1] for point in path]
#         print(path_x)
#         ax.plot(path_x, path_y, marker="o", markersize=5, color="darkblue", alpha=0.7, label="Path", linewidth=5)
#         ax.plot(path_x[-1], path_y[-1], marker='o', markersize=5, color='blue', alpha=0.7, label="final_point")

#     if key_points is not None:
#         for point in key_points:
#             ax.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8, color="red", label="Key Point")

#     # 保存图片（可选）
#     if image_filename:
#         fig.savefig(image_filename, bbox_inches='tight', pad_inches=0)

#     # 将绘制的图像保存成 numpy 数组
#     canvas.draw()
#     img_np = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
#     img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # (H, W, 3)

#     plt.close(fig)  # 关闭图像释放内存

#     return img_np

# def display_path_map(topdown_map, env):

#     img = topdown_map
#     sim = env._env._sim
#     yaw = (yaw + 360) % 360
#     agent_rot = sim.get_agent_state().rotation
#     agent_position = sim.get_agent_state().position
#     top_down_map = maps.draw_agent(
#         top_down_map, agent_position, yaw, agent_radius_px=8
#     )
#     # 如果是灰度图，转为三通道方便画彩色
#     if len(img.shape) == 2 or img.shape[2] == 1:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    
#     # 绘制路径
#     if path is not None and len(path) > 1:
#         # 画线段
#         for i in range(len(path) - 1):
#             p1 = tuple(map(int, path[i]))
#             p2 = tuple(map(int, path[i + 1]))
#             cv2.line(img, p1, p2, (139, 0, 0), thickness=5)  # 深蓝线
#         # 起点终点标记
#         cv2.circle(img, tuple(map(int, path[0])), 6, (255, 255, 0), -1)  # 起点：黄
#         cv2.circle(img, tuple(map(int, path[-1])), 6, (255, 0, 0), -1)   # 终点：蓝

#     # 绘制关键点
#     if key_points is not None:
#         for pt in key_points:
#             cv2.circle(img, tuple(map(int, pt)), 8, (0, 0, 255), -1)  # 红色关键点

#     return img

def plot_top_down_map(info, dataset='mp3d', pred=None):
    top_down_map = info["top_down_map"]["map"]
    top_down_map = maps.colorize_topdown_map(
        top_down_map, info["top_down_map"]["fog_of_war_mask"]
    )
    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    if dataset == 'replica':
        agent_radius_px = top_down_map.shape[0] // 16
    else:
        agent_radius_px = top_down_map.shape[0] // 50
    top_down_map = maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=info["top_down_map"]["agent_angle"],
        agent_radius_px=agent_radius_px
    )
    # if pred is not None:
    #     from habitat.utils.geometry_utils import quaternion_rotate_vector

    #     source_rotation = info["top_down_map"]["agent_rotation"]

    #     rounded_pred = np.round(pred[1])
    #     direction_vector_agent = np.array([rounded_pred[1], 0, -rounded_pred[0]])
    #     direction_vector = quaternion_rotate_vector(source_rotation, direction_vector_agent)

    #     grid_size = (
    #         (maps.COORDINATE_MAX - maps.COORDINATE_MIN) / 10000,
    #         (maps.COORDINATE_MAX - maps.COORDINATE_MIN) / 10000,
    #     )
    #     delta_x = int(-direction_vector[0] / grid_size[0])
    #     delta_y = int(direction_vector[2] / grid_size[1])

    #     x = np.clip(map_agent_pos[0] + delta_x, a_min=0, a_max=top_down_map.shape[0])
    #     y = np.clip(map_agent_pos[1] + delta_y, a_min=0, a_max=top_down_map.shape[1])
    #     point_padding = 20
    #     for m in range(x - point_padding, x + point_padding + 1):
    #         for n in range(y - point_padding, y + point_padding + 1):
    #             if np.linalg.norm(np.array([m - x, n - y])) <= point_padding and \
    #                     0 <= m < top_down_map.shape[0] and 0 <= n < top_down_map.shape[1]:
    #                 top_down_map[m, n] = (0, 255, 255)
    #     if np.linalg.norm(rounded_pred) < 1:
    #         assert delta_x == 0 and delta_y == 0

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)
    return top_down_map


import numpy as np
import soundfile as sf

def save_audio_or_silence(audio_data, output_path, samplerate=16000, silence_threshold=1e-6):
    """
    如果音频能量过低，则保存为静音，否则保存归一化后的音频。
    audio_data: np.ndarray, shape (channels, samples) or (samples, channels)
    """
    audio_data = np.squeeze(audio_data)

    # 转成 (samples, channels)
    if len(audio_data.shape) > 1 and audio_data.shape[0] < audio_data.shape[1]:
        audio_data = audio_data.T  

    # 计算能量
    max_val = np.max(np.abs(audio_data))

    if max_val < silence_threshold:
        print("⚠️ 音频能量过低，保存为静音。")
        audio_int16 = np.zeros_like(audio_data, dtype=np.int16)
    else:
        print(f"音频最大值 {max_val:.2e}，进行归一化保存。")
        audio_normalized = audio_data / max_val
        audio_int16 = (audio_normalized * 32767).astype(np.int16)

    sf.write(output_path, audio_int16, samplerate, subtype="PCM_16")
    print(f"保存完成：{output_path}, shape={audio_int16.shape}")

def combine_all_audio_data(data, output_path="combined_audio.wav", samplerate=16000):
    """
    将所有 data['obs'] 的音频数据合并成一个文件
    """
    all_audio_chunks = []
    
    print(f"开始处理 {len(data['obs'])} 个音频片段...")
    
    for i, obs in enumerate(data['obs']):
        if 'spectrogram' in obs and len(obs['spectrogram']) > 1:
            audio_chunk = obs['spectrogram'][1]
            
            # 预处理音频数据
            audio_chunk = np.squeeze(audio_chunk)
            
            # 转成 (samples, channels)
            if len(audio_chunk.shape) > 1 and audio_chunk.shape[0] < audio_chunk.shape[1]:
                audio_chunk = audio_chunk.T
            
            # 检查是否为静音
            max_val = np.max(np.abs(audio_chunk))
            if max_val < 1e-6:
                print(f"片段 {i}: 静音")
                # 创建静音片段（保持相同长度）
                if len(audio_chunk.shape) == 1:
                    silent_chunk = np.zeros_like(audio_chunk)
                else:
                    silent_chunk = np.zeros_like(audio_chunk)
                all_audio_chunks.append(silent_chunk)
            else:
                print(f"片段 {i}: 有声音 (最大值: {max_val:.2e})")
                # 归一化并添加到列表
                audio_normalized = audio_chunk / max_val
                all_audio_chunks.append(audio_normalized)
        else:
            print(f"片段 {i}: 无有效音频数据")
            # 创建默认静音片段（假设长度与其他相同）
            if all_audio_chunks:
                silent_shape = all_audio_chunks[0].shape
                all_audio_chunks.append(np.zeros(silent_shape))
    
    if not all_audio_chunks:
        print("错误：没有找到有效的音频数据")
        return
    
    # 合并所有音频片段
    print("正在合并音频片段...")
    combined_audio = np.concatenate(all_audio_chunks, axis=0)
    
    # 最终归一化
    combined_max = np.max(np.abs(combined_audio))
    if combined_max > 0:
        combined_audio = combined_audio / combined_max
    
    # 转换为 int16 并保存
    audio_int16 = (combined_audio * 32767).astype(np.int16)
    
    sf.write(output_path, audio_int16, samplerate, subtype="PCM_16")
    print(f"✅ 所有音频已合并保存至：{output_path}")
    print(f"合并后音频形状：{audio_int16.shape}")
    print(f"总时长：{len(audio_int16) / samplerate:.2f} 秒")

# 使用方法
# combine_all_audio_data(data, "all_combined_audio.wav")

import numpy as np
import soundfile as sf

def save_audio_or_silence(audio_data, output_path, samplerate=16000, silence_threshold=1e-6):
    """
    如果音频能量过低，则保存为静音，否则保存归一化后的音频。
    audio_data: np.ndarray, shape (channels, samples) or (samples, channels)
    """
    audio_data = np.squeeze(audio_data)

    # 转成 (samples, channels)
    if audio_data.shape[0] < audio_data.shape[1]:
        audio_data = audio_data.T  

    # 计算能量
    max_val = np.max(np.abs(audio_data))

    if max_val < silence_threshold:
        print("⚠️ 音频能量过低，保存为静音。")
        audio_int16 = np.zeros_like(audio_data, dtype=np.int16)
    else:
        print(f"音频最大值 {max_val:.2e}，进行归一化保存。")
        audio_normalized = audio_data / max_val
        audio_int16 = (audio_normalized * 32767).astype(np.int16)

    sf.write(output_path, audio_int16, samplerate, subtype="PCM_16")
    print(f"保存完成：{output_path}, shape={audio_int16.shape}")

# 用法
# save_audio_or_silence(data['obs'][6]['spectrogram'][1], "output.wav")
from moviepy.audio.AudioClip import CompositeAudioClip, AudioArrayClip
import moviepy.editor as mpy
def images_to_video_with_audio(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    audios: List[str],
    sr: int,
    fps: int = 1,
    quality: Optional[float] = 5,
    **kwargs
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        audios: raw audio files
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"

    audio_clips = []
    multiplier = 0.5
    for i, audio in enumerate(audios):
        audio_clip = AudioArrayClip(audio.T[:int(sr * 1 / fps)] * multiplier, fps=sr)
        audio_clip = audio_clip.set_start(1 / fps * i)
        audio_clips.append(audio_clip)
    composite_audio_clip = CompositeAudioClip(audio_clips)
    video_clip = mpy.ImageSequenceClip(images, fps=fps)
    video_with_new_audio = video_clip.set_audio(composite_audio_clip)
    video_with_new_audio.write_videofile(os.path.join(output_dir, video_name))

def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    scene_name: str,
    sound: str,
    sr: int,
    episode_id: int,
    checkpoint_idx: int,
    metric_name: str,
    metric_value: float,
    fps: int = 10,
    audios: List[str] = None
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
        audios: raw audio files
    Returns:
        None
    """
    if len(images) < 1:
        return

    video_name = f"{scene_name}_{episode_id}_{sound}_{metric_name}{metric_value:.2f}"
    if "disk" in video_option:
        assert video_dir is not None
        if audios is None:
            images_to_video(images, video_dir, video_name)
        else:
            images_to_video_with_audio(images, video_dir, video_name, audios, sr, fps=fps)
