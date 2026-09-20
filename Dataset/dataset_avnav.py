
import os
import pickle
import random
from typing import Any
import numpy as np
import torch
from tqdm import tqdm

def load_two_frame_pt(cls, path, config):
    files = cls.get_files(path=path)
    random.shuffle(files)

    os.makedirs(config.TO_PATH, exist_ok=True)

    shard_size = 10000  # 每个shard包含的样本数
    shard_id = 0

    buffer_rgb, buffer_depth, buffer_audios, buffer_actions, buffer_angles, buffer_sound_ids = [], [], [], [], [], []
    sound_label_to_id = {}

    for file in tqdm(files, desc="Loading offline RL data"):
        with open(file, 'rb') as f:
            data = pickle.load(f)

        obs = data['obs']
        action_id = np.array(data['action_id']).reshape(-1).tolist()
        rewards = np.array(data['reward']).reshape(-1).tolist()
        dones = np.array(data['done']).reshape(-1).tolist() # 取出reset的时候的done。后续要删除这个地方。更改数据收集策略
        num_steps = max(0, len(obs) - 1)
        sound_values = cls._extract_sound_values_per_step(data, num_steps)
        sound_ids = cls._encode_sound_values(sound_values, sound_label_to_id)

        for i in range(len(obs) - 1): # 去除最后没有动作的终态
            v_now = obs[i]
            
            if i != 0:
                v_pre = obs[i - 1]
            a = action_id[i]
            r = rewards[i]
            d = dones[i]

            # 提取特征
            rgb_now = torch.from_numpy(v_now['rgb']).float() / 255.0
            depth_now = torch.from_numpy(v_now['depth']).float() 
            if i == 0:
                rgb_pre = torch.zeros_like(rgb_now)
                depth_pre = torch.zeros_like(depth_now)
            else:
                rgb_pre = torch.from_numpy(v_pre['rgb']).float() / 255.0
                depth_pre = torch.from_numpy(v_pre['depth']).float()
            audio = torch.from_numpy(v_now['spectrogram'][0]).float()

            # 去除上一步的audio，将两帧img拼接一起
            rgb = torch.cat([rgb_pre , rgb_now] , dim=2)
            depth = torch.cat([depth_pre , depth_now] , dim=2)

            action = torch.tensor(a, dtype=torch.long)
            angel = np.degrees(obs[i]['angle'][1])
            sound_id = torch.tensor(sound_ids[i], dtype=torch.long)
            buffer_rgb.append(rgb)
            buffer_depth.append(depth)
            buffer_audios.append(audio)
            buffer_actions.append(action)
            buffer_angles.append(torch.tensor(angel))
            buffer_sound_ids.append(sound_id)
            # 写一个 shard
            if len(buffer_rgb) >= shard_size:
                torch.save({
                    'rgb': torch.stack(buffer_rgb),
                    'depth':torch.stack(buffer_depth),
                    'audios': torch.stack(buffer_audios),
                    'actions': torch.stack(buffer_actions),
                    'angles':torch.stack(buffer_angles),
                    'sound_ids': torch.stack(buffer_sound_ids),
                }, f"{config.TO_PATH}/foundation_model_shard_{shard_id}.pt")

                print(f"保存 shard {shard_id}, size={len(buffer_rgb)}")

                buffer_rgb, buffer_depth, buffer_audios, buffer_actions, buffer_angles, buffer_sound_ids = [], [], [], [], [], []
                shard_id += 1

    # 保存最后一个不满 shard 的数据
    if buffer_rgb:
        torch.save({
            'rgb': torch.stack(buffer_rgb),
            'depth':torch.stack(buffer_depth),
            'audios': torch.stack(buffer_audios),
            'actions': torch.stack(buffer_actions),
            'angles':torch.stack(buffer_angles),
            'sound_ids': torch.stack(buffer_sound_ids),
        }, f"{config.TO_PATH}/foundation_model_shard_{shard_id}.pt")
        print(f"保存 shard {shard_id}, size={len(buffer_rgb)}")