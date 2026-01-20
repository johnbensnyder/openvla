"""
libero_eval_utils.py

Utility functions for running LIBERO simulation rollouts during training.
Returns metrics and video frames for TensorBoard logging.
"""

import numpy as np
import torch
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import get_vla_action

# Max steps per task suite
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def run_libero_rollouts(
    model,
    processor,
    task_suite_name: str,
    unnorm_key: str,
    num_episodes: int = 10,
    num_videos: int = 5,
    center_crop: bool = True,
    num_steps_wait: int = 10,
    seed: int = 0,
) -> tuple:
    """
    Run LIBERO simulation rollouts and return success rate + video frames.
    
    Args:
        model: OpenVLA model
        processor: HuggingFace processor
        task_suite_name: One of libero_spatial, libero_object, libero_goal, libero_10
        unnorm_key: Key for action un-normalization
        num_episodes: Total episodes to run across all tasks
        num_videos: Number of rollout videos to capture
        center_crop: Whether to center crop images (use True if trained with augmentation)
        num_steps_wait: Steps to wait for objects to settle
        seed: Random seed
        
    Returns:
        (success_rate, videos_dict) where videos_dict maps task_name -> list of (frames, success) tuples
    """
    model.eval()
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    
    episodes_per_task = max(1, num_episodes // num_tasks)
    videos_per_task = max(1, num_videos // num_tasks)
    max_steps = MAX_STEPS.get(task_suite_name, 300)
    
    total_successes = 0
    total_episodes = 0
    videos_dict = {}
    
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, "openvla", resolution=256)
        
        task_videos = []
        
        for ep_idx in range(episodes_per_task):
            env.reset()
            obs = env.set_init_state(initial_states[ep_idx % len(initial_states)])
            
            frames = []
            capture_video = len(task_videos) < videos_per_task
            done = False
            
            for t in range(max_steps + num_steps_wait):
                # Wait for objects to settle
                if t < num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                    continue
                
                img = get_libero_image(obs, 224)
                if capture_video:
                    frames.append(get_libero_image(obs, 256))  # Higher res for video
                
                observation = {"full_image": img}
                
                with torch.no_grad():
                    action = get_vla_action(
                        model, processor, "openvla/openvla-7b",
                        observation, task_description, unnorm_key,
                        center_crop=center_crop
                    )
                
                # Normalize and invert gripper action
                action[-1] = 2 * action[-1] - 1  # [0,1] -> [-1,1]
                action[-1] = np.sign(action[-1])  # Binarize
                action[-1] = -action[-1]  # Invert for LIBERO
                
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    break
            
            if capture_video and frames:
                task_videos.append((np.stack(frames), done))
            
            if done:
                total_successes += 1
            total_episodes += 1
        
        env.close()
        if task_videos:
            videos_dict[task_description] = task_videos
    
    success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    return success_rate, videos_dict
