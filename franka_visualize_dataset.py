#!/usr/bin/env python3
"""
Visualize Franka dataset episodes and per-dimension pose velocities.

Because the recorded actions are 6D poses rather than instantaneous velocities,
the script first computes a finite-difference 6D velocity vector over the first
six action dimensions and then derives the scalar action velocity magnitude via
an L2 norm, matching the paper-style definition v_t = ||a_t^{gt}||_2.

Supports either a single episode directory / robot_data.npz file or a dataset
directory containing multiple episode_* folders.

Examples:
    python franka_visualize_dataset.py ./collected_data_basic_demo/episode_0000
    python franka_visualize_dataset.py ./collected_data_basic_demo --episode episode_0003
    python franka_visualize_dataset.py ./collected_data_basic_demo --output summary.png --no_show
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform as st


POSE_LABELS = ['x', 'y', 'z', 'rx', 'ry', 'rz']
POSE_UNITS = ['m', 'm', 'm', 'rad', 'rad', 'rad']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize Franka dataset episodes and pose velocities')
    parser.add_argument('path', help='Episode path, robot_data.npz path, or dataset directory')
    parser.add_argument('--episode',
                        help='Episode name (e.g. episode_0003) or zero-based index when path is a dataset directory')
    parser.add_argument('--camera', default=None,
                        help='Camera directory to preview, e.g. fisheye or l515_0')
    parser.add_argument('--velocity_threshold', type=float, default=None,
                        help='Optional threshold for separating slow/fast actions; defaults to dataset median')
    parser.add_argument('--output', default=None,
                        help='Optional output figure path (.png recommended)')
    parser.add_argument('--no_show', action='store_true',
                        help='Save figure without opening an interactive window')
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'Path not found: {path}')
    return path


def list_episode_dirs(dataset_dir):
    return sorted(path for path in dataset_dir.glob('episode_*') if path.is_dir())


def resolve_dataset_and_episode(path, episode_arg=None):
    if path.is_file():
        if path.name != 'robot_data.npz':
            raise ValueError('Expected robot_data.npz when a file path is provided')
        episode_dir = path.parent
        dataset_dir = episode_dir.parent
        episode_dirs = list_episode_dirs(dataset_dir)
        return dataset_dir, episode_dir, episode_dirs

    if (path / 'robot_data.npz').exists():
        episode_dir = path
        dataset_dir = episode_dir.parent
        episode_dirs = list_episode_dirs(dataset_dir)
        return dataset_dir, episode_dir, episode_dirs

    episode_dirs = list_episode_dirs(path)
    if not episode_dirs:
        raise ValueError(f'No episode_* folders found in {path}')

    if episode_arg is None:
        episode_dir = episode_dirs[0]
    elif episode_arg.isdigit():
        index = int(episode_arg)
        if index < 0 or index >= len(episode_dirs):
            raise IndexError(f'Episode index out of range: {index}')
        episode_dir = episode_dirs[index]
    else:
        episode_dir = path / episode_arg
        if not episode_dir.is_dir():
            raise FileNotFoundError(f'Episode directory not found: {episode_dir}')

    return path, episode_dir, episode_dirs


def load_episode_data(episode_dir):
    npz_path = episode_dir / 'robot_data.npz'
    with np.load(npz_path) as data:
        timestamps = np.asarray(data['timestamps'], dtype=np.float64)
        actions = np.asarray(data['actions'], dtype=np.float64)
        robot_states = np.asarray(data['robot_states'], dtype=np.float64)
        joint_positions = np.asarray(data['joint_positions'], dtype=np.float64)

    if timestamps.ndim != 1:
        raise ValueError(f'timestamps must be 1D, got {timestamps.shape}')
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f'actions must have shape (N, 7), got {actions.shape}')
    if robot_states.ndim != 2 or robot_states.shape[1] != 7:
        raise ValueError(f'robot_states must have shape (N, 7), got {robot_states.shape}')
    if joint_positions.ndim != 2 or joint_positions.shape[1] != 7:
        raise ValueError(f'joint_positions must have shape (N, 7), got {joint_positions.shape}')
    if len(timestamps) != len(actions):
        raise ValueError('timestamps and actions must have the same length')
    if len(timestamps) == 0:
        raise ValueError(f'Empty episode: {episode_dir}')

    return {
        'timestamps': timestamps,
        'time_s': timestamps - timestamps[0],
        'actions': actions,
        'robot_states': robot_states,
        'joint_positions': joint_positions,
    }


def compute_pose_velocity(values, timestamps):
    velocities = np.zeros_like(values[:, :6])
    if len(values) < 2:
        return velocities

    delta_t = np.diff(timestamps)
    safe_delta_t = np.clip(delta_t, 1e-6, None)
    velocities[1:, :3] = np.diff(values[:, :3], axis=0) / safe_delta_t[:, None]

    prev_rot = st.Rotation.from_rotvec(values[:-1, 3:6])
    next_rot = st.Rotation.from_rotvec(values[1:, 3:6])
    delta_rot = next_rot * prev_rot.inv()
    velocities[1:, 3:6] = delta_rot.as_rotvec() / safe_delta_t[:, None]
    velocities[0] = velocities[1]
    return velocities


def compute_velocity_magnitude(velocity_vectors):
    return np.linalg.norm(velocity_vectors[:, :6], axis=1)


def compute_dataset_velocity_threshold(episode_dirs, manual_threshold=None):
    if manual_threshold is not None:
        return float(manual_threshold), 'manual'

    magnitudes = []
    for episode_dir in episode_dirs:
        episode_data = load_episode_data(episode_dir)
        action_vel = compute_pose_velocity(episode_data['actions'], episode_data['timestamps'])
        magnitudes.append(compute_velocity_magnitude(action_vel))

    if not magnitudes:
        raise ValueError('Cannot derive velocity threshold from an empty dataset')

    dataset_magnitude = np.concatenate(magnitudes)
    nonzero_magnitude = dataset_magnitude[dataset_magnitude > 1e-8]
    if len(nonzero_magnitude) == 0:
        return 0.0, 'dataset median (all zero)'
    return float(np.median(nonzero_magnitude)), 'dataset median (non-zero)'


def select_camera_dir(episode_dir, camera_name=None):
    camera_dirs = sorted(path for path in episode_dir.iterdir()
                         if path.is_dir() and path.name != '__pycache__')
    if not camera_dirs:
        return None
    if camera_name is None:
        return camera_dirs[0]
    for camera_dir in camera_dirs:
        if camera_dir.name == camera_name:
            return camera_dir
    raise FileNotFoundError(f'Camera directory not found: {camera_name}')


def build_preview_strip(camera_dir):
    if camera_dir is None:
        return None, 'No camera directories found'

    image_paths = sorted(camera_dir.glob('color_*.jpg'))
    if not image_paths:
        return None, f'No color images found in {camera_dir.name}'

    sample_indices = sorted({0, len(image_paths) // 2, len(image_paths) - 1})
    preview_images = []
    labels = []
    for index in sample_indices:
        image = cv2.imread(str(image_paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(image, f'frame {index}', (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        preview_images.append(image)
        labels.append(index)

    if not preview_images:
        return None, f'Failed to load preview images from {camera_dir.name}'

    strip = np.concatenate(preview_images, axis=1)
    return strip, f'{camera_dir.name}: frames {labels}'


def plot_dataset_overview(ax, episode_dirs, selected_episode, dataset_stats):
    ax.clear()
    episode_names = [path.name for path in episode_dirs]
    frame_counts = [dataset_stats[name]['num_frames'] for name in episode_names]
    durations = [dataset_stats[name]['duration_s'] for name in episode_names]
    selected_name = selected_episode.name
    x = np.arange(len(episode_names))
    colors = ['tab:orange' if name == selected_name else 'tab:blue' for name in episode_names]

    ax.bar(x, frame_counts, color=colors, alpha=0.85)
    ax.set_title('Dataset Overview')
    ax.set_ylabel('Frames')
    ax.set_xticks(x)
    ax.set_xticklabels(episode_names, rotation=45, ha='right', fontsize=8)

    twin = ax.twinx()
    twin.plot(x, durations, color='tab:red', marker='o', linewidth=2)
    twin.set_ylabel('Duration (s)')


def plot_summary_text(ax, dataset_dir, episode_dir, episode_data, preview_caption,
                      action_speed_mag, velocity_threshold, threshold_source):
    ax.clear()
    ax.axis('off')
    timestamps = episode_data['timestamps']
    actions = episode_data['actions']
    robot_states = episode_data['robot_states']
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    dt = np.diff(timestamps)
    mean_dt = float(np.mean(dt)) if len(dt) else 0.0
    mean_hz = 1.0 / mean_dt if mean_dt > 0 else 0.0
    slow_ratio = float(np.mean(action_speed_mag < velocity_threshold)) if len(action_speed_mag) else 0.0
    text = '\n'.join([
        f'Dataset: {dataset_dir.name}',
        f'Episode: {episode_dir.name}',
        f'Frames: {len(timestamps)}',
        f'Duration: {duration:.2f} s',
        f'Average sample rate: {mean_hz:.2f} Hz',
        f'Action speed threshold: {velocity_threshold:.4f} ({threshold_source})',
        f'Slow-action ratio: {slow_ratio:.2%}',
        f'Action speed mean/max: {action_speed_mag.mean():.4f} / {action_speed_mag.max():.4f}',
        f'Action start xyz: {np.array2string(actions[0, :3], precision=3)}',
        f'Action end xyz:   {np.array2string(actions[-1, :3], precision=3)}',
        f'State start xyz:  {np.array2string(robot_states[0, :3], precision=3)}',
        f'State end xyz:    {np.array2string(robot_states[-1, :3], precision=3)}',
        f'Preview: {preview_caption}',
    ])
    ax.text(0.01, 0.98, text, va='top', ha='left', family='monospace', fontsize=10)


def plot_camera_preview(ax, preview_strip, caption):
    ax.clear()
    ax.set_title('Camera Preview')
    if preview_strip is None:
        ax.axis('off')
        ax.text(0.5, 0.5, caption, ha='center', va='center')
        return
    ax.imshow(preview_strip)
    ax.set_xlabel(caption)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_pose_timeseries(ax, time_s, actions, robot_states, start_dim, end_dim, title):
    ax.clear()
    for dim in range(start_dim, end_dim):
        label = f'{POSE_LABELS[dim]} ({POSE_UNITS[dim]})'
        ax.plot(time_s, actions[:, dim], linewidth=1.8, label=f'action {label}')
        ax.plot(time_s, robot_states[:, dim], linestyle='--', linewidth=1.0,
                label=f'state {label}')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)


def plot_velocity_timeseries(ax, time_s, action_vel, state_vel, start_dim, end_dim, title):
    ax.clear()
    for dim in range(start_dim, end_dim):
        label = f'{POSE_LABELS[dim]} dot'
        ax.plot(time_s, action_vel[:, dim], linewidth=1.8, label=f'action {label}')
        ax.plot(time_s, state_vel[:, dim], linestyle='--', linewidth=1.0,
                label=f'state {label}')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)


def plot_velocity_magnitude(ax, time_s, action_speed_mag, state_speed_mag, velocity_threshold):
    ax.clear()
    ax.plot(time_s, action_speed_mag, linewidth=2.0, label='action speed magnitude')
    ax.plot(time_s, state_speed_mag, linestyle='--', linewidth=1.3, label='state speed magnitude')
    ax.axhline(velocity_threshold, color='tab:red', linestyle=':', linewidth=2,
               label=f'threshold = {velocity_threshold:.4f}')
    slow_mask = action_speed_mag < velocity_threshold
    ax.fill_between(time_s, 0, action_speed_mag, where=slow_mask,
                    color='tab:orange', alpha=0.2, label='critical / slow')
    ax.set_title('6D Action Velocity Magnitude')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('L2 speed')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_velocity_histogram(ax, action_speed_mag, velocity_threshold):
    ax.clear()
    ax.hist(action_speed_mag, bins=40, color='tab:blue', alpha=0.8)
    ax.axvline(velocity_threshold, color='tab:red', linestyle=':', linewidth=2,
               label=f'threshold = {velocity_threshold:.4f}')
    slow_ratio = float(np.mean(action_speed_mag < velocity_threshold)) if len(action_speed_mag) else 0.0
    ax.set_title(f'Action Speed Distribution (slow ratio {slow_ratio:.2%})')
    ax.set_xlabel('L2 speed')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_xy_trajectory(ax, actions, robot_states):
    ax.clear()
    ax.plot(actions[:, 0], actions[:, 1], linewidth=2.0, label='action xy')
    ax.plot(robot_states[:, 0], robot_states[:, 1], linestyle='--', linewidth=1.5,
            label='state xy')
    ax.scatter(actions[0, 0], actions[0, 1], c='tab:green', label='start', zorder=3)
    ax.scatter(actions[-1, 0], actions[-1, 1], c='tab:red', label='end', zorder=3)
    ax.set_title('XY Trajectory')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.axis('equal')


def plot_gripper(ax, time_s, actions, robot_states):
    ax.clear()
    ax.plot(time_s, actions[:, 6], linewidth=1.8, label='action gripper')
    ax.plot(time_s, robot_states[:, 6], linestyle='--', linewidth=1.2, label='state gripper')
    ax.set_title('Gripper Width')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('width (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def build_dataset_stats(episode_dirs):
    stats = {}
    for episode_dir in episode_dirs:
        npz_path = episode_dir / 'robot_data.npz'
        with np.load(npz_path) as data:
            timestamps = np.asarray(data['timestamps'], dtype=np.float64)
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        stats[episode_dir.name] = {
            'num_frames': int(len(timestamps)),
            'duration_s': float(duration),
        }
    return stats


def default_output_path(dataset_dir, episode_dir):
    return dataset_dir / f'{episode_dir.name}_visualization.png'


def main():
    args = parse_args()
    path = resolve_path(args.path)
    dataset_dir, episode_dir, episode_dirs = resolve_dataset_and_episode(path, args.episode)
    episode_data = load_episode_data(episode_dir)
    action_vel = compute_pose_velocity(episode_data['actions'], episode_data['timestamps'])
    state_vel = compute_pose_velocity(episode_data['robot_states'], episode_data['timestamps'])
    action_speed_mag = compute_velocity_magnitude(action_vel)
    state_speed_mag = compute_velocity_magnitude(state_vel)
    velocity_threshold, threshold_source = compute_dataset_velocity_threshold(
        episode_dirs, args.velocity_threshold)
    dataset_stats = build_dataset_stats(episode_dirs)
    camera_dir = select_camera_dir(episode_dir, args.camera)
    preview_strip, preview_caption = build_preview_strip(camera_dir)

    fig, axes = plt.subplots(5, 2, figsize=(18, 20), constrained_layout=True)
    time_s = episode_data['time_s']
    actions = episode_data['actions']
    robot_states = episode_data['robot_states']

    plot_dataset_overview(axes[0, 0], episode_dirs, episode_dir, dataset_stats)
    plot_camera_preview(axes[0, 1], preview_strip, preview_caption)
    plot_pose_timeseries(axes[1, 0], time_s, actions, robot_states, 0, 3,
                         'Position Dimensions (x, y, z)')
    plot_pose_timeseries(axes[1, 1], time_s, actions, robot_states, 3, 6,
                         'Rotation Dimensions (rx, ry, rz)')
    plot_velocity_timeseries(axes[2, 0], time_s, action_vel, state_vel, 0, 3,
                             'Linear Velocity by Dimension')
    plot_velocity_timeseries(axes[2, 1], time_s, action_vel, state_vel, 3, 6,
                             'Rotational Velocity by Dimension')
    plot_velocity_magnitude(axes[3, 0], time_s, action_speed_mag, state_speed_mag,
                            velocity_threshold)
    plot_velocity_histogram(axes[3, 1], action_speed_mag, velocity_threshold)
    plot_xy_trajectory(axes[4, 0], actions, robot_states)
    plot_gripper(axes[4, 1], time_s, actions, robot_states)
    fig.suptitle(f'Franka Dataset Visualization: {dataset_dir.name} / {episode_dir.name}', fontsize=16)

    summary_ax = fig.add_axes([0.02, 0.86, 0.18, 0.12])
    plot_summary_text(summary_ax, dataset_dir, episode_dir, episode_data, preview_caption,
                      action_speed_mag, velocity_threshold, threshold_source)

    output_path = Path(args.output).expanduser().resolve() if args.output else None
    if args.no_show and output_path is None:
        output_path = default_output_path(dataset_dir, episode_dir)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')
        print(f'Saved visualization to: {output_path}')

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()