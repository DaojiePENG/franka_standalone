#!/usr/bin/env python3
"""
Clean Franka dataset episodes by removing waiting frames while keeping any frame
with arm motion or gripper action.

For each episode, the script:
1. Computes 6D pose velocity over the first six action dimensions.
2. Keeps frames with non-zero pose motion.
3. Also keeps frames around any gripper-width change event so grasp/place phases
   are not removed when the arm is stationary.
4. Writes a cleaned episode with filtered robot_data.npz and synchronized images.
5. Generates a clean 3-view video for the cleaned episode.
6. Generates one standalone video per selected camera view.
7. Generates an annotated 3-view inspection video.

Examples:
    python franka_clean_dataset.py ./collected_data_basic_demo ./collected_data_basic_demo_cleaned
    python franka_clean_dataset.py ./collected_data_basic_demo ./collected_data_basic_demo_cleaned --episode episode_0000
"""
import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import scipy.spatial.transform as st


DEFAULT_VIEWS = ['fisheye', 'l515_0', 'l515_1']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Clean Franka dataset by removing waiting frames and exporting reference videos')
    parser.add_argument('input_path', help='Input dataset directory, episode directory, or robot_data.npz path')
    parser.add_argument('output_path', help='Output directory for the cleaned dataset')
    parser.add_argument('--episode',
                        help='Episode name (e.g. episode_0003) or zero-based index when cleaning from a dataset directory')
    parser.add_argument('--pose_velocity_threshold', type=float, default=1e-8,
                        help='Frames with 6D pose speed above this threshold are kept')
    parser.add_argument('--gripper_delta_threshold', type=float, default=1e-8,
                        help='Frames with gripper width change above this threshold are kept')
    parser.add_argument('--gripper_context_frames', type=int, default=5,
                        help='Also keep this many frames before/after a gripper action event')
    parser.add_argument('--views', nargs='+', default=DEFAULT_VIEWS,
                        help='Camera views to compose into the reference video')
    parser.add_argument('--video_name', default='three_view.mp4',
                        help='Clean 3-view video filename saved inside each cleaned episode')
    parser.add_argument('--annotated_video_name', default='three_view_annotated.mp4',
                        help='Annotated 3-view inspection video filename saved inside each cleaned episode')
    parser.add_argument('--per_view_video_suffix', default='_only.mp4',
                        help='Suffix used for per-view videos, e.g. fisheye_only.mp4')
    parser.add_argument('--video_height', type=int, default=360,
                        help='Target height of each view tile in the reference video')
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'Path not found: {path}')
    return path


def list_episode_dirs(dataset_dir):
    return sorted(path for path in dataset_dir.glob('episode_*') if path.is_dir())


def resolve_input(path, episode_arg=None):
    if path.is_file():
        if path.name != 'robot_data.npz':
            raise ValueError('Expected robot_data.npz when a file path is provided')
        episode_dir = path.parent
        dataset_dir = episode_dir.parent
        return dataset_dir, [episode_dir]

    if (path / 'robot_data.npz').exists():
        episode_dir = path
        dataset_dir = episode_dir.parent
        return dataset_dir, [episode_dir]

    episode_dirs = list_episode_dirs(path)
    if not episode_dirs:
        raise ValueError(f'No episode_* folders found in {path}')

    if episode_arg is None:
        return path, episode_dirs
    if episode_arg.isdigit():
        index = int(episode_arg)
        if index < 0 or index >= len(episode_dirs):
            raise IndexError(f'Episode index out of range: {index}')
        return path, [episode_dirs[index]]

    episode_dir = path / episode_arg
    if not episode_dir.is_dir():
        raise FileNotFoundError(f'Episode directory not found: {episode_dir}')
    return path, [episode_dir]


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
    if len(actions) == 0:
        raise ValueError(f'Empty episode: {episode_dir}')

    return {
        'timestamps': timestamps,
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


def expand_event_mask(event_mask, context_frames):
    if context_frames <= 0 or not np.any(event_mask):
        return event_mask.copy()
    expanded = event_mask.copy()
    event_indices = np.flatnonzero(event_mask)
    for index in event_indices:
        start = max(0, index - context_frames)
        end = min(len(event_mask), index + context_frames + 1)
        expanded[start:end] = True
    return expanded


def build_keep_mask(actions, timestamps, pose_velocity_threshold,
                    gripper_delta_threshold, gripper_context_frames):
    pose_velocity = compute_pose_velocity(actions, timestamps)
    pose_speed = np.linalg.norm(pose_velocity, axis=1)

    gripper_delta = np.zeros(len(actions), dtype=np.float64)
    if len(actions) > 1:
        gripper_delta[1:] = np.abs(np.diff(actions[:, 6]))
        gripper_delta[0] = gripper_delta[1]

    pose_motion_mask = pose_speed > pose_velocity_threshold
    gripper_event_mask = gripper_delta > gripper_delta_threshold
    gripper_keep_mask = expand_event_mask(gripper_event_mask, gripper_context_frames)

    keep_mask = pose_motion_mask | gripper_keep_mask
    if not np.any(keep_mask):
        keep_mask[0] = True

    return {
        'keep_mask': keep_mask,
        'pose_speed': pose_speed,
        'gripper_delta': gripper_delta,
        'pose_motion_mask': pose_motion_mask,
        'gripper_event_mask': gripper_event_mask,
    }


def filtered_episode_data(episode_data, keep_mask):
    kept_indices = np.flatnonzero(keep_mask)
    original_timestamps = episode_data['timestamps'][keep_mask]
    if len(episode_data['timestamps']) > 1:
        nominal_dt = float(np.median(np.diff(episode_data['timestamps'])))
    else:
        nominal_dt = 0.1
    nominal_dt = max(nominal_dt, 1e-6)
    cleaned_timestamps = original_timestamps[0] + np.arange(len(original_timestamps)) * nominal_dt

    filtered = {
        'timestamps': cleaned_timestamps,
        'source_timestamps': original_timestamps,
        'actions': episode_data['actions'][keep_mask],
        'robot_states': episode_data['robot_states'][keep_mask],
        'joint_positions': episode_data['joint_positions'][keep_mask],
        'original_indices': kept_indices,
        'nominal_dt': nominal_dt,
    }
    return filtered


def list_camera_dirs(episode_dir):
    return sorted(path for path in episode_dir.iterdir()
                  if path.is_dir() and path.name != '__pycache__')


def copy_filtered_camera_frames(src_cam_dir, dst_cam_dir, kept_indices):
    image_paths = sorted(src_cam_dir.glob('color_*.jpg'))
    if image_paths and len(image_paths) < int(kept_indices.max()) + 1:
        raise ValueError(
            f'Camera {src_cam_dir.name} has only {len(image_paths)} images, '
            f'but frame index {int(kept_indices.max())} is required')

    dst_cam_dir.mkdir(parents=True, exist_ok=True)
    for new_index, old_index in enumerate(kept_indices):
        if image_paths:
            src_image = image_paths[int(old_index)]
            dst_image = dst_cam_dir / f'color_{new_index:05d}.jpg'
            shutil.copy2(src_image, dst_image)

    depth_path = src_cam_dir / 'depth.npz'
    if depth_path.exists():
        with np.load(depth_path) as depth_data:
            depth = np.asarray(depth_data['depth'])
        filtered_depth = depth[kept_indices]
        np.savez_compressed(dst_cam_dir / 'depth.npz', depth=filtered_depth)


def save_cleaned_episode(dst_episode_dir, filtered_data):
    dst_episode_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst_episode_dir / 'robot_data.npz',
        timestamps=filtered_data['timestamps'],
        source_timestamps=filtered_data['source_timestamps'],
        actions=filtered_data['actions'],
        robot_states=filtered_data['robot_states'],
        joint_positions=filtered_data['joint_positions'],
        original_indices=filtered_data['original_indices'],
        nominal_dt=np.array(filtered_data['nominal_dt'], dtype=np.float64),
    )


def resize_to_height(image, target_height):
    if image is None:
        return np.zeros((target_height, target_height, 3), dtype=np.uint8)
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return np.zeros((target_height, target_height, 3), dtype=np.uint8)
    scale = target_height / height
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def pad_to_width(image, target_width):
    if image.shape[1] == target_width:
        return image
    pad_total = target_width - image.shape[1]
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right,
                              borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))


def draw_label(image, text):
    cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2, lineType=cv2.LINE_AA)


def load_or_placeholder(image_path, target_height, label):
    if image_path is None or not image_path.exists():
        placeholder = np.zeros((target_height, target_height, 3), dtype=np.uint8)
        draw_label(placeholder, f'{label}: missing')
        return placeholder

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        placeholder = np.zeros((target_height, target_height, 3), dtype=np.uint8)
        draw_label(placeholder, f'{label}: unreadable')
        return placeholder

    image = resize_to_height(image, target_height)
    draw_label(image, label)
    return image


def load_raw_frame(image_path):
    if image_path is None or not image_path.exists():
        return None
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def load_resized_frame_or_placeholder(image_path, target_height, label):
    frame = load_raw_frame(image_path)
    if frame is None:
        placeholder = np.zeros((target_height, target_height, 3), dtype=np.uint8)
        draw_label(placeholder, f'{label}: missing')
        return placeholder
    return resize_to_height(frame, target_height)


def collect_view_image_paths(episode_dir, view_name):
    view_dir = episode_dir / view_name
    if not view_dir.is_dir():
        return []
    return sorted(view_dir.glob('color_*.jpg'))


def create_reference_video(episode_dir, filtered_data, source_timestamps,
                           video_path, view_names, video_height, annotate=False):
    kept_indices = filtered_data['original_indices']
    source_frame_count = len(filtered_data['timestamps'])
    if source_frame_count == 0:
        return

    dt = np.diff(source_timestamps)
    median_dt = float(np.median(dt)) if len(dt) else 0.1
    fps = 1.0 / max(median_dt, 1e-6)

    view_images = {view_name: collect_view_image_paths(episode_dir, view_name)
                   for view_name in view_names}

    tile_widths = []
    for view_name in view_names:
        sample_paths = view_images[view_name]
        sample_path = sample_paths[0] if sample_paths else None
        sample_image = load_resized_frame_or_placeholder(sample_path, video_height, view_name)
        tile_widths.append(sample_image.shape[1])
    target_tile_width = max(tile_widths) if tile_widths else video_height
    frame_width = target_tile_width * max(len(view_names), 1)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, video_height),
    )

    try:
        for new_index, old_index in enumerate(kept_indices):
            tiles = []
            for view_name in view_names:
                view_paths = view_images[view_name]
                image_path = view_paths[int(old_index)] if len(view_paths) > int(old_index) else None
                tile = load_resized_frame_or_placeholder(image_path, video_height, view_name)
                if annotate:
                    draw_label(tile, view_name)
                tiles.append(pad_to_width(tile, target_tile_width))

            frame = np.hstack(tiles)
            if annotate:
                overlay = f'cleaned frame {new_index:05d} | original frame {int(old_index):05d}'
                cv2.putText(frame, overlay, (12, video_height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()


def create_single_view_videos(episode_dir, filtered_data, source_timestamps,
                              output_dir, view_names, video_height, suffix):
    kept_indices = filtered_data['original_indices']
    if len(filtered_data['timestamps']) == 0:
        return []

    dt = np.diff(source_timestamps)
    median_dt = float(np.median(dt)) if len(dt) else 0.1
    fps = 1.0 / max(median_dt, 1e-6)
    written_paths = []

    for view_name in view_names:
        view_paths = collect_view_image_paths(episode_dir, view_name)
        sample_image = None
        for old_index in kept_indices:
            if len(view_paths) > int(old_index):
                sample_image = load_raw_frame(view_paths[int(old_index)])
                if sample_image is not None:
                    break
        if sample_image is None:
            sample_image = np.zeros((video_height, video_height, 3), dtype=np.uint8)
        frame_height, frame_width = sample_image.shape[:2]

        video_path = output_dir / f'{view_name}{suffix}'
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height),
        )

        try:
            for new_index, old_index in enumerate(kept_indices):
                image_path = view_paths[int(old_index)] if len(view_paths) > int(old_index) else None
                frame = load_raw_frame(image_path)
                if frame is None:
                    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

        written_paths.append(video_path)

    return written_paths


def clean_episode(src_episode_dir, dst_episode_dir, args):
    episode_data = load_episode_data(src_episode_dir)
    motion_info = build_keep_mask(
        episode_data['actions'],
        episode_data['timestamps'],
        args.pose_velocity_threshold,
        args.gripper_delta_threshold,
        args.gripper_context_frames,
    )
    filtered_data = filtered_episode_data(episode_data, motion_info['keep_mask'])

    save_cleaned_episode(dst_episode_dir, filtered_data)
    for camera_dir in list_camera_dirs(src_episode_dir):
        copy_filtered_camera_frames(camera_dir, dst_episode_dir / camera_dir.name,
                                    filtered_data['original_indices'])

    create_reference_video(
        src_episode_dir,
        filtered_data,
        episode_data['timestamps'],
        dst_episode_dir / args.video_name,
        args.views,
        args.video_height,
        annotate=False,
    )
    create_reference_video(
        src_episode_dir,
        filtered_data,
        episode_data['timestamps'],
        dst_episode_dir / args.annotated_video_name,
        args.views,
        args.video_height,
        annotate=True,
    )
    per_view_videos = create_single_view_videos(
        src_episode_dir,
        filtered_data,
        episode_data['timestamps'],
        dst_episode_dir,
        args.views,
        args.video_height,
        args.per_view_video_suffix,
    )

    total_frames = len(episode_data['timestamps'])
    kept_frames = len(filtered_data['timestamps'])
    removed_frames = total_frames - kept_frames
    removed_ratio = removed_frames / total_frames if total_frames else 0.0

    return {
        'episode': src_episode_dir.name,
        'total_frames': total_frames,
        'kept_frames': kept_frames,
        'removed_frames': removed_frames,
        'removed_ratio': removed_ratio,
        'pose_motion_frames': int(np.count_nonzero(motion_info['pose_motion_mask'])),
        'gripper_event_frames': int(np.count_nonzero(motion_info['gripper_event_mask'])),
        'output_episode': str(dst_episode_dir),
        'annotated_video': args.annotated_video_name,
        'per_view_videos': ','.join(path.name for path in per_view_videos),
    }


def write_summary_csv(output_root, summaries):
    csv_path = output_root / 'cleaning_summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'episode', 'total_frames', 'kept_frames', 'removed_frames',
            'removed_ratio', 'pose_motion_frames', 'gripper_event_frames',
            'output_episode', 'annotated_video', 'per_view_videos',
        ])
        writer.writeheader()
        writer.writerows(summaries)
    return csv_path


def main():
    args = parse_args()
    input_path = resolve_path(args.input_path)
    output_root = Path(args.output_path).expanduser().resolve()
    dataset_dir, episode_dirs = resolve_input(input_path, args.episode)

    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for episode_dir in episode_dirs:
        dst_episode_dir = output_root / episode_dir.name
        summary = clean_episode(episode_dir, dst_episode_dir, args)
        summaries.append(summary)
        print(
            f"[{summary['episode']}] kept {summary['kept_frames']}/{summary['total_frames']} frames "
            f"({1.0 - summary['removed_ratio']:.1%} retained); "
            f"video -> {dst_episode_dir / args.video_name}; "
            f"annotated -> {dst_episode_dir / args.annotated_video_name}; "
            f"per-view -> {summary['per_view_videos']}"
        )

    csv_path = write_summary_csv(output_root, summaries)
    print(f'Cleaning summary saved to: {csv_path}')
    print(f'Cleaned dataset root: {output_root}')


if __name__ == '__main__':
    main()