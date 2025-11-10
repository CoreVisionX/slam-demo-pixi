"""Live stereo camera runner for the refactored SLAM system."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import cast

import cv2
import gtsam
import numpy as np
import rerun as rr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from registration.registration import StereoCalibration, StereoDepthFrame, StereoFrame
from slam import SlamConfig, create_default_slam_system
from slam.camera_utils import load_stereo_calibration_npz, split_wide_frame
from slam.matcher_factory import MatcherType
from slam.vision_odometry import OdometryEstimate, VisionOdometryEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the stereo SLAM pipeline on frames from a physical stereo camera."
    )
    parser.add_argument("--calib", type=str, default="calib.npz", help="Path to calib.npz.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width for the stereo stream.")
    parser.add_argument("--height", type=int, default=640, help="Capture height for the stereo stream.")
    parser.add_argument("--fps", type=int, default=None, help="Requested capture framerate.")
    parser.add_argument(
        "--split-mode",
        choices=("half", "px", "ratio"),
        default="half",
        help="How to split the incoming frame into left/right halves.",
    )
    parser.add_argument("--split-px", type=int, default=None, help="Split column when --split-mode=px.")
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.5,
        help="Split ratio (0..1) when --split-mode=ratio (default: 0.5).",
    )
    parser.add_argument(
        "--swap-halves",
        action="store_true",
        help="Swap the left/right halves after splitting.",
    )
    parser.add_argument(
        "--resize-to-calib",
        action="store_true",
        help="Resize each half to the calibration resolution after splitting.",
    )
    parser.add_argument(
        "--keyframe-translation",
        type=float,
        default=0.2,
        help="Translation threshold (meters) for creating a new keyframe.",
    )
    parser.add_argument(
        "--keyframe-rotation",
        type=float,
        default=10.0,
        help="Rotation threshold (degrees) for creating a new keyframe.",
    )
    parser.add_argument(
        "--loop-min-inliers",
        type=int,
        default=30,
        help="Minimum inliers required for accepting a loop closure.",
    )
    parser.add_argument(
        "--loop-workers",
        type=int,
        default=None,
        help="Override number of loop closure workers (defaults to heuristic).",
    )
    parser.add_argument(
        "--rerun-tcp",
        type=str,
        default=None,
        help="Optional rerun TCP endpoint (host:port). Leave unset to log locally only.",
    )
    parser.add_argument(
        "--rerun-app-id",
        type=str,
        default="live-stereo-slam",
        help="rerun app id used when logging is enabled.",
    )
    parser.add_argument(
        "--disable-rerun",
        action="store_true",
        help="Disable rerun logging entirely.",
    )
    parser.add_argument(
        "--disable-huber",
        action="store_true",
        help="Disable Huber loss on loop closures.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep time between processed frames (seconds).",
    )
    parser.add_argument(
        "--perf-log-threshold",
        type=float,
        default=None,
        help="If set, log performance sections for non-keyframe frames whose processing time "
        "exceeds this threshold (seconds).",
    )
    parser.add_argument(
        "--log-loop-closures",
        action="store_true",
        help="Log loop closures to the console.",
    )
    parser.add_argument(
        "--matcher",
        choices=("lighterglue", "orb"),
        default="orb",
        help="Feature matcher backend to use for both the frontend and odometry.",
    )
    parser.add_argument(
        "--vo-min-matches",
        type=int,
        default=30,
        help="Minimum feature matches required before accepting a visual odometry estimate.",
    )
    parser.add_argument(
        "--vo-min-inliers",
        type=int,
        default=30,
        help="Minimum PnP inliers required before accepting a visual odometry estimate.",
    )
    parser.add_argument(
        "--vo-failure-behavior",
        choices=("repeat", "identity"),
        default="repeat",
        help="Fallback odometry to use when visual odometry fails. "
        "'repeat' reuses the last successful delta, 'identity' injects no motion.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Optional limit on the number of frames to process before exiting.",
    )
    parser.add_argument(
        "--no-align-gt",
        action="store_true",
        help="Disable Umeyama alignment of ground-truth keyframes for logging (no GT is available, "
        "but this toggles rerun visualization behavior).",
    )
    parser.add_argument(
        "--no-rectify-inputs",
        dest="rectify_inputs",
        action="store_false",
        help="Disable frontend rectification (default: rectify).",
    )
    parser.add_argument(
        "--no-vo-rectify-inputs",
        dest="vo_rectify_inputs",
        action="store_false",
        help="Disable rectification inside the visual odometry estimator (default: rectify).",
    )
    parser.add_argument(
        "--max-read-failures",
        type=int,
        default=10,
        help="Number of consecutive camera read failures tolerated before exiting.",
    )
    parser.set_defaults(rectify_inputs=True, vo_rectify_inputs=True)
    return parser.parse_args()


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if os.name == "nt" else 0
    cap = cv2.VideoCapture(args.camera, backend)
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {args.camera}.")
    return cap


def _resize_if_needed(image: np.ndarray, width: int, height: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return image
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def acquire_frame(
    cap: cv2.VideoCapture,
    args: argparse.Namespace,
    calibration: StereoCalibration,
) -> StereoFrame | None:
    ret, wide_frame = cap.read()
    if not ret:
        return None
    left, right = split_wide_frame(
        wide_frame,
        mode=args.split_mode,
        split_px=args.split_px,
        ratio=args.split_ratio,
        swap_halves=args.swap_halves,
    )
    left = _resize_if_needed(left, calibration.width, calibration.height, args.resize_to_calib)
    right = _resize_if_needed(right, calibration.width, calibration.height, args.resize_to_calib)
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    return StereoFrame(left=left_rgb, right=right_rgb, calibration=calibration)


def should_accept_vo(estimate: OdometryEstimate, args: argparse.Namespace) -> bool:
    if estimate.pose is None:
        return False
    if estimate.match_count < args.vo_min_matches:
        return False
    if estimate.inlier_count < args.vo_min_inliers:
        return False
    return True


class LiveRerunStreamLogger:
    """Log rectified imagery, depth, and live pose each frame for easier debugging."""

    def __init__(self, calibration: StereoCalibration, *, base_path: str = "live_stream") -> None:
        self._base_path = base_path
        self._calibration = calibration
        self._positions: list[np.ndarray] = []
        rr.log(base_path, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            f"{base_path}/camera",
            rr.Pinhole(
                focal_length=[
                    float(calibration.K_left_rect[0, 0]),
                    float(calibration.K_left_rect[1, 1]),
                ],
                principal_point=[
                    float(calibration.K_left_rect[0, 2]),
                    float(calibration.K_left_rect[1, 2]),
                ],
                width=int(calibration.width),
                height=int(calibration.height),
            ),
            static=True,
        )

    def log_frame(self, frame_index: int, depth_frame: StereoDepthFrame | None) -> None:
        if depth_frame is None:
            return
        rr.set_time("frame", sequence=frame_index)
        rr.log(
            f"{self._base_path}/left_rect",
            rr.Image(np.ascontiguousarray(depth_frame.left_rect)),
        )
        rr.log(
            f"{self._base_path}/right_rect",
            rr.Image(np.ascontiguousarray(depth_frame.right_rect)),
        )
        depth_image = np.nan_to_num(
            depth_frame.left_depth.astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        rr.log(
            f"{self._base_path}/depth",
            rr.DepthImage(depth_image, meter=1.0),
        )

    def log_pose(self, frame_index: int, pose: gtsam.Pose3) -> None:
        translation = pose.translation()
        rotation = pose.rotation().toQuaternion().coeffs()
        position_array = translation
        self._positions.append(position_array)

        rr.set_time("frame", sequence=frame_index)
        rr.log(
            f"{self._base_path}/pose",
            rr.Transform3D(translation=position_array, quaternion=rotation),
        )

        if len(self._positions) >= 2:
            strips = [
                [self._positions[i], self._positions[i + 1]]
                for i in range(len(self._positions) - 1)
            ]
            rr.log(f"{self._base_path}/trajectory", rr.LineStrips3D(strips=strips, colors=[[0, 128, 255]], radii=[0.05]))


def main() -> None:
    multiprocessing.set_start_method("spawn")
    args = parse_args()

    calibration = load_stereo_calibration_npz(args.calib)
    matcher_type = cast(MatcherType, args.matcher)
    config = SlamConfig(
        keyframe_translation_threshold=args.keyframe_translation,
        keyframe_rotation_threshold=float(np.deg2rad(args.keyframe_rotation)),
        loop_min_inliers=args.loop_min_inliers,
        loop_worker_count=args.loop_workers,
        rerun_tcp_address=args.rerun_tcp,
        rerun_app_id=args.rerun_app_id,
        use_huber_loss=not args.disable_huber,
        align_ground_truth=not args.no_align_gt,
        enable_rerun_logging=not args.disable_rerun,
        feature_matcher=matcher_type,
        rectify_inputs=args.rectify_inputs,
    )

    slam = create_default_slam_system(
        calibration_matrix=calibration.K_left_rect,
        config=config,
    )
    vision_odometry = VisionOdometryEstimator(matcher_type, rectify_inputs=args.vo_rectify_inputs)
    stream_logger = None
    if not args.disable_rerun:
        stream_logger = LiveRerunStreamLogger(calibration)

    cap = open_camera(args)
    last_good_odometry: gtsam.Pose3 | None = None
    keyframe_count = 0
    loop_closure_count = 0
    last_snapshot = None
    frame_count = 0
    consecutive_read_failures = 0
    first_frame = True

    try:
        while args.frame_limit is None or frame_count < args.frame_limit:
            frame = acquire_frame(cap, args, calibration)
            if frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= args.max_read_failures:
                    print("Camera read failed too many times, exiting.")
                    break
                print(f"Camera read failed ({consecutive_read_failures}/{args.max_read_failures}), retrying...")
                time.sleep(0.05)
                continue

            consecutive_read_failures = 0
            odometry_input: gtsam.Pose3 | None = None
            vo_estimate: OdometryEstimate | None = None

            if first_frame:
                vision_odometry.prime(frame)
                first_frame = False
            else:
                vo_estimate = vision_odometry.estimate(frame)
                if should_accept_vo(vo_estimate, args):
                    odometry_input = vo_estimate.pose
                    last_good_odometry = vo_estimate.pose
                else:
                    fallback_mode = args.vo_failure_behavior
                    fallback = None
                    if fallback_mode == "repeat" and last_good_odometry is not None:
                        fallback = last_good_odometry
                    if fallback is None:
                        fallback_mode = "identity"
                        fallback = gtsam.Pose3.Identity()

                    odometry_input = fallback
                    print(
                        f"[VisionOdom] Using {fallback_mode} fallback "
                        f"(matches={vo_estimate.match_count if vo_estimate else 0}, "
                        f"inliers={vo_estimate.inlier_count if vo_estimate else 0})."
                    )

            result = slam.process_step(
                frame,
                odometry_input,
                ground_truth_pose=None,
            )
            frame_count += 1
            last_snapshot = result.performance

            if stream_logger is not None:
                stream_logger.log_frame(result.frame_index, vision_odometry.latest_depth_frame)
                stream_logger.log_pose(result.frame_index, result.latest_pose)

            if (
                vo_estimate is not None
                and not should_accept_vo(vo_estimate, args)
                and vo_estimate.failure_reason is not None
            ):
                print(
                    f"[VisionOdom] Estimation failure at frame {result.frame_index:04d}: "
                    f"{vo_estimate.failure_reason} "
                    f"(matches={vo_estimate.match_count}, inliers={vo_estimate.inlier_count})"
                )

            total_duration = result.performance.total_duration
            log_sections = result.is_keyframe or (
                args.perf_log_threshold is not None and total_duration >= args.perf_log_threshold
            )

            if result.is_keyframe:
                keyframe_count += 1

            if log_sections:
                section_breakdown = ", ".join(
                    f"{name}:{duration:.3f}s"
                    for name, duration in sorted(result.performance.sections.items())
                    if duration > 0.0
                )
                if result.is_keyframe:
                    preprocess = result.frontend_timings.total if result.frontend_timings else 0.0
                    msg = f"[Frame {result.frame_index:04d}] keyframe {keyframe_count:04d} "
                    msg += f"(preprocess {preprocess:.2f}s, total {total_duration:.2f}s)\n"
                    msg += f"    sections: {section_breakdown}\n"
                    print(msg)

                    if not args.disable_rerun:
                        rr.set_time("frame", sequence=result.frame_index)
                        rr.log("logs", rr.TextLog(msg))
                else:
                    header = (
                        f"[Frame {result.frame_index:04d}] non-keyframe total {total_duration:.2f}s"
                    )
                    if args.perf_log_threshold is not None:
                        header += f" (threshold {args.perf_log_threshold:.2f}s)"
                    msg = f"{header}\n    sections: {section_breakdown}"
                    print(msg)

                    if not args.disable_rerun:
                        rr.set_time("frame", sequence=result.frame_index)
                        rr.log("logs", rr.TextLog(msg))

            if result.loop_closures_added:
                loop_closure_count += result.loop_closures_added
                if args.log_loop_closures:
                    print(
                        f"    Integrated {result.loop_closures_added} loop closures "
                        f"(total {loop_closure_count})"
                    )

            time.sleep(args.sleep)
    finally:
        cap.release()
        slam.shutdown()
        print(
            f"Finished with {keyframe_count} keyframes and {loop_closure_count} loop closures."
        )
        if last_snapshot is not None and last_snapshot.rolling_stats:
            total_runtime = last_snapshot.cumulative_time
            avg_frame_time = last_snapshot.mean_frame_time
            fps = last_snapshot.mean_fps
            print(
                f"Processed {last_snapshot.frame_count} frames | total {total_runtime:.2f}s | "
                f"avg {avg_frame_time:.3f}s ({fps:.2f} fps)"
            )


if __name__ == "__main__":
    main()