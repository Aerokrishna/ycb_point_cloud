#!/usr/bin/env python3
"""
ROS2 node (Jetson Thor target): subscribes to synchronized RealSense
color + aligned-depth + camera_info topics, runs Grounded-SAM detection
restricted to the object list in config/target_objects.yaml, deprojects
each matched object's masked pixels into a point cloud, publishes one
PointCloud2 per matched object, saves .pcd files, and shows a live popup
window with bounding boxes drawn around each match.

Objects not on the YAML list are never detected for (they're not even in
the text prompt) or published. Objects on the list that aren't currently
visible simply produce no message on that frame -- their topic just goes
quiet until the object reappears.

Run (after building the workspace and sourcing it):
    ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py
"""

import os
import yaml
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import message_filters
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from ycb_pointcloud_pipeline.pointcloud_utils import (
    CameraIntrinsics, mask_to_pointcloud, points_to_ros2_pointcloud2, save_pcd_ascii,
)
from ycb_pointcloud_pipeline.ycb_detector import (
    DummyDetector, SAMEverythingDetector, GroundedSAMDetector,
)


def _default_target_objects_path():
    try:
        return os.path.join(
            get_package_share_directory('ycb_pointcloud_pipeline'),
            'config', 'target_objects.yaml',
        )
    except Exception:
        # Falls back gracefully if run outside an installed/sourced package
        # (e.g. quick local testing straight from the source tree).
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'target_objects.yaml')


def load_target_objects(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    targets = data.get('target_objects', []) if data else []
    if not targets:
        raise ValueError(f"No target_objects found in {yaml_path} -- add at least one YCB "
                          f"object name to the list before running detection.")
    return [str(t).strip() for t in targets]


def _normalize(name):
    return name.strip().lower()


class YCBSegmentationNode(Node):
    def __init__(self):
        super().__init__('ycb_segmentation_node')

        # ---- Parameters -----------------------------------------------
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('frame_id', 'camera_color_optical_frame')
        # "dummy" (no models), "sam_all" (SAM segment-everything, generic
        # labels, can't be filtered by name), or "grounded_sam" (DEFAULT --
        # named detections matched against target_objects.yaml).
        self.declare_parameter('detector_mode', 'grounded_sam')
        self.declare_parameter('target_objects_yaml', _default_target_objects_path())
        self.declare_parameter('save_pcd', True)
        self.declare_parameter('output_dir', os.path.expanduser('~/ycb_pointclouds'))
        self.declare_parameter('depth_scale', 1000.0)  # RealSense: mm -> m
        self.declare_parameter('depth_trunc', 3.0)      # meters
        self.declare_parameter('show_debug_window', True)
        # Jetson Thor has a real GPU, so unlike a CPU-only setup this can
        # usually keep up close to every frame. Raise this if the popup
        # window still lags on your unit.
        self.declare_parameter('detect_every_n_frames', 1)

        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.save_pcd = self.get_parameter('save_pcd').value
        self.output_dir = self.get_parameter('output_dir').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.depth_trunc = self.get_parameter('depth_trunc').value
        self.show_debug_window = self.get_parameter('show_debug_window').value
        self.detect_every_n_frames = self.get_parameter('detect_every_n_frames').value

        if self.save_pcd:
            os.makedirs(self.output_dir, exist_ok=True)

        # ---- Target object list (YAML) -----------------------------------
        target_yaml_path = self.get_parameter('target_objects_yaml').value
        self.target_objects = load_target_objects(target_yaml_path)
        self._target_lookup = {_normalize(t) for t in self.target_objects}
        self.get_logger().info(
            f"Loaded {len(self.target_objects)} target object(s) from {target_yaml_path}: "
            f"{self.target_objects}"
        )

        # ---- Detector ----------------------------------------------------
        detector_mode = self.get_parameter('detector_mode').value
        if detector_mode == 'dummy':
            self.detector = DummyDetector()
            self.get_logger().warn(
                "detector_mode='dummy': publishing a placeholder center-box mask. "
                "Set detector_mode to 'grounded_sam' for real, target-filtered detection."
            )
        elif detector_mode == 'sam_all':
            self.detector = SAMEverythingDetector()
            self.get_logger().warn(
                "detector_mode='sam_all': segments everything with generic labels -- "
                "these CANNOT be matched against target_objects.yaml, so nothing will "
                "pass the target filter below. Use 'grounded_sam' for named, filtered output."
            )
        elif detector_mode == 'grounded_sam':
            self.detector = GroundedSAMDetector(prompts=self.target_objects)
            self.get_logger().info(
                "detector_mode='grounded_sam': detecting only the objects listed in "
                "target_objects.yaml. Make sure SAM_CHECKPOINT_PATH and "
                "GROUNDING_DINO_*_PATH are set correctly in ycb_detector.py."
            )
        else:
            raise ValueError(f"Unknown detector_mode: {detector_mode!r}")

        # ---- Intrinsics (placeholder until CameraInfo arrives) -----------
        self.intrinsics = CameraIntrinsics()
        self._got_camera_info = False

        # ---- I/O -----------------------------------------------------
        self.bridge = CvBridge()
        self.publishers_by_label = {}

        self.create_subscription(CameraInfo, info_topic, self.camera_info_cb, 10)

        color_sub = message_filters.Subscriber(self, Image, color_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=0.05
        )
        self.ts.registerCallback(self.rgbd_cb)

        self.frame_count = 0
        self._incoming_frame_idx = 0
        self.get_logger().info(
            f"Listening on:\n  color: {color_topic}\n  depth: {depth_topic}\n  info: {info_topic}"
        )

    def camera_info_cb(self, msg: CameraInfo):
        if not self._got_camera_info:
            self.intrinsics = CameraIntrinsics.from_camera_info(msg)
            self._got_camera_info = True
            self.get_logger().info(
                f"Got real camera intrinsics: fx={self.intrinsics.fx:.2f} "
                f"fy={self.intrinsics.fy:.2f} cx={self.intrinsics.cx:.2f} "
                f"cy={self.intrinsics.cy:.2f} ({self.intrinsics.width}x{self.intrinsics.height})"
            )

    def get_publisher(self, label):
        safe_label = label.replace(' ', '_')
        if safe_label not in self.publishers_by_label:
            topic = f"/ycb_pointclouds/{safe_label}"
            self.publishers_by_label[safe_label] = self.create_publisher(
                PointCloud2, topic, 10,
            )
            self.get_logger().info(f"Publishing new object topic: {topic}")
        return self.publishers_by_label[safe_label]

    def _filter_to_targets(self, detections):
        """Keep only detections whose label matches (or contains/is
        contained by) one of the configured target_objects. Everything
        else -- even if the detector found it -- is dropped here, so it's
        never published."""
        kept = []
        for det in detections:
            norm_label = _normalize(det['label'])
            match = None
            if norm_label in self._target_lookup:
                match = norm_label
            else:
                for target in self._target_lookup:
                    if target in norm_label or norm_label in target:
                        match = target
                        break
            if match is not None:
                det = dict(det)
                det['label'] = match  # normalize to the YAML name for topic naming
                kept.append(det)
        return kept

    def rgbd_cb(self, color_msg: Image, depth_msg: Image):
        self._incoming_frame_idx += 1
        if self._incoming_frame_idx % self.detect_every_n_frames != 0:
            return

        rgb = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        raw_detections = self.detector.detect_and_segment(rgb)
        detections = self._filter_to_targets(raw_detections)

        if self.show_debug_window:
            self._show_debug_window(rgb, detections)

        if not detections:
            return

        self.frame_count += 1
        for det in detections:
            label = det['label']
            mask = det['mask']

            points, colors = mask_to_pointcloud(
                rgb, depth, mask, self.intrinsics,
                depth_scale=self.depth_scale, depth_trunc=self.depth_trunc,
            )
            if points.shape[0] == 0:
                continue

            msg = points_to_ros2_pointcloud2(points, colors, self.frame_id, color_msg.header.stamp)
            self.get_publisher(label).publish(msg)

            if self.save_pcd:
                fname = os.path.join(
                    self.output_dir, f"{label.replace(' ', '_')}_{self.frame_count:05d}.pcd"
                )
                save_pcd_ascii(fname, points, colors)

        self.get_logger().info(
            f"Frame {self.frame_count}: published {len(detections)} target object cloud(s): "
            f"{[d['label'] for d in detections]}"
        )

    def _show_debug_window(self, rgb, detections):
        """Popup window (OpenCV) with a bounding box + label drawn around
        each matched target object, derived from that object's mask."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        for det in detections:
            mask = det['mask']
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())

            cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
            label_text = f"{det['label']} ({det['score']:.2f})"
            text_y = max(y0 - 8, 15)
            cv2.putText(bgr, label_text, (x0, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        status = f"{len(detections)}/{len(self.target_objects)} target object(s) visible"
        cv2.putText(bgr, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow("YCB Detections", bgr)
        cv2.waitKey(1)  # pumps the GUI event loop; required for the window to render/update


def main(args=None):
    rclpy.init(args=args)
    node = YCBSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
