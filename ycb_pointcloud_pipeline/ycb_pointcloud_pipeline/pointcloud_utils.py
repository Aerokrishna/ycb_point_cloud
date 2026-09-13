"""
Utilities for turning a masked RGB-D frame into a 3D point cloud,
and converting to a ROS2 PointCloud2 message / .pcd file.

No Open3D dependency -- pure NumPy. Open3D's packaging has proven
unreliable across environments (unguarded auto-imports of sklearn/
torch/tensorflow/jax inside its `ml` submodule), so this module
does the deprojection, a simple density-based outlier filter, and
PCD writing itself instead.
"""

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


# ---------------------------------------------------------------------------
# CALIBRATED INTRINSICS (D435i, this unit)
# ---------------------------------------------------------------------------
# From the camera's own factory calibration at 848x480 color resolution:
#   K = [[604.725353,   0,        425.857373],
#        [  0,        602.581260, 236.580043],
#        [  0,          0,          1       ]]
#   D = [0, 0, 0, 0, 0]  (RealSense color streams are pre-rectified, so this
#                         pipeline doesn't apply distortion correction)
#
# These are only a FALLBACK: the ROS node overwrites them automatically the
# moment it receives a real CameraInfo message (see camera_info_cb in
# segmentation_node.py), so they only matter if you run mask_to_pointcloud()
# offline without a live camera_info topic. If you re-run calibration or
# switch resolutions, update these to match (they are resolution-specific --
# cx/cy scale with width/height, fx/fy do not stay valid across resolutions).
# Re-check via: ros2 topic echo /camera/camera/color/camera_info --once
# ---------------------------------------------------------------------------
PLACEHOLDER_FX = 604.725353
PLACEHOLDER_FY = 602.581260
PLACEHOLDER_CX = 425.857373
PLACEHOLDER_CY = 236.580043
PLACEHOLDER_WIDTH = 848
PLACEHOLDER_HEIGHT = 480


class CameraIntrinsics:
    def __init__(self, fx=PLACEHOLDER_FX, fy=PLACEHOLDER_FY,
                 cx=PLACEHOLDER_CX, cy=PLACEHOLDER_CY,
                 width=PLACEHOLDER_WIDTH, height=PLACEHOLDER_HEIGHT):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

    @classmethod
    def from_camera_info(cls, msg):
        """Build from a sensor_msgs/CameraInfo message."""
        k = msg.k  # row-major 3x3
        return cls(fx=k[0], fy=k[4], cx=k[2], cy=k[5],
                   width=msg.width, height=msg.height)


def mask_to_pointcloud(color_img, depth_img, mask, intrinsics,
                        depth_scale=1000.0, depth_trunc=3.0,
                        remove_outliers=True):
    """
    Deproject the masked pixels of an aligned RGB-D frame into a 3D point cloud.

    color_img : HxWx3 uint8 array (RGB order)
    depth_img : HxW uint16 (or float32) array, raw depth units (mm if uint16)
    mask      : HxW bool array, True where the object of interest is
    intrinsics: CameraIntrinsics instance
    depth_scale: divide raw depth by this to get meters (RealSense default: 1000.0 for mm)
    depth_trunc: ignore points farther than this many meters

    Returns (points, colors): Nx3 float32 array (meters, camera frame) and
    Nx3 uint8 array (RGB). N == 0 if nothing valid falls inside the mask.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    z = depth_img[ys, xs].astype(np.float32) / depth_scale
    valid = (z > 0.0) & (z <= depth_trunc)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    ys, xs, z = ys[valid], xs[valid], z[valid]

    x = (xs.astype(np.float32) - intrinsics.cx) * z / intrinsics.fx
    y = (ys.astype(np.float32) - intrinsics.cy) * z / intrinsics.fy

    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = color_img[ys, xs].astype(np.uint8)

    if remove_outliers and points.shape[0] > 20:
        points, colors = _density_outlier_filter(points, colors)

    return points, colors


def _density_outlier_filter(points, colors, voxel_size=0.005, min_neighbors=3):
    """
    Lightweight stand-in for Open3D's remove_statistical_outlier, using only
    NumPy: bins points into a voxel grid, sums point counts over each
    voxel's 3x3x3 neighborhood, and drops points sitting in low-density
    neighborhoods. Cheap way to strip sparse noise from mask-edge depth
    bleeding without needing a KD-tree library.
    """
    voxel_idx = np.floor(points / voxel_size).astype(np.int64)

    keys, inverse, counts = np.unique(
        voxel_idx, axis=0, return_inverse=True, return_counts=True
    )
    key_to_count = {tuple(k): c for k, c in zip(keys.tolist(), counts.tolist())}

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    ]

    neighborhood_counts = np.zeros(keys.shape[0], dtype=np.int64)
    for i, k in enumerate(keys.tolist()):
        total = 0
        for dx, dy, dz in neighbor_offsets:
            total += key_to_count.get((k[0] + dx, k[1] + dy, k[2] + dz), 0)
        neighborhood_counts[i] = total

    keep_voxel = neighborhood_counts >= min_neighbors
    keep_mask = keep_voxel[inverse]

    return points[keep_mask], colors[keep_mask]


def points_to_ros2_pointcloud2(points, colors, frame_id, stamp):
    """Pack Nx3 points + Nx3 uint8 colors into a sensor_msgs/PointCloud2."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)

    rgb_packed = (colors[:, 0].astype(np.uint32) << 16) | \
                 (colors[:, 1].astype(np.uint32) << 8) | \
                 (colors[:, 2].astype(np.uint32))
    rgb_float = rgb_packed.view(np.float32)

    cloud_arr = np.zeros(points.shape[0], dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('rgb', np.float32),
    ])
    cloud_arr['x'] = points[:, 0]
    cloud_arr['y'] = points[:, 1]
    cloud_arr['z'] = points[:, 2]
    cloud_arr['rgb'] = rgb_float

    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = cloud_arr.shape[0]
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * cloud_arr.shape[0]
    msg.is_dense = True
    msg.data = cloud_arr.tobytes()
    return msg


def save_pcd_ascii(path, points, colors):
    """Write an ASCII .pcd file (PCL format) -- readable by PCL, Open3D,
    CloudCompare, MeshLab, etc. No Open3D needed to produce it."""
    n = points.shape[0]
    rgb_packed = (colors[:, 0].astype(np.uint32) << 16) | \
                 (colors[:, 1].astype(np.uint32) << 8) | \
                 (colors[:, 2].astype(np.uint32))
    rgb_float = rgb_packed.view(np.float32)

    with open(path, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {n}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")
        for i in range(n):
            f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} {rgb_float[i]}\n")
