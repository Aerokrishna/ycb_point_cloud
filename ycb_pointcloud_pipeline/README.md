# ycb_pointcloud_pipeline (Jetson Thor, Ubuntu 24.04 / ROS2 Jazzy)

RealSense D435i -> RGB + aligned depth -> Grounded-SAM detection, filtered
against a YAML list of target YCB objects -> per-object point cloud
(published as `PointCloud2` + saved as `.pcd`) -> live popup window with
bounding boxes around whatever's currently detected.

Only objects listed in `config/target_objects.yaml` are ever detected for
or published. Anything else in the frame is ignored; a listed object that
isn't currently visible just produces no message that frame.

---

## 0. Assumed setup

This revision assumes:
- **Jetson Thor**, Ubuntu **24.04 (Noble)** as the base OS
- **ROS 2 Jazzy already installed and working**. Unlike Humble, Jazzy is
  the distro upstream OSRF officially targets *for* 24.04 -- so this is
  the standard package set, not a vendor backport, which makes this
  combination the more straightforward one of the two.
- **Python 3.12** as the system `python3`. This is also Jazzy's own
  official target Python version (unlike Humble, which targets 3.10),
  so there's no version mismatch to work around here.

Confirm before proceeding:
```bash
lsb_release -a          # expect: Ubuntu 24.04
python3 --version       # expect: Python 3.12.x
ros2 --version          # confirm it reports Jazzy
echo $ROS_DISTRO        # expect: jazzy
nvcc --version
```
If any of these don't match, some commands below may need adjusting
(especially the PyTorch wheel source in step 3) -- treat this file as a
starting point, not gospel, for a platform this new.

---

## 1. System packages (apt)

Since ROS 2 Jazzy is already installed, this just adds the packages
this pipeline needs on top of it:

```bash
sudo apt update
sudo apt install -y \
    python3-venv \
    python3-pip \
    ros-jazzy-realsense2-camera \
    ros-jazzy-diagnostic-updater \
    ros-jazzy-diagnostic-msgs \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-opencv \
    python3-colcon-common-extensions
```

If any `ros-jazzy-*` package 404s, your ROS 2 apt sources aren't fully
set up -- follow OSRF's standard ROS 2 Jazzy install instructions for
Ubuntu 24.04 (adding `packages.ros.org` as an apt source) before retrying
the block above.

Plug in the D435i via USB and sanity-check it's detected:
```bash
rs-enumerate-devices
```

---

## 2. Create the venv

```bash
python3 -m venv ~/pcld --system-site-packages
source ~/pcld/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
```

`--system-site-packages` is required -- it's what lets the venv see
ROS2's apt-installed Python packages (`rclpy`, `cv_bridge`, etc.)
alongside everything you `pip install` below.

**Keep this venv active for every command in the rest of this file.**

---

## 3. PyTorch (GPU build for Thor)

NVIDIA changed how they distribute PyTorch for Thor specifically: unlike
Orin (which needs the `jetson-ai-lab` mirror), Thor is meant to get
properly built wheels through more standard channels. Try this first:

```bash
pip install torch torchvision
```

Then immediately verify it actually sees the GPU:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

**If `torch.cuda.is_available()` prints `False`** (i.e. you silently got
a CPU-only wheel), uninstall and pull from the Jetson-specific index
instead, adjusting the JetPack/CUDA tag to match what `nvcc --version`
reported in step 0:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp7/cu130
```

That exact URL path (`jp7/cu130`) is a best guess based on the JetPack
7/CUDA 13 pairing -- if it 404s, browse
`https://pypi.jetson-ai-lab.io/` directly to find whatever folder
matches your JetPack version, since Jetson AI Lab's available builds
change over time and I can't guarantee this exact path stays valid.

---

## 4. Segment Anything (SAM) + Grounding DINO

```bash
pip install opencv-python numpy pyyaml

# Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git

# Grounding DINO -- installing from source is more reliable than the
# groundingdino-py pip package for getting a build that matches your
# installed torch/CUDA version.
cd ~
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

**Do NOT install these** -- none of them are needed, and jax/tensorflow
in particular are what caused import crashes in earlier iterations of
this pipeline (`transformers`, a Grounding DINO dependency, auto-probes
for them even when unused, and a partial/broken install of either one
crashes the whole node on startup):
```
jax, jaxlib, flax, optax, chex, tensorflow, open3d
```
If anything later pulls one of these in as a transitive dependency,
uninstall it immediately (`pip uninstall -y jax jaxlib flax tensorflow`)
-- the code also sets `USE_TF=0` / `USE_FLAX=0` before importing
`transformers` as a second line of defense, so a stray jax/tensorflow
install shouldn't crash things anymore, but keeping them out entirely is
cleaner and saves disk space.

---

## 5. Download model checkpoints

```bash
mkdir -p ~/models && cd ~/models

# SAM ViT-H -- the largest/most accurate checkpoint (~2.4 GB). Thor's GPU
# can afford this; this pipeline defaults to it.
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Grounding DINO (Swin-T config + weights)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
```

Verify both large files downloaded completely (not a few KB from a
truncated connection):
```bash
ls -lh ~/models/
```

If you cloned GroundingDINO into `~/GroundingDINO/` above, the config
path is likely also already sitting at
`~/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py` --
either that or the `wget`'d copy above works, they're identical.

The paths in `ycb_pointcloud_pipeline/ycb_detector.py` already default to
`~/models/...` matching the commands above:
```python
SAM_CHECKPOINT_PATH = "/home/krishnapranav/models/sam_vit_h_4b8939.pth"
GROUNDING_DINO_CONFIG_PATH = "/home/krishnapranav/models/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/home/krishnapranav/models/groundingdino_swint_ogc.pth"
```
If your username or paths differ, edit these three lines to match.

---

## 6. Choose your target YCB objects

Edit `config/target_objects.yaml`:
```yaml
target_objects:
  - "mustard bottle"
  - "banana"
  - "mug"
```
Only objects listed here get detected, segmented, and published. Add or
remove lines freely -- no code changes needed, the node reads this file
at startup.

---

## 7. Build the ROS2 package

```bash
mkdir -p ~/ycb_pcld_ws/src
cp -r ycb_pointcloud_pipeline ~/ycb_pcld_ws/src/
cd ~/ycb_pcld_ws
source /opt/ros/jazzy/setup.bash
python3 -m colcon build --packages-select ycb_pointcloud_pipeline
source install/setup.bash
```

Always build with `python3 -m colcon` (not bare `colcon`) while your venv
is active -- `colcon` itself is an apt-installed script with its own
shebang pointing at the system Python, so invoking it directly bypasses
your venv entirely and produces node scripts that can't see anything you
pip-installed above.

---

## 8. Run

```bash
source ~/pcld/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/ycb_pcld_ws/install/setup.bash
ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py
```

Expect, in order:
1. RealSense driver comes up, opens 848x480@30 color + depth.
2. Segmentation node loads your YAML target list, then loads Grounding
   DINO + SAM ViT-H onto the GPU (this step can take a while the first
   time -- model loading, not per-frame inference).
3. A window titled **"YCB Detections"** pops up showing the live color
   feed with green bounding boxes + labels drawn around any currently
   visible target object, plus an "N/M target object(s) visible" counter.
4. For each detected target object, a point cloud publishes on
   `/ycb_pointclouds/<object_name>` and a `.pcd` file lands in
   `~/ycb_pointclouds/`.

Check topics are live:
```bash
ros2 topic list | grep ycb_pointclouds
ros2 topic hz /ycb_pointclouds/mustard_bottle
```

---

## 9. Verifying the point cloud

- **RViz2**: Fixed Frame `camera_color_optical_frame`, add a `PointCloud2`
  display on `/ycb_pointclouds/<object_name>`, Color Transformer "RGB".
  You should see a cluster of colored points roughly tracing the object's
  shape, at roughly its real distance from the camera.
- **Metric check**: open a saved `.pcd` and compare its bounding-box
  extent against the object's real measured dimensions:
  ```python
  import numpy as np
  pts = np.loadtxt('/path/to/file.pcd', skiprows=11, usecols=(0,1,2))
  print("bounding box (m):", pts.max(axis=0) - pts.min(axis=0))
  ```

`.pcd` files live in `~/ycb_pointclouds/` by default (override via the
`output_dir` parameter) -- one file per detected target object per
processed frame, named `<object_name>_<frame_number>.pcd`. Nothing
currently caps or cleans this directory up, so it'll grow quickly if left
running.

---

## 10. Tuning

All via ROS2 parameters (edit `launch/ycb_pipeline.launch.py` or override
on the command line):

- `detect_every_n_frames` (default 1): raise this if the popup window
  still feels laggy even on Thor's GPU.
- `show_debug_window` (default true): set false to run headless.
- `target_objects_yaml`: point at a different YAML file without touching
  code, e.g. to swap object sets between runs.
- `save_pcd` (default true): set false if you only need the live
  `PointCloud2` topics and don't want files piling up on disk.

---

## 11. If something breaks: nuke and rebuild the venv

Rather than uninstalling packages one at a time, it's cleaner to start
over if things get into a bad state:

```bash
deactivate 2>/dev/null
rm -rf ~/pcld
python3 -m venv ~/pcld --system-site-packages
source ~/pcld/bin/activate
```
Then redo steps 2 onward. Also `rm -rf ~/ycb_pcld_ws/build ~/ycb_pcld_ws/install ~/ycb_pcld_ws/log` and rebuild the ROS2 package (step 7) any time you recreate the venv, since a fresh venv means a fresh Python interpreter path baked into the built node scripts.
