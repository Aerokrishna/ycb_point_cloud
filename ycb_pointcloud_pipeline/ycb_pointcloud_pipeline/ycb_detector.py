"""
Object detectors/segmenters for the RGB-D -> point cloud pipeline.

Default (and recommended) path on Jetson Thor: Grounded-SAM --
Grounding DINO (open-vocabulary text-prompted detection) + SAM ViT-H
(highest-quality mask decoder). Text prompts are built directly from
config/target_objects.yaml, and detections are name-matched against that
same list in segmentation_node.py, so only objects on your YAML list ever
get published -- everything else Grounding DINO/SAM might notice in the
frame is discarded.

---------------------------------------------------------------------------
Why the env vars below matter
---------------------------------------------------------------------------
HuggingFace `transformers` (a Grounding DINO dependency) auto-probes for
PyTorch/TensorFlow/Flax(JAX) backends the first time you import certain
classes, even if you never asked it to use TF or JAX. If a partial/stray
`jax` install exists anywhere on the system (pulled in transitively by
something else you installed) without a matching `jaxlib`, that probe
crashes the entire import with a confusing traceback that looks like it's
coming from Grounding DINO, when it's actually transformers' own backend
auto-detection. Setting these three env vars *before* transformers is ever
imported tells it to skip TF/Flax detection entirely, regardless of what
else happens to be installed on the system -- this fixes the failure mode
permanently rather than requiring you to keep jax/jaxlib uninstalled.
---------------------------------------------------------------------------
"""

import os
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch

# ---------------------------------------------------------------------------
# PLACEHOLDER PATHS -- replace with your actual downloaded checkpoint paths.
# See README.md for download commands.
# ---------------------------------------------------------------------------
SAM_CHECKPOINT_PATH = "/home/krishnapranav/models/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"  # highest-quality SAM checkpoint -- Thor's GPU can afford it

GROUNDING_DINO_CONFIG_PATH = "/home/krishnapranav/models/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/home/krishnapranav/models/groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25


class DummyDetector:
    """No model dependencies. Fixed center-box mask, labeled "test_object".
    Use to verify topics/sync/point-cloud plumbing before loading any model."""

    def __init__(self, **kwargs):
        pass

    def detect_and_segment(self, rgb_image):
        h, w = rgb_image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = True
        return [{"label": "test_object", "score": 1.0, "mask": mask}]


class SAMEverythingDetector:
    """
    Segments every distinct object-like region in the frame using SAM's
    automatic mask generator. No text prompts -- useful for a quick sanity
    check of what SAM can see, but returns generic labels (object_0,
    object_1, ...) that can't be matched against target_objects.yaml.
    Prefer GroundedSAMDetector for the actual target-filtered pipeline.
    """

    def __init__(self, checkpoint_path=None, model_type=None, device=None,
                 min_mask_area=800, points_per_side=32):
        self.checkpoint_path = checkpoint_path or SAM_CHECKPOINT_PATH
        self.model_type = model_type or SAM_MODEL_TYPE
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_mask_area = min_mask_area
        self.points_per_side = points_per_side
        self._mask_generator = None

    def _lazy_load(self):
        if self._mask_generator is not None:
            return
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(self.device)
        self._mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.90,
            min_mask_region_area=self.min_mask_area,
        )

    def detect_and_segment(self, rgb_image):
        self._lazy_load()
        results = self._mask_generator.generate(rgb_image)
        detections = []
        for i, r in enumerate(results):
            detections.append({
                "label": f"object_{i}",
                "score": float(r.get("predicted_iou", 1.0)),
                "mask": r["segmentation"].astype(bool),
            })
        return detections


class GroundedSAMDetector:
    """
    Grounding DINO (text-prompted, open-vocabulary boxes) + SAM ViT-H
    (box -> high-quality mask). This is the default, recommended detector:
    named detections that segmentation_node.py can match against
    target_objects.yaml, so only objects you actually listed ever get a
    point cloud published.

    `prompts` should be the exact list loaded from target_objects.yaml --
    segmentation_node.py passes it in at construction time.
    """

    def __init__(self, prompts, device=None):
        if not prompts:
            raise ValueError("GroundedSAMDetector needs a non-empty prompts list "
                              "(check config/target_objects.yaml)")
        self.prompts = prompts
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._grounding_model = None
        self._sam_predictor = None

    def _lazy_load(self):
        if self._grounding_model is not None:
            return

        from groundingdino.util.inference import load_model, predict
        from segment_anything import sam_model_registry, SamPredictor

        self._gd_predict = predict
        self._grounding_model = load_model(
            GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH
        ).to(self.device)

        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(self.device)
        self._sam_predictor = SamPredictor(sam)

    def detect_and_segment(self, rgb_image):
        """
        rgb_image: HxWx3 uint8 numpy array (RGB order)
        Returns a list of dicts: {"label": str, "score": float, "mask": HxW bool array}
        `label` is whatever phrase Grounding DINO matched from the prompt
        set -- segmentation_node.py does the final exact-match filtering
        against target_objects.yaml.
        """
        self._lazy_load()

        import groundingdino.datasets.transforms as T
        from PIL import Image

        pil_img = Image.fromarray(rgb_image)
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(pil_img, None)

        text_prompt = " . ".join(self.prompts)
        boxes, logits, phrases = self._gd_predict(
            model=self._grounding_model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=self.device,
        )

        if boxes.shape[0] == 0:
            return []

        h, w = rgb_image.shape[:2]
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h

        self._sam_predictor.set_image(rgb_image)
        results = []
        for box, score, phrase in zip(boxes_xyxy, logits, phrases):
            box_np = box.detach().cpu().numpy()
            masks, mask_scores, _ = self._sam_predictor.predict(
                box=box_np, multimask_output=False
            )
            results.append({
                "label": phrase,
                "score": float(score),
                "mask": masks[0].astype(bool),
            })
        return results
