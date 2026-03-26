# Headless GEM-X demo: preprocessing + HPE inference only (no rendering).
# ruff: noqa: E402, I001
import argparse
import os
import sys
import functools
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from tqdm import tqdm

torch.load = functools.partial(torch.load, weights_only=False)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.utils.geo_transform import (
    apply_T_on_points,
    compute_cam_angvel,
    compute_cam_tvel,
    compute_T_ayfz2ay,
    get_bbx_xys_from_xyxy,
    normalize_T_w2c,
)
from gem.utils.net_utils import detach_to_cpu, to_cuda
from gem.utils.pylogger import Log
from gem.utils.sam3db_extractor import SAM3DBExtractor
from gem.utils.vitpose_extractor import VitPoseExtractor
from gem.utils.video_io_utils import get_video_lwh, read_video_np
from gem.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch
from gem.utils.cam_utils import estimate_K, get_a_pred_cam
from gem.utils.kp2d_utils import smooth_bbx_xyxy, render_2d_keypoints


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="outputs/demo_soma")
    parser.add_argument("-s", "--static_cam", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--exp", type=str, default="gem_soma_regression")
    parser.add_argument(
        "--detector_name", type=str, default="vitdet",
        help="Human detector: 'vitdet' or 'sam3'. Set empty to skip.",
    )
    parser.add_argument(
        "--bbx", type=str, default=None,
        help="Pre-computed bbx.pt (dict with bbx_xyxy and bbx_xys). Skips detection.",
    )
    return parser.parse_args()


def _build_cfg(args):
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"
    overrides = [
        f"exp={args.exp}",
        f"video_name={video_path.stem}",
        f"video_path={video_path}",
        f"output_root={args.output_root}",
        f"static_cam={str(args.static_cam).lower()}",
        f"verbose={str(args.verbose).lower()}",
        "render_mhr=false",
        "use_wandb=false",
        "task=test",
        f"detector_name={args.detector_name}",
    ]
    if args.ckpt is not None:
        overrides.append(f"ckpt_path={args.ckpt}")
    with initialize_config_dir(version_base="1.3", config_dir=str(cfg_dir)):
        cfg = compose(config_name="demo_soma", overrides=overrides)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    return cfg


@torch.no_grad()
def _run_human_detection(frames, paths, L, W, H, detector_name):
    """Run batched detection on pre-loaded frames."""
    sam3db_root = str(PROJECT_ROOT / "third_party" / "sam-3d-body")
    if sam3db_root not in sys.path:
        sys.path.insert(0, sam3db_root)
    from tools.build_detector import HumanDetector

    Log.info(f"[Preprocess] Running human detection with '{detector_name}'...")
    detector = HumanDetector(name=detector_name, device="cuda")

    det_batch_size = 8
    all_boxes = []

    if detector_name == "vitdet":
        import detectron2.data.transforms as T
        IMAGE_SIZE = 1024
        transforms = T.ResizeShortestEdge(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE)

        for start in tqdm(range(0, len(frames), det_batch_size), desc="Detect Humans"):
            end = min(start + det_batch_size, len(frames))
            batch_inputs = []
            for i in range(start, end):
                img_bgr = frames[i][..., ::-1].copy()
                img_t = transforms(T.AugInput(img_bgr)).apply_image(img_bgr)
                img_t = torch.as_tensor(img_t.astype("float32").transpose(2, 0, 1))
                batch_inputs.append({"image": img_t, "height": H, "width": W})

            det_outputs = detector.detector(batch_inputs)
            for det_out in det_outputs:
                instances = det_out["instances"]
                valid = (instances.pred_classes == 0) & (instances.scores > 0.5)
                if valid.sum() == 0:
                    all_boxes.append(np.array([0.0, 0.0, W - 1, H - 1]))
                else:
                    boxes = instances.pred_boxes.tensor[valid].cpu().numpy()
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    all_boxes.append(boxes[areas.argmax()])
    else:
        for i in tqdm(range(len(frames)), desc="Detect Humans"):
            img_bgr = frames[i][..., ::-1].copy()
            boxes = detector.run_human_detection(img_bgr)
            if len(boxes) == 0:
                all_boxes.append(np.array([0.0, 0.0, W - 1, H - 1]))
            else:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                all_boxes.append(boxes[areas.argmax()])

    bbx_xyxy = torch.from_numpy(np.stack(all_boxes, axis=0)).float()
    bbx_xyxy = smooth_bbx_xyxy(bbx_xyxy, window=5)
    bbx_xyxy[:, [0, 2]] = bbx_xyxy[:, [0, 2]].clamp(0, W - 1)
    bbx_xyxy[:, [1, 3]] = bbx_xyxy[:, [1, 3]].clamp(0, H - 1)
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
    torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
    Log.info(f"[Preprocess] Detection done. Saved {L} bboxes to {paths.bbx}")


@torch.no_grad()
def run_preprocess(cfg):
    Log.info("[Preprocess] Start")
    video_path = cfg.video_path
    paths = cfg.paths
    L, W, H = get_video_lwh(video_path)

    need_bbx = not Path(paths.bbx).exists()
    need_vitpose = not Path(paths.vitpose).exists()
    need_vit_features = not Path(paths.vit_features).exists()

    # Read video once if any preprocessing stage needs it
    frames = None
    if need_bbx or need_vitpose or need_vit_features:
        frames = read_video_np(video_path)

    if need_bbx:
        detector_name = cfg.get("detector_name", "vitdet")
        if detector_name:
            _run_human_detection(frames, paths, L, W, H, detector_name)
        else:
            raise RuntimeError(f"No bbx file at {paths.bbx} and no detector specified.")

    bbx_xys = torch.load(paths.bbx)["bbx_xys"]

    if need_vitpose:
        vitpose_extractor = VitPoseExtractor(device="cuda:0", pose_type="soma")
        vitpose = vitpose_extractor.extract(frames, bbx_xys, img_ds=1.0, path_type="np", batch_size=32)
        torch.save(vitpose, paths.vitpose)

    if cfg.static_cam:
        L, _, _ = get_video_lwh(video_path)
        eye = torch.eye(4).unsqueeze(0).repeat(L, 1, 1).numpy()
        torch.save(eye, paths.slam)
    elif not Path(paths.slam).exists():
        Log.warning("[Preprocess] VO unavailable. Falling back to static camera.")
        L, _, _ = get_video_lwh(video_path)
        eye = torch.eye(4).unsqueeze(0).repeat(L, 1, 1).numpy()
        torch.save(eye, paths.slam)

    if need_vit_features:
        extractor = SAM3DBExtractor(
            checkpoint_path=cfg.get("sam3d_ckpt_path", None),
            mhr_path=cfg.get("sam3d_mhr_path", None),
            device="cuda:0",
        )
        sam3d_results = extractor.extract_video_features(
            video_path, bbx_xys, render_mhr=False, batch_size=32, frames=frames,
        )
        length = sam3d_results["transls"].shape[0]
        K_fullimg = estimate_K(W, H).repeat(length, 1, 1)
        pred_cam = get_a_pred_cam(sam3d_results["transls"], bbx_xys, K_fullimg)
        vit_features = {
            "pose_tokens": sam3d_results["pose_tokens"],
            "pred_cam": pred_cam,
        }
        torch.save(vit_features, paths.vit_features)

    if frames is not None:
        del frames

    Log.info("[Preprocess] Done")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    vitpose = torch.load(paths.vitpose)
    if isinstance(vitpose, tuple):
        vitpose = vitpose[0]
    bbx_xys = torch.load(paths.bbx)["bbx_xys"].clone()

    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
        t_w2c = torch.zeros(length, 3)
        T_w2c = torch.eye(4).unsqueeze(0).repeat(length, 1, 1)
    else:
        traj = torch.as_tensor(torch.load(paths.slam)).float()
        T_w2c = normalize_T_w2c(traj)
        R_w2c = T_w2c[:, :3, :3]
        t_w2c = T_w2c[:, :3, 3]

    return {
        "meta": [{"vid": Path(cfg.video_path).stem}],
        "length": torch.tensor(length),
        "bbx_xys": bbx_xys,
        "kp2d": vitpose,
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "cam_tvel": compute_cam_tvel(t_w2c),
        "R_w2c": R_w2c,
        "T_w2c": T_w2c,
        "f_imgseq": torch.load(paths.vit_features)["pose_tokens"],
        "noisy_pred_cam": torch.load(paths.vit_features).get("pred_cam", None),
        "has_text": torch.tensor([False]),
        "mask": {
            "valid": torch.ones(length).bool(),
            "has_img_mask": torch.ones(length).bool(),
            "has_2d_mask": torch.ones(length).bool(),
            "has_cam_mask": torch.ones(length).bool(),
            "has_audio_mask": torch.zeros(length).bool(),
            "has_music_mask": torch.zeros(length).bool(),
        },
    }


def resolve_ckpt_path(cfg):
    if cfg.ckpt_path is not None and Path(cfg.ckpt_path).exists():
        return cfg.ckpt_path
    from gem.utils.hf_utils import download_checkpoint
    Log.info("[Checkpoint] Not found locally. Downloading from HuggingFace...")
    return download_checkpoint()


def main():
    args = _parse_args()
    cfg = _build_cfg(args)

    # Use pre-computed bboxes if provided (skips detection)
    if args.bbx is not None:
        import shutil
        bbx_dst = Path(cfg.paths.bbx)
        bbx_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.bbx, bbx_dst)
        Log.info(f"[Preprocess] Using pre-computed bboxes from {args.bbx}")

    run_preprocess(cfg)

    data = load_data_dict(cfg)

    if not Path(cfg.paths.hpe_results).exists():
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        ckpt_path = resolve_ckpt_path(cfg)
        model.load_pretrained_model(ckpt_path)
        model = model.eval().cuda()
        pred = model.predict(data, static_cam=cfg.static_cam, postproc=True)
        torch.save(detach_to_cpu(pred), cfg.paths.hpe_results)

    Log.info(f"[Done] hpe_results saved to {cfg.paths.hpe_results}")


if __name__ == "__main__":
    main()
