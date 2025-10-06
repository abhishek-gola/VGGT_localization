#!/usr/bin/env python3
# Copyright (c) Meta Platforms...
# All rights reserved.

import argparse
import os
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import time
import numpy as np
import torch
import cv2
from PIL import Image as PILImage
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
try:
    # Avoid BLAS oversubscription when using ThreadPoolExecutor
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:
    _threadpool_limits = None

# Local imports
from vggt.dependency.vggsfm_utils import initialize_feature_extractors
import trimesh
from vggt.utils.rotation import mat_to_quat


# ----------------------------- COLMAP Text Parser ---------------------------------- #

class COLMAPReconstruction:
    """Lightweight COLMAP reconstruction parser that reads from text files"""
    
    def __init__(self, sparse_dir: str):
        self.sparse_dir = sparse_dir
        self.cameras = {}  # camera_id -> Camera
        self.images = {}   # image_id -> Image
        self.points3D = {}  # point3d_id -> Point3D
        
        self._load_cameras()
        self._load_images()
        self._load_points3d()
    
    def _load_cameras(self):
        """Parse cameras.txt"""
        cameras_file = os.path.join(self.sparse_dir, "cameras.txt")
        with open(cameras_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(x) for x in parts[4:]]
                
                self.cameras[camera_id] = Camera(camera_id, model, width, height, params)
        print("-----Loaded camera_txt...!!")
    
    def _load_images(self):
        """Parse images.txt"""
        images_file = os.path.join(self.sparse_dir, "images.txt")
        with open(images_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Image line
            parts = line.split()
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            name = ' '.join(parts[9:])  # Handle names with spaces
            
            # Points2D line (next line)
            i += 1
            if i < len(lines):
                points_line = lines[i].strip()
                if points_line and not points_line.startswith('#'):
                    points_parts = points_line.split()
                    points2D = []
                    # Format: x, y, point3D_id (repeating)
                    for j in range(0, len(points_parts), 3):
                        if j + 2 < len(points_parts):
                            x = float(points_parts[j])
                            y = float(points_parts[j + 1])
                            p3d_id = int(points_parts[j + 2])
                            points2D.append(Point2D(x, y, p3d_id))
                else:
                    points2D = []
            else:
                points2D = []
            
            self.images[image_id] = Image(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name, points2D)
            i += 1
        print("-----Loaded images_txt...!!!")

    def _load_points3d(self):
        """Parse points3D.txt"""
        points_file = os.path.join(self.sparse_dir, "points3D.txt")
        if not os.path.exists(points_file):
            return
        
        with open(points_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                point3d_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                
                self.points3D[point3d_id] = Point3D(point3d_id, x, y, z)
        print("-----Loaded points3d...!!!")


class Camera:
    def __init__(self, camera_id: int, model: str, width: int, height: int, params: List[float]):
        self.camera_id = camera_id
        self.model = model
        self.width = width
        self.height = height
        self.params = params
    
    def calibration_matrix(self) -> np.ndarray:
        """Convert camera params to K matrix"""
        if self.model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL']:
            f = self.params[0]
            cx = self.params[1]
            cy = self.params[2]
            return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
        elif self.model in ['PINHOLE', 'RADIAL']:
            fx = self.params[0]
            fy = self.params[1]
            cx = self.params[2]
            cy = self.params[3]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        elif self.model == 'OPENCV':
            fx = self.params[0]
            fy = self.params[1]
            cx = self.params[2]
            cy = self.params[3]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        else:
            # Default: assume first 4 params are fx, fy, cx, cy
            if len(self.params) >= 4:
                return np.array([[self.params[0], 0, self.params[2]], 
                               [0, self.params[1], self.params[3]], 
                               [0, 0, 1]], dtype=np.float32)
            else:
                # Fallback
                f = self.params[0] if len(self.params) > 0 else 500.0
                return np.array([[f, 0, self.width/2], [0, f, self.height/2], [0, 0, 1]], dtype=np.float32)


class Point2D:
    def __init__(self, x: float, y: float, point3D_id: int):
        self.xy = np.array([x, y], dtype=np.float32)
        self.point3D_id = point3D_id


class Point3D:
    def __init__(self, point3d_id: int, x: float, y: float, z: float):
        self.point3d_id = point3d_id
        self.xyz = np.array([x, y, z], dtype=np.float32)


class Image:
    def __init__(self, image_id: int, qw: float, qx: float, qy: float, qz: float, 
                 tx: float, ty: float, tz: float, camera_id: int, name: str, points2D: List[Point2D]):
        self.image_id = image_id
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.camera_id = camera_id
        self.name = name
        self.points2D = points2D
        self._cam_from_world = None
    
    def cam_from_world_matrix(self) -> np.ndarray:
        """Convert quaternion + translation to 3x4 extrinsic matrix [R|t]"""
        if self._cam_from_world is not None:
            return self._cam_from_world
        
        # Quaternion to rotation matrix (qw, qx, qy, qz)
        qw, qx, qy, qz = self.qw, self.qx, self.qy, self.qz
        
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ], dtype=np.float32)
        
        t = np.array([self.tx, self.ty, self.tz], dtype=np.float32).reshape(3, 1)
        self._cam_from_world = np.hstack([R, t])
        return self._cam_from_world


# ----------------------------- Utilities ---------------------------------- #

import torch

def cosine_match_torch(desc_q, desc_d, ratio=0.8, batch=4096):
    """
    Fast cosine matching using GPU + fp16.
    desc_q, desc_d: np.ndarray of shape (Nq, D), (Nd, D)
    Returns: np.ndarray of matches [i_query, i_db]
    """
    if desc_q is None or desc_d is None or len(desc_q) == 0 or len(desc_d) == 0:
        return np.zeros((0,2), dtype=np.int32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # convert to half precision on GPU
    q = torch.from_numpy(desc_q).to(device=device, dtype=torch.float16)
    d = torch.from_numpy(desc_d).to(device=device, dtype=torch.float16)

    # normalize for cosine similarity
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    d = d / (d.norm(dim=1, keepdim=True) + 1e-8)

    # similarity = q @ d^T
    sims = q @ d.T  # (Nq, Nd), runs very fast on GPU
    dist = 1.0 - sims.float()  # back to float32 for safety

    # find best + second-best per query
    vals, idxs = torch.topk(dist, k=2, largest=False, dim=1)
    best_d, second_d = vals[:,0].cpu().numpy(), vals[:,1].cpu().numpy()
    best_idx = idxs[:,0].cpu().numpy()

    # Lowe’s ratio test
    ratio_mask = best_d <= (ratio * second_d)

    # mutual check (optional: do DB->Q pass too, for stricter matches)
    matches = np.where(ratio_mask)[0]
    return np.stack([matches, best_idx[matches]], axis=1).astype(np.int32)


from PIL import Image as PILImage

def _pil_open_rgb(image_path: str) -> PILImage.Image:
    img = PILImage.open(image_path)
    if img.mode == "RGBA":
        background = PILImage.new("RGBA", img.size, (255, 255, 255, 255))
        img = PILImage.alpha_composite(background, img)
    return img.convert("RGB")


def _tensor_from_pil_rgb01(img: PILImage.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _resize_long_side_keep_aspect(img: PILImage.Image, max_long_side: int) -> Tuple[PILImage.Image, float, float]:
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_long_side:
        return img, 1.0, 1.0
    scale = max_long_side / long_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_resized = img.resize((new_w, new_h), PILImage.BILINEAR)
    return img_resized, (w / new_w), (h / new_h)


def _prepare_image_for_extractor(image_path: str, max_long_side: int) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[float, float]]:
    """
    Load PIL image, resize (keeping aspect) so that max(H,W) <= max_long_side,
    return tensor (3,H',W') in [0,1], original size (W,H), and scale-back factors (sx, sy)
    so that kpts_in_resized * [sx,sy] -> kpts_in_original.
    """
    img = _pil_open_rgb(image_path)
    orig_w, orig_h = img.size
    img_r, sx, sy = _resize_long_side_keep_aspect(img, max_long_side)
    tensor = _tensor_from_pil_rgb01(img_r)
    return tensor, (orig_w, orig_h), (sx, sy)


def _extract_features_with_scaling(
    image_tensor_resized: torch.Tensor,
    extractors: Dict[str, torch.nn.Module],
    device: torch.device,
    scale_back: Tuple[float, float],
) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Run extractor on the resized tensor (BCHW), then scale keypoints back to original pixel frame.
    Returns dict[name] = (kpts_xy_in_original_frame, desc or None)
    """
    sx, sy = scale_back
    image_tensor_resized = image_tensor_resized.to(device)
    if image_tensor_resized.dim() == 3:
        image_tensor_resized = image_tensor_resized.unsqueeze(0)  # (1,3,H,W)

    feat_dict: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
    with torch.no_grad():
        for name, extractor in extractors.items():
            out = extractor.extract(image_tensor_resized, invalid_mask=None)
            kps = out.get("keypoints", None)
            desc = out.get("descriptors", None)
            if kps is None:
                continue

            # Expect either (1,N,2) or (N,2)
            if kps.dim() == 3:
                k = kps.squeeze(0).detach().cpu().numpy()
            else:
                k = kps.detach().cpu().numpy()

            # Scale back to ORIGINAL image resolution
            if k.size > 0:
                k[:, 0] *= sx
                k[:, 1] *= sy
            k = k.astype(np.float32)

            d_np = None
            if desc is not None:
                if desc.dim() == 3:
                    d_np = desc.squeeze(0).detach().cpu().numpy().astype(np.float32)
                else:
                    d_np = desc.detach().cpu().numpy().astype(np.float32)

            feat_dict[name] = (k, d_np)

    return feat_dict


# ------------------- DB feature caching (per scene, per extractor) ------------------- #

def _extract_scaled_features_for_path(image_path: str,
                                      extractors: Dict[str, torch.nn.Module],
                                      device: torch.device,
                                      extractor_max_long_side: int) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
    tensor, (_, _), (sx, sy) = _prepare_image_for_extractor(image_path, extractor_max_long_side)
    return _extract_features_with_scaling(tensor, extractors, device, (sx, sy))


def _features_cache_dir(scene_dir: str, extractor_method: str, extractor_max_long_side: int) -> str:
    key = f"{extractor_method}_L{extractor_max_long_side}"
    cache_dir = os.path.join(scene_dir, "db_features", key)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _feature_cache_path(cache_dir: str, image_id: int) -> str:
    return os.path.join(cache_dir, f"{image_id}.npz")


def _save_features_npz(path: str, feats: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]) -> None:
    modalities = list(feats.keys())
    arrays = {}
    arrays["modalities"] = np.array(modalities, dtype=object)
    for name, (kpts, desc) in feats.items():
        arrays[f"{name}_kpts"] = kpts.astype(np.float32, copy=False)
        if desc is None:
            arrays[f"{name}_desc"] = np.zeros((0, 0), dtype=np.float32)
        else:
            arrays[f"{name}_desc"] = desc.astype(np.float32, copy=False)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix="db_feat_", suffix=".npz", dir=os.path.dirname(path))
    os.close(tmp_fd)
    try:
        np.savez(tmp_name, **arrays)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.remove(tmp_name)
        except Exception:
            pass


def _load_features_npz(path: str) -> Optional[Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]]:
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        modalities = list(data["modalities"].tolist())
        out: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        for name in modalities:
            kpts = data[f"{name}_kpts"].astype(np.float32, copy=False)
            desc = data[f"{name}_desc"]
            if desc.size == 0:
                desc_out = None
            else:
                desc_out = desc.astype(np.float32, copy=False)
            out[name] = (kpts, desc_out)
        return out
    except Exception:
        return None


def _ensure_db_features_cache(scene_dir: str,
                              db_id_and_paths: List[Tuple[int, str]],
                              extractors: Dict[str, torch.nn.Module],
                              device: torch.device,
                              extractor_method: str,
                              extractor_max_long_side: int) -> str:
    cache_dir = _features_cache_dir(scene_dir, extractor_method, extractor_max_long_side)
    missing = []
    for image_id, image_path in db_id_and_paths:
        out_path = _feature_cache_path(cache_dir, image_id)
        if not os.path.exists(out_path):
            missing.append((image_id, image_path, out_path))
    if missing:
        print(f"DB feature cache missing for {len(missing)} images; computing and saving...")
        for image_id, image_path, out_path in missing:
            feats = _extract_scaled_features_for_path(image_path, extractors, device, extractor_max_long_side)
            _save_features_npz(out_path, feats)
    else:
        print("DB feature cache complete; using cached features.")
    return cache_dir

def _mutual_nn_match_cosine_ratio_on_distance(
    desc_q: np.ndarray,
    desc_d: np.ndarray,
    ratio: float = 0.8,
) -> np.ndarray:
    """
    Mutual NN using cosine similarity but applying Lowe ratio on the equivalent distances.
    Handles pathological negatives by operating in distance domain.
    Returns (M,2) int indices.
    """
    if desc_q is None or desc_d is None or len(desc_q) == 0 or len(desc_d) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    eps = 1e-8
    q = desc_q / (np.linalg.norm(desc_q, axis=1, keepdims=True) + eps)
    d = desc_d / (np.linalg.norm(desc_d, axis=1, keepdims=True) + eps)
    # Limit BLAS threads per worker to avoid contention
    if _threadpool_limits is not None:
        with _threadpool_limits(limits=1):
            sim = q @ d.T  # (Nq,Nd)
    else:
        sim = q @ d.T  # (Nq,Nd)
    # Convert to distances where smaller is better: cosine distance = 1 - cos_sim in [0,2]
    dist = 1.0 - sim

    # For each query, get best and second-best distances
    idx_sorted = np.argsort(dist, axis=1)  # ascending distance
    best = idx_sorted[:, 0]
    has_second = idx_sorted.shape[1] > 1
    if has_second:
        second = idx_sorted[:, 1]
        best_d = dist[np.arange(dist.shape[0]), best]
        second_d = dist[np.arange(dist.shape[0]), second]
        # classic Lowe: best_d <= ratio * second_d
        ratio_mask = best_d <= (ratio * second_d)
    else:
        ratio_mask = np.ones(dist.shape[0], dtype=bool)

    # Mutual check in distance space (column-wise argmin)
    idx_sorted_d = np.argsort(dist, axis=0)  # for each db, queries ranked
    best_back = idx_sorted_d[0, best]
    mutual = best_back == np.arange(dist.shape[0])

    final_mask = ratio_mask & mutual
    if not np.any(final_mask):
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([np.arange(dist.shape[0])[final_mask], best[final_mask]], axis=1).astype(np.int32)



def _gather_2d3d_from_matches(
    matches: np.ndarray,
    kpts_q: np.ndarray,
    kpts_d: np.ndarray,
    recon_points2d_xy: np.ndarray,
    recon_points3d_xyz: np.ndarray,
    snap_px_thresh: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Snap DB keypoints to nearest registered 2D track and return query 2D ↔ world 3D correspondences.
    All coordinates MUST be in the ORIGINAL db image pixel frame (which we now ensure).
    """
    if matches.size == 0 or kpts_d.size == 0 or recon_points2d_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    pts2d_query = []
    pts3d_world = []

    # Brute-force nearest neighbor (OK at typical sizes)
    for qi, di in matches:
        db_xy = kpts_d[di]
        diff = recon_points2d_xy - db_xy[None, :]
        j = np.argmin((diff * diff).sum(axis=1))
        dist = np.sqrt(((recon_points2d_xy[j] - db_xy) ** 2).sum())
        if dist <= snap_px_thresh:
            pts2d_query.append(kpts_q[qi])
            pts3d_world.append(recon_points3d_xyz[j])

    if not pts2d_query:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    return np.asarray(pts2d_query, dtype=np.float32), np.asarray(pts3d_world, dtype=np.float32)


def _gather_2d3d_from_matches_gpu(
    matches: np.ndarray,
    kpts_q: np.ndarray,
    kpts_d: np.ndarray,
    recon_points2d_xy: np.ndarray,
    recon_points3d_xyz: np.ndarray,
    snap_px_thresh: float = 3.0,
    device: Optional[torch.device] = None,
    batch_size: int = 8192,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized GPU snapping:
    - Finds nearest registered 2D track for ALL matched DB keypoints at once (batched).
    - Applies pixel threshold.
    - Returns (pts2d_query, pts3d_world) as NumPy arrays (drop-in compatible).

    Complexity: O(K * M) but implemented as large batched matmul on GPU
    instead of slow Python loops. K = number of matches, M = registered 2D tracks.
    """
    if matches.size == 0 or kpts_d.size == 0 or recon_points2d_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert inputs (only once)
    # Gather the *matched* db keypoints and corresponding query indices
    qi_np = matches[:, 0].astype(np.int64, copy=False)
    di_np = matches[:, 1].astype(np.int64, copy=False)

    kpts_d_t = torch.from_numpy(kpts_d[di_np]).to(device=device, dtype=torch.float32)     # (K,2)
    kpts_q_t = torch.from_numpy(kpts_q[qi_np]).to(device=device, dtype=torch.float32)     # (K,2)
    recon_xy_t = torch.from_numpy(recon_points2d_xy).to(device=device, dtype=torch.float32)   # (M,2)
    recon_xyz_t = torch.from_numpy(recon_points3d_xyz).to(device=device, dtype=torch.float32) # (M,3)

    K = kpts_d_t.shape[0]
    M = recon_xy_t.shape[0]
    if K == 0 or M == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    # Batched nearest neighbor on GPU using cdist + topk
    # We compute in chunks over K to cap memory.
    nn_idx_chunks = []
    nn_dist_chunks = []

    # squared threshold to avoid sqrt per element
    thr2 = float(snap_px_thresh ** 2)

    for start in range(0, K, batch_size):
        end = min(start + batch_size, K)
        kd = kpts_d_t[start:end]             # (B,2)
        # (B,M,2) -> (B,M) squared distances
        # torch.cdist returns Euclidean; to avoid sqrt cost we can compute squared distances ourselves:
        # dist2 = (a-b)^2 = a^2 + b^2 - 2ab
        # But cdist is quite optimized; we’ll use it and square its output if needed.
        dists = torch.cdist(kd, recon_xy_t, p=2)  # (B,M), float32
        vals, idxs = torch.min(dists, dim=1)      # nearest neighbor per DB keypoint
        nn_idx_chunks.append(idxs)
        nn_dist_chunks.append(vals)

    nn_idx = torch.cat(nn_idx_chunks, dim=0)          # (K,)
    nn_dist = torch.cat(nn_dist_chunks, dim=0)        # (K,)

    # Threshold keep mask
    keep = nn_dist <= snap_px_thresh

    if keep.sum().item() == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    # Select valid correspondences
    sel_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)   # (Kv,)
    # Query 2D points come from the *query* kpts aligned with matches[:,0]
    pts2d_q = kpts_q_t[sel_idx]                                # (Kv,2)
    # 3D world points from nearest registered track
    pts3d_w = recon_xyz_t[nn_idx[sel_idx]]                     # (Kv,3)

    # Move back to CPU/NumPy to keep function signature compatible
    return pts2d_q.detach().cpu().numpy().astype(np.float32), \
           pts3d_w.detach().cpu().numpy().astype(np.float32)



def _median_intrinsics_from_subset(recon: COLMAPReconstruction, image_ids: List[int]) -> np.ndarray:
    """
    Compute median K using only the provided subset of images (e.g., retrieved top-K),
    which is more robust than global-median across the whole scene.
    """
    fx_list = []
    fy_list = []
    cx_list = []
    cy_list = []
    for image_id in image_ids:
        cam = recon.cameras[recon.images[image_id].camera_id]
        K = cam.calibration_matrix()
        fx_list.append(K[0, 0])
        fy_list.append(K[1, 1])
        cx_list.append(K[0, 2])
        cy_list.append(K[1, 2])
    fx = float(np.median(fx_list))
    fy = float(np.median(fy_list))
    cx = float(np.median(cx_list))
    cy = float(np.median(cy_list))
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _project_points3d(E_3x4: np.ndarray, K_3x3: np.ndarray, pts3d_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = pts3d_xyz.astype(np.float64)
    N = P.shape[0]
    Pw = np.hstack([P, np.ones((N, 1), dtype=np.float64)])
    Xc = (E_3x4 @ Pw.T)
    z = Xc[2, :]
    valid = z > 1e-6
    Xn = Xc[:, valid] / Xc[2:3, valid]
    uv_h = (K_3x3 @ Xn)
    uv = uv_h[:2, :].T.astype(np.float32)
    return uv, valid


# ------------------- Lightweight image retrieval (pHash) ------------------- #

def _phash(image_path: str, hash_size: int = 16) -> np.ndarray:
    """
    Tiny perceptual hash (DCT) for retrieval. Returns an L2-normalized vector.
    """
    img = _pil_open_rgb(image_path)
    img = img.convert("L").resize((hash_size*4, hash_size*4), PILImage.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    dct = cv2.dct(arr)
    dct_low = dct[:hash_size, :hash_size].flatten()
    v = dct_low - dct_low.mean()
    n = np.linalg.norm(v) + 1e-8
    return (v / n).astype(np.float32)


def _load_or_build_db_phashes(scene_dir: str, db_paths: List[Tuple[int, str]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cached DB pHashes for a scene, or build and cache them if missing or stale.

    Returns:
        ids: (N,) int64 array of image_ids aligned with hashes rows
        hashes: (N,D) float32 array of L2-normalized pHash vectors
    """
    cache_path = os.path.join(scene_dir, "db_phashes.npz")
    ids_list = [iid for iid, _ in db_paths]
    paths_list = [p for _, p in db_paths]

    def _compute_all() -> Tuple[np.ndarray, np.ndarray]:
        print("Precomputed hashes not found , Computing all ...!!!!")
        vecs: List[np.ndarray] = []
        for p in paths_list:
            vecs.append(_phash(p))
        hashes = np.stack(vecs, axis=0).astype(np.float32)
        ids = np.asarray(ids_list, dtype=np.int64)
        # atomic save
        tmp_fd, tmp_name = tempfile.mkstemp(prefix="db_phashes_", suffix=".npz", dir=scene_dir)
        os.close(tmp_fd)
        try:
            np.savez(tmp_name, ids=ids, hashes=hashes)
            os.replace(tmp_name, cache_path)
        except Exception:
            # Best-effort: remove temp on failure
            try:
                os.remove(tmp_name)
            except Exception:
                pass
        return ids, hashes

    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            ids_cached = data["ids"]
            hashes_cached = data["hashes"]
            # Validate shape and identity/order match
            if ids_cached.shape[0] == len(ids_list) and np.array_equal(ids_cached, np.asarray(ids_list, dtype=np.int64)):
                print("Precomputed hashes found, loading..")
                return ids_cached, hashes_cached.astype(np.float32, copy=False)
        except Exception:
            pass  # fall through to recompute

    return _compute_all()


def _retrieve_top_k_by_phash(scene_dir: str, query_path: str, db_paths: List[Tuple[int, str]], k: int,
                              preloaded_ids_hashes: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> List[int]:
    """
    Cached pHash retrieval. Computes pHash for the query only, compares to cached DB hashes.

    db_paths: list of (image_id, path). Returns a list of selected image_ids.
    """
    if preloaded_ids_hashes is not None:
        ids, hashes = preloaded_ids_hashes
    else:
        ids, hashes = _load_or_build_db_phashes(scene_dir, db_paths)
    qv = _phash(query_path)  # (D,)
    # Cosine similarity because both are L2-normalized
    sims = (hashes @ qv.astype(np.float32))  # (N,)
    order = np.argsort(sims)[::-1]
    if k > 0:
        order = order[:k]
    return [int(ids[i]) for i in order.tolist()]


def localize_query_image(
    scene_dir: str,
    query_image_path: str,
    extractor_method: str = "aliked+sp",
    top_k_db: int = 20,
    min_pnp_inliers: int = 64,
    px_thresh_snap: float = 3.0,
    ransac_reproj_error: float = 8.0,
    extractor_max_long_side: int = 2048,
    prepared_scene: Optional[Dict] = None,
    extractors: Optional[Dict[str, torch.nn.Module]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Localize a query image against an existing COLMAP reconstruction,
    with detailed timing breakdown for feature matching.
    """

    sparse_dir = os.path.join(scene_dir, "sparse")
    if not os.path.isdir(sparse_dir):
        raise FileNotFoundError(f"No sparse reconstruction found at {sparse_dir}")

    # --- 1) Load reconstruction and DB mappings -----------------
    if prepared_scene is None:
        recon = COLMAPReconstruction(sparse_dir)
        images_dir = os.path.join(scene_dir, "images")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"No images/ folder found at {images_dir}")

        id_to_item = {}
        db_id_and_paths = []
        for image_id, img in recon.images.items():
            name = img.name
            image_path = os.path.join(images_dir, name)
            pts2d_xy, pts3d_xyz = [], []
            for p2d in img.points2D:
                if p2d.point3D_id == -1:
                    continue
                pts2d_xy.append(np.array([p2d.xy[0], p2d.xy[1]], dtype=np.float32))
                pts3d_xyz.append(recon.points3D[p2d.point3D_id].xyz.astype(np.float32))
            if pts2d_xy:
                id_to_item[image_id] = (image_path, np.stack(pts2d_xy), np.stack(pts3d_xyz))
            db_id_and_paths.append((image_id, image_path))
        proj_pts3d_pre = None
    else:
        recon = prepared_scene["recon"]
        images_dir = prepared_scene["images_dir"]
        id_to_item = prepared_scene["id_to_item"]
        db_id_and_paths = prepared_scene["db_id_and_paths"]
        proj_pts3d_pre = prepared_scene.get("proj_pts3d", None)

    # --- 2) Query features --------------------------------------
    t_qfeat_start = time.time()
    q_tensor_resized, (qW, qH), (q_sx, q_sy) = _prepare_image_for_extractor(
        query_image_path, extractor_max_long_side
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if extractors is None:
        print("initailisation extractor this shall happen at max once")
        extractors = initialize_feature_extractors(
            extractor_max_long_side, extractor_method=extractor_method, device=device
        )
    feats_q = _extract_features_with_scaling(q_tensor_resized, extractors, device, (q_sx, q_sy))
    t_qfeat = time.time() - t_qfeat_start

    # --- 3) Retrieval -------------------------------------------
    preloaded = None
    if prepared_scene is not None and "db_phashes_mem" in prepared_scene:
        preloaded = prepared_scene["db_phashes_mem"]

    t_retr_start = time.time()
    retrieved_ids = _retrieve_top_k_by_phash(
        scene_dir, query_image_path, db_id_and_paths, top_k_db, preloaded_ids_hashes=preloaded
    )
    t_retrieval = time.time() - t_retr_start
    if top_k_db <= 0 or len(retrieved_ids) == 0:
        retrieved_ids = [iid for iid, _ in db_id_and_paths]

    # --- 4) Decide fallback mode --------------------------------
    use_projection_fallback = (len(id_to_item) == 0)
    proj_pts3d = proj_pts3d_pre
    if use_projection_fallback and proj_pts3d is None:
        pts_list = [recon.points3D[pid].xyz.astype(np.float32) for pid in recon.points3D]
        if not pts_list:
            ply_path = os.path.join(sparse_dir, "points3D.ply")
            if not os.path.exists(ply_path):
                raise RuntimeError("Reconstruction has no 3D points; cannot localize.")
            cloud = trimesh.load(ply_path)
            proj_pts3d = np.asarray(cloud.vertices, dtype=np.float32)
        else:
            proj_pts3d = np.stack(pts_list, axis=0)
        if proj_pts3d.shape[0] > 200000:
            sel = np.random.choice(proj_pts3d.shape[0], 200000, replace=False)
            proj_pts3d = proj_pts3d[sel]

    # --- 5) Matching loop with timers ----------------------------
    t_match_start = time.time()
    time_load_db, time_matching_desc, time_projection, time_snapping = 0.0, 0.0, 0.0, 0.0

    pts2d_all, pts3d_all = [], []

    for image_id in retrieved_ids:
        img = recon.images[image_id]
        image_path = os.path.join(images_dir, img.name)

        # --- Load DB features ---
        t0 = time.time()
        feats_d = None
        if prepared_scene is not None and "db_features_mem" in prepared_scene:
            feats_d = prepared_scene["db_features_mem"].get(image_id)
        if feats_d is None and prepared_scene is not None and "db_feature_cache_dir" in prepared_scene:
            cache_path = _feature_cache_path(prepared_scene["db_feature_cache_dir"], image_id)
            feats_d = _load_features_npz(cache_path)
        if feats_d is None:
            db_tensor_resized, (dbW, dbH), (db_sx, db_sy) = _prepare_image_for_extractor(
                image_path, extractor_max_long_side
            )
            feats_d = _extract_features_with_scaling(db_tensor_resized, extractors, device, (db_sx, db_sy))
        time_load_db += (time.time() - t0)

        common_keys = set(feats_q.keys()).intersection(set(feats_d.keys()))
        if not common_keys:
            continue

        per_image_pts2d, per_image_pts3d = [], []

        have_tracks = (not use_projection_fallback and (image_id in id_to_item))
        if have_tracks:
            recon_xy, recon_xyz = id_to_item[image_id][1], id_to_item[image_id][2]

        for key in common_keys:
            kpts_q_mod, desc_q_mod = feats_q[key]
            kpts_d_mod, desc_d_mod = feats_d[key]
            if kpts_d_mod.shape[0] == 0 or kpts_q_mod.shape[0] == 0:
                continue

            # --- Descriptor matching ---
            t1 = time.time()
            matches = cosine_match_torch(desc_q_mod, desc_d_mod, ratio=0.8)

            time_matching_desc += (time.time() - t1)
            if matches.shape[0] == 0:
                continue

            if have_tracks:
                # --- Snap to COLMAP tracks ---
                t2 = time.time()
                # pts2d_q, pts3d = _gather_2d3d_from_matches(
                #     matches, kpts_q_mod, kpts_d_mod, recon_xy, recon_xyz, snap_px_thresh=px_thresh_snap
                # )
                pts2d_q, pts3d = _gather_2d3d_from_matches_gpu(
                        matches, kpts_q_mod, kpts_d_mod, recon_xy, recon_xyz,
                        snap_px_thresh=px_thresh_snap, device=device, batch_size=8192
                    )
                time_snapping += (time.time() - t2)
            else:
                # --- Projection fallback ---
                t_proj = time.time()
                K_cam = recon.cameras[img.camera_id].calibration_matrix().astype(np.float32)
                E = img.cam_from_world_matrix()
                uv, valid_mask = _project_points3d(E, K_cam, proj_pts3d)
                width, height = recon.cameras[img.camera_id].width, recon.cameras[img.camera_id].height
                in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] < height)
                uv = uv[in_bounds]
                pts3d_proj = proj_pts3d[valid_mask][in_bounds]
                time_projection += (time.time() - t_proj)
                if uv.shape[0] == 0:
                    continue

                # --- Snap ---
                t_snap = time.time()
                pts2d_q, pts3d = _gather_2d3d_from_matches(
                    matches, kpts_q_mod, kpts_d_mod, uv, pts3d_proj, snap_px_thresh=px_thresh_snap
                )
                time_snapping += (time.time() - t_snap)

            if pts2d_q.shape[0] > 0:
                per_image_pts2d.append(pts2d_q)
                per_image_pts3d.append(pts3d)

        if per_image_pts2d:
            pts2d_all.append(np.concatenate(per_image_pts2d, axis=0))
            pts3d_all.append(np.concatenate(per_image_pts3d, axis=0))

    if not pts2d_all:
        raise RuntimeError("No 2D–3D correspondences found.")

    pts2d = np.concatenate(pts2d_all, axis=0)
    pts3d = np.concatenate(pts3d_all, axis=0)
    t_matching = time.time() - t_match_start

    # --- 6) Estimate intrinsics ---------------------------------
    retrieved_cam_ids = [recon.images[iid].camera_id for iid in retrieved_ids]
    cam_count = Counter(retrieved_cam_ids)
    dom_cam_id, _ = cam_count.most_common(1)[0]
    Ks = [recon.cameras[dom_cam_id].calibration_matrix()
          for iid in retrieved_ids if recon.images[iid].camera_id == dom_cam_id]

    if Ks:
        K = np.median(np.stack(Ks, axis=0), axis=0).astype(np.float32)
        ref_w, ref_h = float(recon.cameras[dom_cam_id].width), float(recon.cameras[dom_cam_id].height)
    else:
        K = _median_intrinsics_from_subset(recon, retrieved_ids)
        ref_ws = [float(recon.cameras[recon.images[iid].camera_id].width) for iid in retrieved_ids]
        ref_hs = [float(recon.cameras[recon.images[iid].camera_id].height) for iid in retrieved_ids]
        ref_w = float(np.median(ref_ws)) if ref_ws else float(qW)
        ref_h = float(np.median(ref_hs)) if ref_hs else float(qH)

    sx_k, sy_k = float(qW) / max(ref_w, 1e-8), float(qH) / max(ref_h, 1e-8)
    if abs(sx_k - 1.0) > 1e-6 or abs(sy_k - 1.0) > 1e-6:
        K = K.copy()
        K[0, 0] *= sx_k; K[1, 1] *= sy_k
        K[0, 2] *= sx_k; K[1, 2] *= sy_k

    # --- 7) Solve PnP ------------------------------------------
    t_pnp_start = time.time()
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float32),
        pts2d.astype(np.float32),
        K.astype(np.float32),
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=float(ransac_reproj_error),
        confidence=0.999,
        iterationsCount=10000,
    )
    t_pnp = time.time() - t_pnp_start
    if not success or inliers is None or inliers.shape[0] < min_pnp_inliers:
        raise RuntimeError(f"PnP failed or insufficient inliers: {0 if inliers is None else inliers.shape[0]} found.")

    R, _ = cv2.Rodrigues(rvec)
    extrinsic = np.hstack([R.astype(np.float32), tvec.reshape(3, 1).astype(np.float32)])

    # --- 8) Debug timings --------------------------------------
    debug = {
        "num_corr": np.array([pts2d.shape[0]], dtype=np.int32),
        "num_inliers": np.array([inliers.shape[0]], dtype=np.int32),
        "t_qfeat": np.array([t_qfeat], dtype=np.float32),
        "t_retrieval": np.array([t_retrieval], dtype=np.float32),
        "t_matching_total": np.array([t_matching], dtype=np.float32),
        "t_db_load": np.array([time_load_db], dtype=np.float32),
        "t_desc_match": np.array([time_matching_desc], dtype=np.float32),
        "t_projection": np.array([time_projection], dtype=np.float32),
        "t_snapping": np.array([time_snapping], dtype=np.float32),
        "t_pnp": np.array([t_pnp], dtype=np.float32),
    }

    return extrinsic, K, debug



# New helper to convert extrinsic [R|t] (cam_from_world) to (qw, qx, qy, qz, tx, ty, tz)
def _extrinsic_to_qwxyz_txyz(extrinsic: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    R = extrinsic[:, :3].astype(np.float32)
    t = extrinsic[:, 3].astype(np.float32)
    # mat_to_quat returns (x, y, z, w) with scalar last; convert to scalar-first (w, x, y, z)
    quat_xyzw = mat_to_quat(torch.from_numpy(R)[None, ...]).detach().cpu().numpy()[0]
    qwxyz = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
    return (*qwxyz, float(t[0]), float(t[1]), float(t[2]))


# ------------------------------ Main logic --------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Localize a query image against a VGGT/COLMAP reconstruction (without pycolmap).")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory with images/ and sparse/ from COLMAP")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query_image", type=str, help="Path to a single query image to localize")
    group.add_argument("--query_dir", type=str, help="Path to a directory of query images to localize")
    parser.add_argument("--output_txt", type=str, default="poses.txt", help="Output txt file to write poses as: name qw qx qy qz tx ty tz")
    parser.add_argument("--extractor", type=str, default="aliked+sp", help="Feature extractor combo, e.g., 'aliked+sp' or 'aliked+sp+sift'")
    parser.add_argument("--top_k_db", type=int, default=20, help="Number of database images to match (retrieved by pHash)")
    parser.add_argument("--px_thresh_snap", type=float, default=3.0, help="Max px distance to snap db keypoint to recon 2D point")
    parser.add_argument("--ransac_reproj_error", type=float, default=8.0, help="RANSAC reprojection error in pixels")
    parser.add_argument("--min_pnp_inliers", type=int, default=64, help="Minimum inliers to accept PnP solution")
    parser.add_argument("--extractor_max_long_side", type=int, default=2048, help="Resizes inputs for extractor; kpts are scaled back")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of parallel threads for directory mode")
    args = parser.parse_args()

    if args.query_dir is None:
        # Single-image mode
        extrinsic, K, dbg = localize_query_image(
            scene_dir=args.scene_dir,
            query_image_path=args.query_image,
            extractor_method=args.extractor,
            top_k_db=args.top_k_db,
            px_thresh_snap=args.px_thresh_snap,
            ransac_reproj_error=args.ransac_reproj_error,
            min_pnp_inliers=args.min_pnp_inliers,
            extractor_max_long_side=args.extractor_max_long_side,
        )

        print("Localization successful.")
        print("Estimated extrinsic (cam_from_world [R|t]):")
        print(extrinsic)
        print("Estimated intrinsics K:")
        print(K)
        print(f"Num correspondences: {int(dbg['num_corr'][0])}, inliers: {int(dbg['num_inliers'][0])}")

        qwxyz_t = _extrinsic_to_qwxyz_txyz(extrinsic)
        with open(args.output_txt, "a") as f:
            f.write(f"{os.path.basename(args.query_image)} {qwxyz_t[0]:.8f} {qwxyz_t[1]:.8f} {qwxyz_t[2]:.8f} {qwxyz_t[3]:.8f} {qwxyz_t[4]:.6f} {qwxyz_t[5]:.6f} {qwxyz_t[6]:.6f}\n")
        print(f"Saved pose to {args.output_txt}")
        return

    # Directory mode
    if not os.path.isdir(args.query_dir):
        raise FileNotFoundError(f"Query directory not found: {args.query_dir}")

    entries = sorted([e for e in os.listdir(args.query_dir)])
    entries = entries[:722]
    entries = entries[::-1]
    print(f"length of entries : {len(entries)}")
    print("entries: ", entries[:5])

    if len(entries) == 0:
        raise RuntimeError(f"No images found in directory: {args.query_dir}")

    # Prepare shared scene and extractors once
    print("Loading Reconstruction...")
    recon = COLMAPReconstruction(os.path.join(args.scene_dir, "sparse"))
    print("Reconstruction Loaded...")
    images_dir = os.path.join(args.scene_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"No images/ folder found at {images_dir}")
    
    id_to_item = {}
    db_id_and_paths = []
    
    

    for image_id, img in recon.images.items():
        name = img.name
        image_path = os.path.join(images_dir, name)
        points2d_xy = []
        points3d_xyz = []
        for p2d in img.points2D:
            pid = p2d.point3D_id
            if pid == -1:
                continue
            xy = p2d.xy
            points2d_xy.append(np.array([xy[0], xy[1]], dtype=np.float32))
            p3d = recon.points3D[pid]
            points3d_xyz.append(p3d.xyz.astype(np.float32))
        if len(points2d_xy) > 0:
            id_to_item[image_id] = (image_path, np.stack(points2d_xy, axis=0), np.stack(points3d_xyz, axis=0))
        db_id_and_paths.append((image_id, image_path))
   
    prepared_scene = {
        "recon": recon,
        "images_dir": images_dir,
        "id_to_item": id_to_item,
        "db_id_and_paths": db_id_and_paths,
    }


    shared_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Shared Device : {shared_device}......")
    print("Initializing feature extractors....")
    shared_extractors = initialize_feature_extractors(
        args.extractor_max_long_side, extractor_method=args.extractor, device=shared_device
    )
    print("Initialisation complete....")

    # Preload DB pHashes into memory once
    db_ids_cached, db_hashes_cached = _load_or_build_db_phashes(args.scene_dir, db_id_and_paths)
    prepared_scene["db_phashes_mem"] = (db_ids_cached, db_hashes_cached)

    # Ensure DB feature cache exists (per scene, per extractor config)
    cache_dir = _ensure_db_features_cache(
        args.scene_dir,
        db_id_and_paths,
        shared_extractors,
        shared_device,
        args.extractor,
        args.extractor_max_long_side,
    )
    prepared_scene["db_feature_cache_dir"] = cache_dir

    # Preload DB features into memory once to avoid per-thread npz loads
    db_features_mem: Dict[int, Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]] = {}
    for image_id, _ in db_id_and_paths:
        cache_path = _feature_cache_path(cache_dir, image_id)
        feats = _load_features_npz(cache_path)
        if feats is None:
            continue
        db_features_mem[image_id] = feats
    prepared_scene["db_features_mem"] = db_features_mem

    failures = []
    with open(args.output_txt, "w") as f:
        def _process(name: str):
            import threading
            qpath = os.path.join(args.query_dir, name)
            start_pred = time.time()
            if torch.cuda.is_available():
                 torch.cuda.reset_peak_memory_stats(shared_device)

            extrinsic, K, dbg = localize_query_image(
                scene_dir=args.scene_dir,
                query_image_path=qpath,
                extractor_method=args.extractor,
                top_k_db=args.top_k_db,
                px_thresh_snap=args.px_thresh_snap,
                ransac_reproj_error=args.ransac_reproj_error,
                min_pnp_inliers=args.min_pnp_inliers,
                extractor_max_long_side=args.extractor_max_long_side,
                prepared_scene=prepared_scene,
                extractors=shared_extractors,
                device=shared_device,
            )
            elapsed_pred = time.time() - start_pred
            worker_name = threading.current_thread().name
            qwxyz_t = _extrinsic_to_qwxyz_txyz(extrinsic)
            line = (
                f"{name} {qwxyz_t[0]:.8f} {qwxyz_t[1]:.8f} {qwxyz_t[2]:.8f} {qwxyz_t[3]:.8f} "
                f"{qwxyz_t[4]:.6f} {qwxyz_t[5]:.6f} {qwxyz_t[6]:.6f}\n"
            )

            # timings
            num_corr = int(dbg.get("num_corr", np.array([0], dtype=np.int32))[0])
            num_inl = int(dbg.get("num_inliers", np.array([0], dtype=np.int32))[0])
            t_qfeat = float(dbg.get("t_qfeat", [0])[0])
            t_retrieval = float(dbg.get("t_retrieval", [0])[0])
            t_matching_total = float(dbg.get("t_matching_total", [0])[0])
            t_db_load = float(dbg.get("t_db_load", [0])[0])
            t_desc_match = float(dbg.get("t_desc_match", [0])[0])
            t_projection = float(dbg.get("t_projection", [0])[0])
            t_snapping = float(dbg.get("t_snapping", [0])[0])
            t_pnp = float(dbg.get("t_pnp", [0])[0])

            # GPU usage per image
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(shared_device) / (1024**2)   # MB
                gpu_mem_max = torch.cuda.max_memory_allocated(shared_device) / (1024**2)
            else:
                gpu_mem, gpu_mem_max = 0.0, 0.0

            return (
                name, line, num_corr, num_inl, elapsed_pred, worker_name,
                t_qfeat, t_retrieval, t_matching_total, t_db_load, t_desc_match,
                t_projection, t_snapping, t_pnp, gpu_mem, gpu_mem_max
            )


        start_time = time.time()
        print("Inference started...!!!!")
        with ThreadPoolExecutor(max_workers=max(1, int(args.num_workers))) as executor:
            futures = {executor.submit(_process, name): name for name in entries}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    (
                        name, line, num_corr, num_inl, elapsed_pred, worker_name,
                        t_qfeat, t_retrieval, t_matching_total, t_db_load, t_desc_match,
                        t_projection, t_snapping, t_pnp, gpu_mem, gpu_mem_max
                    ) = fut.result()
                    f.write(line)
                    f.flush()
                    # print status per image
                    # print(
                    #     f"OK {name}: corr={num_corr}, inliers={num_inl}, "
                    #     f"time={elapsed_pred:.3f}s, worker={worker_name}, "
                    #     f"f={t_qfeat:.3f}s r={t_retrieval:.3f}s m_total={t_matching_total:.3f}s "
                    #     f"[db_load={t_db_load:.3f}s desc_match={t_desc_match:.3f}s "
                    #     f"proj={t_projection:.3f}s snap={t_snapping:.3f}s] p={t_pnp:.3f}s"
                    # )
                    print(
                        f"OK {name}: corr={num_corr}, inliers={num_inl}, "
                        f"time={elapsed_pred:.3f}s, "
                        f"f={t_qfeat:.3f}s r={t_retrieval:.3f}s m_total={t_matching_total:.3f}s "
                        f"[db_load={t_db_load:.3f}s desc_match={t_desc_match:.3f}s "
                        f"proj={t_projection:.3f}s snap={t_snapping:.3f}s] p={t_pnp:.3f}s "
                        f"| GPU={gpu_mem:.1f}MB (peak {gpu_mem_max:.1f}MB)"
                    )
                except Exception as e:
                    failures.append((name, str(e)))
                    print(f"FAIL {name}: {e}", flush=True)
        end_time = time.time()

        duration = end_time - start_time
        print(f"Total inference time in seconds: {duration:.3f}")
        print(f"Average time per frame: {duration/len(entries):.3f}")

    print(f"Processed {len(entries) - len(failures)} images, {len(failures)} failed. Saved poses to {args.output_txt}")

    # if failures:
    #     print("Failures (image -> reason):")
    #     for name, reason in failures[:20]:
    #         print(f"  {name} -> {reason}")
    #     if len(failures) > 20:
    #         print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    with torch.no_grad():
        main()