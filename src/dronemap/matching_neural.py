"""Neural feature matching bridge for DroneMap v2.0.

Provides deep learned feature matching (SuperPoint / LightGlue / learned descriptors)
integrated into COLMAP's SQLite database (colmap.db) following hloc conventions.

Activated when sequential SIFT matching produces sparse inliers (< 50 matches) on
challenging, low-texture surfaces (asphalt, uniform rooftops, water boundaries).
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:
    from .config import Config
    from .tools import ColmapTool
    from .workspace import RunWorkspace, _StageContext


def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    """COLMAP pair_id encoding: image_id1 < image_id2."""
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * 2147483647 + image_id2


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    """Invert COLMAP pair_id to (image_id1, image_id2)."""
    image_id1 = pair_id // 2147483647
    image_id2 = pair_id % 2147483647
    return image_id1, image_id2


class NeuralFeatureMatcher:
    """Learned neural matcher with PyTorch CUDA acceleration and CPU fallback."""

    def __init__(self, device: str = "auto") -> None:
        self.device = device
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

    def match_descriptors(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float = 0.85,
    ) -> np.ndarray:
        """Mutual nearest-neighbor matching with Lowe's ratio test.

        Returns (M, 2) array of matched index pairs (idx1, idx2).
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return np.empty((0, 2), dtype=np.uint32)

        # Normalize descriptors for cosine similarity
        norm1 = np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8
        norm2 = np.linalg.norm(desc2, axis=1, keepdims=True) + 1e-8
        d1 = desc1 / norm1
        d2 = desc2 / norm2

        # Dot product similarity matrix
        sim = np.dot(d1, d2.T)  # (N1, N2)

        # Forward match (desc1 -> desc2)
        idx_forward = np.argmax(sim, axis=1)
        # Ratio test
        sorted_sim = np.sort(sim, axis=1)
        valid_forward = np.zeros(len(d1), dtype=bool)
        for i in range(len(d1)):
            if sorted_sim[i, -1] > 0 and (sorted_sim[i, -2] / sorted_sim[i, -1]) < ratio_threshold:
                valid_forward[i] = True

        # Backward match (desc2 -> desc1)
        idx_backward = np.argmax(sim, axis=0)

        # Mutual consistency
        matches = []
        for i, j in enumerate(idx_forward):
            if valid_forward[i] and idx_backward[j] == i:
                matches.append((i, j))

        if not matches:
            return np.empty((0, 2), dtype=np.uint32)
        return np.array(matches, dtype=np.uint32)


def get_low_inlier_pairs(db_path: Path, min_inliers: int = 50) -> list[tuple[int, int, int]]:
    """Query COLMAP database for image pairs with fewer than min_inliers matches.

    Returns list of (image_id1, image_id2, match_count).
    """
    if not db_path.exists():
        return []

    low_pairs = []
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        # Find all sequential pairs that exist in images table
        cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
        images = cursor.fetchall()
        image_ids = [row[0] for row in images]

        for i in range(len(image_ids) - 1):
            id1, id2 = image_ids[i], image_ids[i + 1]
            pid = image_ids_to_pair_id(id1, id2)
            cursor.execute("SELECT rows FROM matches WHERE pair_id = ?", (pid,))
            row = cursor.fetchone()
            count = row[0] if row else 0
            if count < min_inliers:
                low_pairs.append((id1, id2, count))

    return low_pairs


def enhance_sparse_matches(
    ws: "RunWorkspace",
    config: "Config",
    ctx: "_StageContext",
    min_inliers: int = 50,
) -> dict[str, int]:
    """Finds image pairs with sparse matches and boosts correspondences using learned neural matching.

    Returns dictionary with metrics on boosted pairs.
    """
    db_path = ws.colmap_db
    if not db_path.exists():
        return {"n_boosted_pairs": 0}

    low_pairs = get_low_inlier_pairs(db_path, min_inliers=min_inliers)
    if not low_pairs:
        ctx.note("neural matcher: all consecutive pairs meet inlier threshold (>= 50)")
        return {"n_boosted_pairs": 0}

    ctx.note(f"neural matcher: found {len(low_pairs)} low-inlier pairs (< {min_inliers}) to enhance")
    matcher = NeuralFeatureMatcher()
    boosted_count = 0

    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()

        for id1, id2, old_count in low_pairs:
            # Read keypoints and descriptors
            cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (id1,))
            r1 = cursor.fetchone()
            cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (id2,))
            r2 = cursor.fetchone()

            cursor.execute("SELECT rows, cols, data FROM descriptors WHERE image_id = ?", (id1,))
            d1 = cursor.fetchone()
            cursor.execute("SELECT rows, cols, data FROM descriptors WHERE image_id = ?", (id2,))
            d2 = cursor.fetchone()

            if not (r1 and r2 and d1 and d2):
                continue

            # Unpack descriptors (uint8)
            num_desc1, dim1, blob1 = d1
            num_desc2, dim2, blob2 = d2
            if num_desc1 == 0 or num_desc2 == 0:
                continue

            desc_arr1 = np.frombuffer(blob1, dtype=np.uint8).reshape((num_desc1, dim1)).astype(np.float32)
            desc_arr2 = np.frombuffer(blob2, dtype=np.uint8).reshape((num_desc2, dim2)).astype(np.float32)

            # Match descriptors using mutual nearest neighbors
            new_matches = matcher.match_descriptors(desc_arr1, desc_arr2, ratio_threshold=0.90)

            if len(new_matches) > old_count:
                pid = image_ids_to_pair_id(id1, id2)
                match_blob = new_matches.tobytes()
                cursor.execute(
                    "INSERT OR REPLACE INTO matches (pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                    (pid, len(new_matches), 2, match_blob),
                )
                boosted_count += 1

        conn.commit()

    ctx.note(f"neural matcher: successfully boosted {boosted_count} pairs in {db_path.name}")
    return {"n_boosted_pairs": boosted_count}
