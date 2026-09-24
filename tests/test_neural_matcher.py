"""Unit tests for matching_neural.py (LightGlue & Neural Feature Matching Bridge)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from dronemap.config import Config
from dronemap.matching_neural import (
    NeuralFeatureMatcher,
    enhance_sparse_matches,
    get_low_inlier_pairs,
    image_ids_to_pair_id,
    pair_id_to_image_ids,
)


def test_pair_id_reversibility():
    """Verify COLMAP pair_id encoding/decoding is consistent."""
    id1, id2 = 14, 29
    pid = image_ids_to_pair_id(id1, id2)
    rec1, rec2 = pair_id_to_image_ids(pid)
    assert (rec1, rec2) == (id1, id2)

    # Order invariance
    pid_rev = image_ids_to_pair_id(id2, id1)
    assert pid == pid_rev


def test_neural_matcher_mutual_nearest_neighbors():
    """Verify NeuralFeatureMatcher finds mutual consistent matches."""
    matcher = NeuralFeatureMatcher(device="cpu")

    # Generate 10 identical descriptors + 5 disjoint descriptors
    np.random.seed(42)
    shared = np.random.randn(10, 128).astype(np.float32)
    unique1 = np.random.randn(5, 128).astype(np.float32) * 5.0
    unique2 = np.random.randn(5, 128).astype(np.float32) * 5.0

    desc1 = np.vstack([shared, unique1])
    desc2 = np.vstack([shared, unique2])

    matches = matcher.match_descriptors(desc1, desc2, ratio_threshold=0.99)
    assert len(matches) >= 8
    # Shared elements (0..9) should match (i, i)
    correct_matches = sum(1 for m in matches if m[0] == m[1] and m[0] < 10)
    assert correct_matches >= 8


def test_get_low_inlier_pairs_and_enhancement(tmp_path: Path):
    """Test database inspection and match enhancement on synthetic SQLite db."""
    db_path = tmp_path / "colmap.db"
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
        cursor.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
        cursor.execute("CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
        cursor.execute("CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")

        # Insert 3 images
        cursor.execute("INSERT INTO images VALUES (1, 'frame_000001.jpg')")
        cursor.execute("INSERT INTO images VALUES (2, 'frame_000002.jpg')")
        cursor.execute("INSERT INTO images VALUES (3, 'frame_000003.jpg')")

        # Insert descriptors (10 features each, 128-d uint8)
        np.random.seed(123)
        shared = np.random.randint(0, 255, size=(10, 128), dtype=np.uint8)
        kpts_blob = np.zeros((10, 2), dtype=np.float32).tobytes()

        for img_id in (1, 2, 3):
            cursor.execute("INSERT INTO keypoints VALUES (?, 10, 2, ?)", (img_id, kpts_blob))
            cursor.execute("INSERT INTO descriptors VALUES (?, 10, 128, ?)", (img_id, shared.tobytes()))

        # Pair (1, 2) has only 4 matches (low inlier)
        pid12 = image_ids_to_pair_id(1, 2)
        cursor.execute("INSERT INTO matches VALUES (?, 4, 2, ?)", (pid12, b""))

        # Pair (2, 3) has 80 matches (healthy)
        pid23 = image_ids_to_pair_id(2, 3)
        cursor.execute("INSERT INTO matches VALUES (?, 80, 2, ?)", (pid23, b""))
        conn.commit()

    low_pairs = get_low_inlier_pairs(db_path, min_inliers=50)
    assert len(low_pairs) == 1
    assert low_pairs[0][0] == 1 and low_pairs[0][1] == 2 and low_pairs[0][2] == 4

    # Mock workspace and context to test enhance_sparse_matches
    ws = MagicMock()
    ws.colmap_db = db_path
    cfg = Config()
    ctx = MagicMock()

    res = enhance_sparse_matches(ws, cfg, ctx, min_inliers=50)
    assert res["n_boosted_pairs"] == 1
