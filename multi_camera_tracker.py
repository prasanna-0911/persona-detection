"""
Multi-Camera Person Tracking System

This module enables tracking persons across multiple cameras while
maintaining the SAME ID for each person regardless of which camera
they appear in.

Key Features:
- Unified video file and RTSP stream support
- Global person gallery for cross-camera Re-ID
- Same person ID maintained across all cameras
- Automatic ID assignment based on appearance similarity

Architecture:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Camera 1   │     │  Camera 2   │     │  Camera 3   │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Global Gallery    │
                    │  (Person Features)  │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Same ID Assigned  │
                    │   Across Cameras    │
                    └─────────────────────┘

Usage:
    # Single source (video or RTSP)
    tracker = MultiCameraTracker(reid_model_path)
    tracker.process_source("video.mp4", output_path="output.mp4")
    tracker.process_source("rtsp://user:pass@ip:554/stream", output_path="live.mp4")
    
    # Multiple cameras with global ID
    tracker.process_multi_camera([
        {"name": "Entrance", "source": "rtsp://..."},
        {"name": "Lobby", "source": "rtsp://..."},
        {"name": "Exit", "source": "rtsp://..."},
    ], output_dir="outputs/")
"""

import os

# Fix OpenMP error: "Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import cv2
import time
import threading
import queue
import numpy as np
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
import random
from typing import List, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Bug #1 fix: Use dynamic path resolution instead of hardcoded Colab path.
# The old code used '/content/drive/MyDrive/persona_detection_final' which
# only works in Google Colab and crashes on Windows/Linux local environments.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'phase2_reid'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'phase3_tracking'))

# OSNet backend (Option B — industry-standard pre-trained Re-ID)
#
# Two torchreid versions exist with DIFFERENT package structures:
#   Source install:  torchreid.utils.FeatureExtractor          (correct)
#   PyPI install:    torchreid.reid.utils.FeatureExtractor      (fallback)
#
# Recommended install (always works):
#   pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
_TORCHREID_AVAILABLE = False
_TORCHREID_IMPORT_ERROR = None

# Try 1: official source install path (KaiyangZhou/deep-person-reid)
try:
    from torchreid.utils import FeatureExtractor as OsNetExtractor
    _TORCHREID_AVAILABLE = True
except Exception as _e1:
    # Try 2: PyPI version path (torchreid on PyPI has different namespace)
    try:
        from torchreid.reid.utils import FeatureExtractor as OsNetExtractor
        _TORCHREID_AVAILABLE = True
    except Exception as _e2:
        _TORCHREID_AVAILABLE = False
        _TORCHREID_IMPORT_ERROR = (
            f"Path 1 (torchreid.utils): {type(_e1).__name__}: {_e1}\n"
            f"Path 2 (torchreid.reid.utils): {type(_e2).__name__}: {_e2}"
        )

# Custom model backend (backward compatible with project-trained weights)
from models.reid_net import ReIDNetwork
# DeepSORTTracker removed; using Ultralytics BoT-SORT natively
from ultralytics import YOLO


@dataclass
class PersonRecord:
    """
    Record of a tracked person in the global gallery.
    
    Attributes:
        global_id: Unique ID across all cameras
        features: List of Re-ID feature vectors
        last_seen: Timestamp of last detection
        cameras_seen: Set of camera names where person was seen
        total_detections: Total number of detections
        stable_count: Number of full-body (non-edge) embeddings stored.
            Used to identify 'provisional' entries that were built from
            partial-view crops and therefore deserve lenient match thresholds.
    """
    global_id: int
    features: List[np.ndarray]
    last_seen: float
    cameras_seen: set
    total_detections: int = 0
    stable_count: int = 0   # counts non-edge (full-body) embeddings only
    stable_features: List[np.ndarray] = field(default_factory=list)
    # ^ Full-body embeddings ONLY, spanning the entire video timeline (max 30).
    # Kept separate from `features` (which includes edge/partial crops) so that
    # _compute_best_similarity can search the person's complete appearance history
    # rather than just the most recent 15 mixed embeddings.
    # Solves the temporal-mismatch problem: when Camera_1 has processed 1200 frames
    # before Camera_2 starts, Camera_1's gallery still retains the person's
    # frame-0 full-body embedding here, allowing Camera_2 to match it.
    last_bbox: Optional[np.ndarray] = None
    # ^ Last known bounding box [x1,y1,x2,y2] for spatial fallback matching.
    # When embedding similarity fails (e.g. person turns 90° and OSNet changes),
    # high bbox IoU with a gallery entry's last position is still a strong signal
    # of the same person.  Updated every frame regardless of partial/edge status.
    departure_features: List[np.ndarray] = field(default_factory=list)
    # ^ Last N full-body embeddings captured just before the person's BotSort
    # track was last dropped (i.e. the "leaving" frames).  Stored per-departure
    # and capped at 5 embeddings.  These transitional views (e.g. left-front
    # when walking toward a door) are closer in OSNet cosine space to the
    # re-entry view (back-right entering) than the stable sitting embeddings
    # are, boosting best-match similarity from 0.38-0.52 to 0.50-0.65 for the
    # same person — enough to clear the re_entry_threshold of 0.45.
    # Replaced (not appended) on each new departure so the pool always
    # reflects the most recent leaving direction.

    @property
    def is_provisional(self) -> bool:
        """True while this entry has fewer than 5 full-body embeddings.

        Provisional entries were built from partial-view crops (edge detections,
        head-only, arm-only, etc.) whose OSNet embeddings are noisy.  Requiring a
        strict 0.70 threshold when matching AGAINST such an entry would cause
        missed matches — the stored embeddings themselves are weak, so the gallery
        similarity is inherently lower even for the correct person.  The caller
        should lower the effective match threshold for provisional entries.
        """
        return self.stable_count < 5

    def get_average_feature(self) -> np.ndarray:
        """Get average feature vector for matching."""
        if not self.features:
            return None
        return np.mean(self.features, axis=0)

    def add_feature(self, feature: np.ndarray, camera_name: str,
                    is_stable: bool = True):
        """Add a new feature observation.

        Args:
            feature: ReID embedding vector.
            camera_name: Which camera detected this person.
            is_stable: True if from a full-body (non-edge) crop.
                       False for partial/edge crops whose embeddings are noisy.
                       Only stable observations increment stable_count.
        """
        self.features.append(feature)
        # Keep only last 50 features to bound memory
        if len(self.features) > 50:
            self.features = self.features[-50:]
        self.last_seen = time.time()
        self.cameras_seen.add(camera_name)
        self.total_detections += 1
        if is_stable:
            self.stable_count += 1  # track entry quality
            # ── Anchored Reservoir Sampling ──────────────────────────────
            # User insight: "why not collect embeddings from when the person
            # FIRST appeared?" — Absolutely right.  Early embeddings are the
            # most valuable because:
            #   1. Camera_B (starting late) queries with EARLY-video embeddings
            #      and needs them preserved in Camera_A's gallery.
            #   2. A person’s first appearance is often their cleanest full-body
            #      view (just entered the scene, not yet occluded).
            #
            # Two-zone strategy:
            #   ANCHOR zone  (slots 0–9, first 10):  NEVER replaced.  Locked in
            #     from the person’s first 10 full-body frames.  Guarantees that
            #     Camera_B / sequential processing always finds a temporally-early
            #     embedding to match against.
            #   RESERVOIR zone (slots 10–29, next 20): Vitter Algorithm R.
            #     Each new embedding replaces a random slot in this zone with
            #     probability 20/(stable_count−10), giving a uniform sample of
            #     mid/late-video appearances on top of the fixed early anchor.
            ANCHOR_SIZE    = 10   # first N embeddings — locked forever
            RESERVOIR_SIZE = 20   # uniform sample of all later embeddings
            TOTAL_CAP      = ANCHOR_SIZE + RESERVOIR_SIZE  # 30 total

            if len(self.stable_features) < TOTAL_CAP:
                # Phase 1: fill all 30 slots sequentially
                self.stable_features.append(feature)
            elif self.stable_count > ANCHOR_SIZE:
                # Phase 2: reservoir-replace only in the RESERVOIR zone (10–29)
                reservoir_observations = self.stable_count - ANCHOR_SIZE
                idx = random.randint(0, reservoir_observations - 1)
                if idx < RESERVOIR_SIZE:
                    self.stable_features[ANCHOR_SIZE + idx] = feature

    def add_departure_feature(self, feature: np.ndarray):
        """Record a 'leaving' embedding captured just before departing camera view.

        Departure embeddings are transitional views (e.g., person turning to
        walk toward the door). They score 0.50-0.65 in cosine space against
        re-entry views (back-right entering) vs 0.38-0.52 for stable sitting
        embeddings, making them essential for successful re-entry matching.

        Max 5 departure embeddings kept (last 5 frames before disappearance).
        They are REPLACED on each new departure (not accumulated across trips),
        so the search pool always reflects the most recent leaving direction.
        """
        self.departure_features.append(feature)
        if len(self.departure_features) > 5:
            self.departure_features = self.departure_features[-5:]  # keep newest 5


class GlobalPersonGallery:
    """
    Global gallery for cross-camera person Re-ID.
    
    Maintains a database of all known persons and their features.
    When a new person is detected, checks if they match any existing
    person in the gallery.
    """
    
    def __init__(self, similarity_threshold: float = 0.80,
                 edge_match_threshold: float = 0.50,
                 re_entry_threshold: float = 0.45,
                 re_entry_window_secs: float = 120.0,
                 max_gallery_size: int = 1000):
        """
        Initialize global gallery.

        Args:
            similarity_threshold: Minimum cosine similarity to match a
                FULL-BODY detection against a stable gallery entry (0–1).
                Strict threshold (0.70) prevents false positives between
                similarly-dressed different people in crowded scenes.
            edge_match_threshold: Minimum cosine similarity to match a
                PARTIAL-BODY (edge/occluded) detection, or any detection
                against a PROVISIONAL gallery entry (stable_count < 5).
                Must be lower (0.45–0.55) because OSNet produces 0.42–0.65
                similarity for the SAME person when one side is partial-crop.
            max_gallery_size: Maximum number of persons to track
        """
        self.persons: Dict[int, PersonRecord] = {}
        self.next_global_id = 1
        self.similarity_threshold = similarity_threshold
        self.edge_match_threshold = edge_match_threshold
        self.re_entry_threshold = re_entry_threshold
        self.re_entry_window_secs = re_entry_window_secs
        self.max_gallery_size = max_gallery_size

        # Local to global ID mapping per camera
        # camera_name -> {local_track_id -> global_id}
        self.local_to_global: Dict[str, Dict[int, int]] = defaultdict(dict)

        # Lock for thread safety
        self.lock = threading.Lock()
    
    def find_or_create_global_id(
        self,
        local_track_id: int,
        feature: np.ndarray,
        camera_name: str,
        active_local_ids: set,
        update_gallery: bool = True,
        is_edge: bool = False,
        bbox: Optional[np.ndarray] = None
    ) -> int:
        """
        Find existing global ID or create new one.

        Args:
            local_track_id: Track ID from local camera tracker
            feature: Re-ID feature vector
            camera_name: Name of the camera
            active_local_ids: Set of all local track IDs actively detected in this frame
            update_gallery: If False, do NOT add this embedding to existing gallery
                entries. Used for edge/partial detections whose noisy features
                would corrupt gallery entries. New entries are still created
                (so the track is visible), but existing entries are not polluted.
            is_edge: True if this detection is at the frame border OR has a low
                aspect ratio (partial body). When True, a lenient
                edge_match_threshold is used for gallery search instead of
                similarity_threshold, because OSNet partial-crop embeddings
                score only 0.42–0.65 for the SAME person vs a full-body entry.
            bbox: Bounding box [x1,y1,x2,y2] of this detection (float32).
                Stored as last_bbox on matched/created entries and used as a
                spatial fallback when embedding similarity fails (IoU ≥ 0.25).

        Returns:
            Global ID for this person
        """
        with self.lock:
            # Prevent 2 distinct local persons from mapping to the same Global ID
            # by identifying which Global IDs are already "taken" in this exact frame.
            active_global_ids_in_camera = {
                self.local_to_global[camera_name][tid]
                for tid in active_local_ids
                if tid in self.local_to_global[camera_name]
            }

            # Check if we already have a mapping for this local track
            if local_track_id in self.local_to_global[camera_name]:
                global_id = self.local_to_global[camera_name][local_track_id]
                # Update gallery, marking whether this is a stable (full-body) review
                if global_id in self.persons and update_gallery:
                    self.persons[global_id].add_feature(
                        feature, camera_name, is_stable=not is_edge
                    )
                if global_id in self.persons and bbox is not None:
                    self.persons[global_id].last_bbox = bbox  # always track position
                return global_id

            # ── Dual threshold selection ────────────────────────────────────────
            # Partial/edge detections → lenient edge_match_threshold (e.g. 0.50)
            #   OSNet produces lower cosine similarity for head-only / back-only /
            #   arm-only crops versus a full-body gallery entry for the SAME person.
            #   Using 0.70 here would cause these detections to always miss, creating
            #   new gallery IDs on every partial-view frame.
            # Full-body detections → strict similarity_threshold (e.g. 0.70)
            #   Prevents similarly-dressed DIFFERENT people from merging into one ID.
            query_thresh = self.edge_match_threshold if is_edge else self.similarity_threshold

            # ── New track: search gallery using best-of-N individual similarity ──
            # WHY best-of-N instead of mean-embedding:
            #   The mean of 20+ embeddings from different poses/viewpoints drifts
            #   away from any single frame's embedding, causing both false positives
            #   (wrong cross-camera matches) and false negatives (same person missed).
            #   Comparing against INDIVIDUAL stored embeddings finds the single best
            #   matching view, which is far more robust.
            best_match_id = None
            best_similarity = 0

            for global_id, person in self.persons.items():
                # Do not merge with a Global ID already taken in this frame
                if global_id in active_global_ids_in_camera:
                    continue
                if not person.features:
                    continue

                # Best-of-N: compare query against each recent individual embedding
                similarity = self._compute_best_similarity(feature, person)

                # ── Temporal recency gate + same-camera re-entry ──────────
                # Problem (observed in video): Person A leaves the scene. Person B
                # enters PARTIALLY a few seconds later. The lenient 0.62 threshold
                # accidentally matches Person B against Person A's stale gallery
                # entry ('ID stealing'). Person B then inherits Person A's global ID.
                #
                # General fix: gallery entries older than 8 s revert to the strict
                # similarity_threshold (0.80) to prevent cross-person ID theft.
                #
                # SAME-CAMERA RE-ENTRY EXCEPTION (office camera fix):
                # If the gallery entry was last seen in THIS camera within
                # re_entry_window_secs (default 120 s), use the much more lenient
                # re_entry_threshold (default 0.45) even when stale.
                # Rationale: the person was physically in THIS room recently.
                # OSNet front-to-back view change yields similarity 0.38-0.58,
                # below the strict 0.80 threshold but above 0.45. Without this
                # exception, every re-entry through the same door creates a new ID.
                RECENCY_SECS = 8.0
                seconds_since_seen = time.time() - person.last_seen
                is_stale = seconds_since_seen > RECENCY_SECS

                # Same-camera re-entry check
                was_seen_in_this_camera = camera_name in person.cameras_seen
                is_fresh_in_this_camera = seconds_since_seen < self.re_entry_window_secs

                # Effective threshold logic:
                #   same-camera re-entry (stale but fresh in THIS camera):
                #     -> lenient re_entry_threshold (0.45) -- bypass stale gate
                #   cross-time / cross-camera stale:
                #     -> strict similarity_threshold (0.80) -- prevent ID theft
                #   fresh (within 8 s):
                #     -> query/entry leniency as before
                if is_stale and was_seen_in_this_camera and is_fresh_in_this_camera:
                    effective_thresh = self.re_entry_threshold   # 0.45 -- same-cam re-entry
                elif is_stale:
                    effective_thresh = self.similarity_threshold  # 0.80 -- strict for stale
                else:
                    entry_thresh = (
                        self.edge_match_threshold if person.is_provisional
                        else self.similarity_threshold
                    )
                    effective_thresh = min(query_thresh, entry_thresh)

                if similarity > best_similarity and similarity >= effective_thresh:
                    best_similarity = similarity
                    best_match_id = global_id

            # ── Spatial IoU fallback ───────────────────────────────────────
            # Embedding similarity failed for every gallery entry.  Before creating
            # a new global ID, check whether any gallery entry's LAST KNOWN POSITION
            # overlaps this detection's bbox (IoU ≥ 0.25).
            #
            # Use case: person turns a corner or emerges from behind an obstacle.
            # Their OSNet embedding changes substantially (front → back changes
            # cosine similarity from ~0.85 to ~0.55), but they're still at the
            # same spatial location.  Without this fallback, every direction-change
            # would spawn a new global ID.
            #
            # Safety: only gallery entries NOT active in this frame are eligible
            # (active_global_ids_in_camera is excluded), so we never steal IDs
            # from still-alive tracks that happen to be nearby.
            if best_match_id is None and bbox is not None:
                # ── Appearance-weighted IoU fallback ──────────────────────
                # Combines spatial overlap with appearance similarity:
                #   combined_score = 0.4 * IoU + 0.6 * appearance_similarity
                #
                # Why: pure IoU picks whoever's last_bbox happens to overlap
                # most with the new detection. In an office, multiple people
                # pass through the same door, so their last_bboxes cluster
                # near the door frame. Pure IoU would pick the wrong person.
                # Weighting appearance more heavily (0.6) ensures the person
                # with the most similar OSNet embedding wins the fallback,
                # even when IoU values are similar across candidates.
                best_combined = 0.0
                IOU_FLOOR = 0.20   # minimum IoU to enter the candidate set
                for global_id, person in self.persons.items():
                    if global_id in active_global_ids_in_camera:
                        continue
                    if person.last_bbox is None:
                        continue
                    iou = self._bbox_iou(bbox, person.last_bbox)
                    if iou < IOU_FLOOR:
                        continue
                    # Appearance similarity as primary discriminator
                    appearance_sim = self._compute_best_similarity(feature, person)
                    combined = 0.4 * iou + 0.6 * appearance_sim
                    if combined > best_combined:
                        best_combined = combined
                        best_match_id = global_id

            if best_match_id is not None:
                # Found a match — use existing global ID.
                # Only update the gallery if detection is stable (not at frame edge).
                if update_gallery:
                    self.persons[best_match_id].add_feature(
                        feature, camera_name, is_stable=not is_edge
                    )
                if bbox is not None:
                    self.persons[best_match_id].last_bbox = bbox  # always update position
                self.local_to_global[camera_name][local_track_id] = best_match_id
                return best_match_id
            else:
                # No match — create new gallery entry.
                # Always add first embedding even for edge detections (needed to
                # make the new entry searchable). Mark its quality correctly.
                new_global_id = self._create_new_person(
                    feature, camera_name, is_stable=not is_edge, bbox=bbox
                )
                self.local_to_global[camera_name][local_track_id] = new_global_id
                return new_global_id
    
    def _compute_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """Compute cosine similarity between two features."""
        f1 = feature1 / (np.linalg.norm(feature1) + 1e-8)
        f2 = feature2 / (np.linalg.norm(feature2) + 1e-8)
        return float(np.dot(f1, f2))

    @staticmethod
    def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """
        Intersection-over-Union between two [x1,y1,x2,y2] bounding boxes.

        Used as a spatial fallback in find_or_create_global_id when embedding
        similarity drops below threshold (e.g. person turns a corner, changing
        their OSNet embedding significantly while remaining at the same position).
        IoU threshold 0.25 is intentionally low: partial bboxes and frame-to-frame
        position jitter mean a strict threshold would miss valid matches.
        """
        ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
        bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / (area_a + area_b - inter + 1e-8)

    def _compute_best_similarity(self, query: np.ndarray, person: PersonRecord) -> float:
        """
        Return the MAXIMUM cosine similarity between `query` and any stored
        full-body embedding for this person.

        Search pool priority:
          1. stable_features (full-body only, up to 30, spans entire timeline)
             Used when >= 3 stable embeddings are available.  This preserves
             early-video appearances so a Camera_2 query from the START of the
             footage can still match Camera_1's frame-0 embedding even after
             Camera_1 has processed 1200+ additional frames.
          2. features[-15:] (fallback for new/provisional entries with < 3 stable)
             Most-recent 15 mixed embeddings — the original behaviour.

        Why best-of-N instead of mean-embedding:
          - Mean of 20 embeddings from mixed viewpoints/poses drifts off the
            unit hypersphere, degrading discriminability.
          - A single GOOD embedding at the right angle gives 0.90+ similarity
            even when the mean would only score 0.60.
        """
        if len(person.stable_features) >= 3:
            # Prefer full-body embeddings that span the whole video timeline
            search_pool = list(person.stable_features)
        elif person.features:
            # Fallback: use last 15 mixed embeddings for new/provisional entries
            search_pool = list(person.features[-15:])
        else:
            return 0.0

        # Include departure embeddings (last 5 'leaving' views).
        # These transitional views bridge sitting (right-front) and re-entry
        # (back-right), boosting best-match similarity from 0.38-0.52 to
        # 0.50-0.65 for the same person re-entering through the same door.
        # Crucially, after each successful re-entry the departure_features
        # are refreshed with the new 'leaving again' frames, making every
        # subsequent re-entry progressively easier to match.
        if person.departure_features:
            search_pool = search_pool + list(person.departure_features)

        q = query / (np.linalg.norm(query) + 1e-8)
        max_sim = 0.0
        for feat in search_pool:
            f = feat / (np.linalg.norm(feat) + 1e-8)
            sim = float(np.dot(q, f))
            if sim > max_sim:
                max_sim = sim
        return max_sim
    
    def _create_new_person(self, feature: np.ndarray, camera_name: str,
                           is_stable: bool = True,
                           bbox: Optional[np.ndarray] = None) -> int:
        """Create a new person record.

        Args:
            feature: Initial ReID embedding.
            camera_name: Camera that first saw this person.
            is_stable: True if the initial detection is full-body (non-edge).
                       Determines whether the first embedding counts toward
                       stable_count, which gates the 'provisional' flag.
            bbox: Initial bounding box; stored as last_bbox for spatial fallback.
        """
        # Clean up old entries if gallery is full
        if len(self.persons) >= self.max_gallery_size:
            self._cleanup_old_entries()

        global_id = self.next_global_id
        self.next_global_id += 1

        self.persons[global_id] = PersonRecord(
            global_id=global_id,
            features=[feature],
            last_seen=time.time(),
            cameras_seen={camera_name},
            total_detections=1,
            stable_count=1 if is_stable else 0,           # track quality from birth
            stable_features=[feature] if is_stable else [], # preserve first good view
            last_bbox=bbox,                                # for spatial fallback
        )

        return global_id
    
    def _cleanup_old_entries(self, max_age_seconds: float = 3600):
        """Remove old entries from gallery."""
        current_time = time.time()
        to_remove = []
        
        for global_id, person in self.persons.items():
            if current_time - person.last_seen > max_age_seconds:
                to_remove.append(global_id)
        
        for global_id in to_remove[:len(self.persons) // 4]:  # Remove at most 25%
            del self.persons[global_id]

    def record_departure(self, global_id: int,
                         embeddings: List[np.ndarray]) -> None:
        """Store departure embeddings for a person leaving this camera's view.

        Called by MultiCameraTracker.process_frame() with the pre-departure
        rolling buffer -- the last N non-zero embeddings captured BEFORE
        BotSort dropped the track.  This buffer contains the person's actual
        last-seen appearance (e.g. walking toward the door, partial side/back
        views near the door threshold) rather than garbage from BotSort's
        ghost-tracking period (Kalman-only predictions from off-screen bboxes).

        The departure_features list is REPLACED on each call (not appended)
        so successive departures always reflect the most recent leaving view.
        Crucially, the near-door frames (last 2-3 in the buffer) are edge/
        partial views showing the person's back or side, which are much closer
        in OSNet cosine space to the re-entry embedding than sitting views are.

        Args:
            global_id: The global ID of the person who just left.
            embeddings: Up to 5 non-zero feature vectors from the rolling
                        pre-departure buffer, newest last.
        """
        with self.lock:
            if global_id in self.persons:
                person = self.persons[global_id]
                person.departure_features = []          # replace, don't accumulate
                for feat in embeddings:
                    person.add_departure_feature(feat)

    def clear_camera_mappings(self, camera_name: str):
        """Clear local-to-global mappings for a camera (for new session)."""
        with self.lock:
            self.local_to_global[camera_name].clear()
    
    def get_statistics(self) -> dict:
        """Get gallery statistics."""
        return {
            'total_persons': len(self.persons),
            'next_global_id': self.next_global_id,
            'cameras': list(set(
                cam for p in self.persons.values() for cam in p.cameras_seen
            )),
            'cross_camera_persons': sum(
                1 for p in self.persons.values() if len(p.cameras_seen) > 1
            )
        }


class VideoSource:
    """
    Unified video source handler for both files and RTSP streams.
    """
    
    def __init__(self, source: Union[str, int], buffer_size: int = 2):
        """
        Initialize video source.
        
        Args:
            source: Video file path, RTSP URL, or camera index
            buffer_size: Frame buffer size for RTSP streams
        """
        self.source = source
        self.buffer_size = buffer_size
        self.cap = None
        self.is_stream = False
        self.frame_queue = None
        self.reader_thread = None
        self.running = False
        
        # Properties
        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_frames = 0
    
    def open(self) -> bool:
        """Open the video source."""
        # Determine source type
        if isinstance(self.source, int):
            self.is_stream = True  # Webcam
        elif isinstance(self.source, str):
            self.is_stream = self.source.lower().startswith(('rtsp://', 'http://', 'https://'))
        
        # Open capture
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            return False
        
        # Get properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # FPS fix: cv2.CAP_PROP_FPS is unreliable for RTSP streams — it often
        # returns 0, 90000, or other garbage values.  Clamp to a sane range
        # (1-60).  If still out of range, fall back to 25 FPS.
        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if raw_fps and 1 <= raw_fps <= 60:
            self.fps = raw_fps
        else:
            self.fps = 25  # safe default

        # For live streams the camera-reported FPS is used only as a hint;
        # we always write output at a safe, capped value (set in process_source).
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.total_frames <= 0:
            self.total_frames = float('inf')  # Stream
        
        # Start threaded reader for streams
        if self.is_stream:
            self.frame_queue = queue.Queue(maxsize=self.buffer_size)
            self.running = True
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()
        
        return True
    
    def _reader_loop(self):
        """Background thread for reading stream frames."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Drop old frames if buffer full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the source."""
        if self.is_stream:
            try:
                frame = self.frame_queue.get(timeout=2.0)
                return True, frame
            except queue.Empty:
                return False, None
        else:
            return self.cap.read()
    
    def release(self):
        """Release the video source."""
        self.running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
    
    def get_info(self) -> dict:
        """Get source information."""
        return {
            'source': str(self.source),
            'is_stream': self.is_stream,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames if self.total_frames != float('inf') else 'Stream'
        }


class MultiCameraTracker:
    """
    Multi-Camera Person Tracking System.
    
    Tracks persons across multiple cameras while maintaining
    consistent global IDs.
    
    Features:
    - Unified video file and RTSP support
    - Global person gallery for cross-camera matching
    - Same ID maintained when person moves between cameras
    - Thread-safe for concurrent camera processing
    """
    
    def __init__(
        self,
        reid_model_path: str,
        device: str = 'cuda',
        similarity_threshold: float = 0.7,
        re_entry_threshold: float = 0.45,
        re_entry_window_secs: float = 120.0,
        yolo_model_path: str = 'yolov8s.pt',
        day_model_path: str = 'yolov8s.pt',
        night_model_path: str = 'runs/detect/yolov8s_rot0/weights/best.pt',
        conf: float = 0.45,
        imgsz: int = 960,
        botsort_cfg: str = 'botsort_custom.yaml'
    ):
        """
        Initialize multi-camera tracker.
        
        Args:
            reid_model_path: Path to trained Re-ID model
            device: 'cuda' or 'cpu'
            similarity_threshold: Threshold for cross-camera matching (0-1)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {self.device}")

        # --- GPU VRAM check & imgsz safety ---
        self.imgsz = imgsz
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU VRAM : {vram_gb:.1f} GB")
            if vram_gb < 6.0:
                print(f"\n⚠️  WARNING: Only {vram_gb:.1f} GB VRAM detected.")
                print(f"   Recommended minimum for imgsz=960: 6 GB")
                try:
                    resp = input("   Drop imgsz from 960 \u2192 640 for safety? (y/n): ").strip().lower()
                except EOFError:
                    resp = 'y'
                if resp == 'y':
                    self.imgsz = 640
                    print("   \u2705 imgsz dropped to 640.")

        # Adaptive day / night model paths (Saved for potential future use)
        self.yolo_model_path  = yolo_model_path
        self.day_model_path   = day_model_path
        self.night_model_path = night_model_path
        self.conf        = conf
        self._botsort_cfg = botsort_cfg

        # YOLO detector is now loaded per-camera in process_source
        # to isolate BoT-SORT tracking states between cameras.
        
        # ── Re-ID backend selection ────────────────────────────────────────
        # Pass model_name='osnet_x1_0' (or x0_75 / x0_5) to use the
        # industry-standard OSNet pre-trained on Market-1501 + MSMT17.
        # Pass a .pth file path to use the project-trained custom model.
        # ─────────────────────────────────────────────────────────────────
        _OSNET_NAMES = ('osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x1_4')
        self._use_osnet = reid_model_path in _OSNET_NAMES

        if self._use_osnet:
            if not _TORCHREID_AVAILABLE:
                msg = (
                    "torchreid failed to import. OSNet backend is unavailable.\n"
                    "\n"
                    "── Fix options (run in Colab) ───────────────────────────\n"
                    "  Option 1 (PyPI):   pip install torchreid\n"
                    "  Option 2 (source): pip install git+https://github.com/KaiyangZhou/deep-person-reid.git\n"
                    "─────────────────────────────────────────────────────────\n"
                )
                if _TORCHREID_IMPORT_ERROR:
                    msg += f"Actual import error was:\n  {_TORCHREID_IMPORT_ERROR}\n"
                raise ImportError(msg)
            print(f"🧠 Loading OSNet Re-ID model ({reid_model_path})...")
            print("   Downloading pre-trained weights (Market-1501 + MSMT17) if not cached...")
            self.reid_extractor = OsNetExtractor(
                model_name=reid_model_path,
                device=str(self.device)
            )
            self._embedding_dim = 512
            print("✅ OSNet loaded! (~90% Rank-1, pre-trained on 128k person images)")
        else:
            print("🧠 Loading custom Re-ID model...")
            self.reid_model = ReIDNetwork(embedding_dim=128, pretrained=False)
            checkpoint = torch.load(reid_model_path, map_location=self.device)
            self.reid_model.load_state_dict(checkpoint['model_state_dict'])
            self.reid_model = self.reid_model.to(self.device)
            self.reid_model.eval()
            self._embedding_dim = 128

            # Preprocessing pipeline (custom model only)
            self.reid_transform = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            print("✅ Custom Re-ID model loaded.")
        
        # Global person gallery (shared across all cameras).
        # edge_match_threshold calibration:
        #
        # Old value: max(0.45, sim - 0.20) = 0.50
        #   → Too lenient. 24 total unique persons detected instead of ~40 real.
        #   → Different people with similar appearance (same clothing color) merged.
        #
        # New value: max(0.62, sim - 0.08) = 0.62 for sim=0.70
        #   → Same person partial-view: cosine similarity ~0.65–0.80 → still matches
        #   → Different people generic: cosine similarity ~0.30–0.60 → most blocked
        #   → The temporal recency gate provides an additional safety net for
        #     the narrow 0.62–0.70 ambiguous zone.
        _edge_thresh = max(0.62, similarity_threshold - 0.08)
        self.gallery = GlobalPersonGallery(
            similarity_threshold=similarity_threshold,
            edge_match_threshold=_edge_thresh,
            re_entry_threshold=re_entry_threshold,
            re_entry_window_secs=re_entry_window_secs,
        )
        
        # Lock for thread-safe Re-ID inference.
        # The Re-ID model is shared across camera threads; PyTorch eval-mode
        # forward() is generally thread-safe but we protect it with a lock
        # to prevent subtle race conditions on the same tensor buffers.
        self._reid_lock = threading.Lock()

        # Per-camera track state for departure detection.
        # Structure: camera_name -> {local_track_id -> (global_id, feature_vector)}
        # Updated at the end of every process_frame() call.  Comparing this dict
        # against the current frame's active track IDs reveals which tracks just
        # disappeared so their last embedding can be stored as departure_features
        # in the gallery for use in future same-camera re-entry matching.
        self._prev_frame_tracks: Dict[str, Dict[int, Tuple[int, np.ndarray]]] = defaultdict(dict)

        # Rolling buffer of the last 5 non-zero embeddings per (camera, local_track_id).
        # Updated EVERY FRAME (including edge/partial frames close to the door).
        # KEY PURPOSE: when a person walks toward a door and exits, the last 5 non-zero
        # embeddings are side/back transitional views (scored 0.55-0.75 vs re-entry)
        # rather than the garbage Kalman-prediction crops produced during BotSort's
        # 3-second ghost-tracking period (scored 0.20-0.30 vs re-entry).
        # When departure fires, this buffer replaces the ghost-frame embedding.
        # camera_name -> {local_track_id -> [feat_t-4, t-3, t-2, t-1, t_last_real]}
        self._pre_departure_buffer: Dict[str, Dict[int, List[np.ndarray]]] = defaultdict(dict)
        
        # Tracklet Stabilization buffer stores features before running Re-ID matching
        self.MIN_STABLE_FRAMES = 15
        self._pending_tracks: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        
        # Colors for visualization
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(10000, 3), dtype=np.uint8)
        
        print("✅ MultiCameraTracker initialized!")
    
    def extract_features(self, frame: np.ndarray, detections: np.ndarray) -> List[np.ndarray]:
        """
        Extract Re-ID features for detected persons.

        Supports two backends:
          OSNet    — batches all crops in one forward pass (fast, accurate)
          Custom   — processes each crop individually (legacy)

        Returns a list of numpy feature vectors, one per detection.
        Zero vectors are returned for invalid/empty bounding boxes.
        """
        # Pre-allocate output: one slot per detection, filled with zeros
        features: List[np.ndarray] = [
            np.zeros(self._embedding_dim) for _ in detections
        ]

        if len(detections) == 0:
            return features

        # ── Step 1: Crop and validate all bounding boxes ──────────────────
        # MIN SIZE FILTER: Skip very small bboxes — they come from distant/
        # partially-visible people whose crops are too low-resolution for
        # OSNet to extract reliable features. These detections still get
        # tracked by BotSort (the zero vector keeps their gallery slot), but
        # they won't corrupt gallery entries with bad embeddings.
        MIN_W, MIN_H = 32, 64   # pixels — tune up if people are always far away
        valid_crops: List[np.ndarray] = []   # BGR crops
        valid_indices: List[int] = []         # maps crop → original det index

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue  # degenerate box — keep zero vector
            if w < MIN_W or h < MIN_H:
                continue  # too small — low-quality crop, keep zero vector

            valid_crops.append(frame[y1:y2, x1:x2])
            valid_indices.append(i)


        if not valid_crops:
            return features

        # ── Step 2: Run inference ──────────────────────────────────────────
        if self._use_osnet:
            # OSNet: accepts a list of RGB numpy arrays, returns (N, 512) tensor
            crops_rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in valid_crops]
            with self._reid_lock:
                feat_tensor = self.reid_extractor(crops_rgb)   # (N, 512)
            feat_array = feat_tensor.cpu().numpy()             # already L2-normalised
            for arr_idx, det_idx in enumerate(valid_indices):
                features[det_idx] = feat_array[arr_idx]
        else:
            # Custom model: process each crop individually
            for arr_idx, det_idx in enumerate(valid_indices):
                crop_rgb = cv2.cvtColor(valid_crops[arr_idx], cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                crop_tensor = self.reid_transform(pil_crop).unsqueeze(0).to(self.device)
                with self._reid_lock:
                    with torch.no_grad():
                        feature = self.reid_model(crop_tensor)
                features[det_idx] = feature.cpu().numpy().flatten()

        return features

    @staticmethod
    def _measure_brightness(frame: np.ndarray) -> float:
        """Return mean pixel brightness (0–255) of a BGR frame.
        Used to auto-select the day/night YOLO model and scale CLAHE strength."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def enhance_frame_lowlight(self, frame: np.ndarray,
                               brightness: float = None) -> np.ndarray:
        """
        Adaptive low-light enhancement before passing to YOLO.

        Brightness-gated pipeline:
          > 90  (daytime)  → skip CLAHE entirely (prevents afternoon flickering)
          50-90 (twilight) → light CLAHE only (clipLimit=1.5)
          < 50  (night)    → full CLAHE (clipLimit=3.0) + gamma correction

        Two-stage pipeline:
        1. CLAHE on the L channel of LAB colour space:
           Boosts local contrast in dark regions (e.g. shadowed person against
           dark road) without over-brightening already-lit areas (street lamps).
        2. Gamma correction (gamma < 1.0 brightens shadows):
           Lifts overall dark-pixel intensity, making silhouettes more visible.

        The original frame is NOT modified; a new frame is returned.
        """
        if brightness is None:
            brightness = self._measure_brightness(frame)

        # Bright scenes: skip all enhancement — eliminates CLAHE flicker on daytime footage
        if brightness > 90:
            return frame

        # Scale CLAHE aggressiveness to darkness level
        clip_limit = 1.5 if brightness > 50 else 3.0

        # --- Adaptive CLAHE on LAB L-channel ---
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        enhanced = cv2.merge([l_ch, a_ch, b_ch])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Gamma correction only for truly dark scenes (brightness < 50)
        if brightness < 50:
            gamma     = 0.6
            inv_gamma = 1.0 / gamma
            table = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
                dtype=np.uint8
            )
            enhanced = cv2.LUT(enhanced, table)

        return enhanced
    
    def process_frame(
        self,
        frame: np.ndarray,
        camera_name: str,
        local_tracker: YOLO,
        brightness: float = None
    ) -> Tuple[List[Tuple[int, np.ndarray]], np.ndarray]:
        """
        Process a single frame.
        
        Args:
            frame: BGR image
            camera_name: Name of the camera
            local_tracker: Camera-specific DeepSORT tracker
            
        Returns:
            tracks: List of (global_id, bbox) tuples
            annotated_frame: Frame with visualizations
        """
        # Adaptive CLAHE: strength auto-scaled to brightness (skipped on bright frames)
        if brightness is None:
            brightness = self._measure_brightness(frame)
        enhanced_frame = self.enhance_frame_lowlight(frame, brightness)

        # Track persons using built-in BoT-SORT with tuned parameters.
        # botsort_custom.yaml lowers match_thresh (0.65 vs default 0.8) to
        # reduce ID swaps when two people cross paths.
        results = local_tracker.track(
            enhanced_frame,
            persist=True,
            tracker=self._botsort_cfg,
            classes=[0],
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False
        )
        
        detections = []
        local_tracks = []
        
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                detections.append([box[0], box[1], box[2], box[3]])
                local_tracks.append({"track_id": track_id, "bbox": box})
                
        detections = np.array(detections) if detections else np.array([])
        
        if len(detections) == 0:
            return [], frame

        # Extract features from the ORIGINAL frame (not enhanced):
        # Re-ID model was trained on normal-exposure images; enhanced frame
        # may distort colour/texture features and hurt matching accuracy.
        features = self.extract_features(frame, detections)
        
        # Get all active local track IDs in this frame to prevent gallery collisions
        active_local_ids = {track_info["track_id"] for track_info in local_tracks}

        # Identify partial/occluded detections.  Two signals are used:
        #
        # 1. Border proximity (adaptive asymmetric margins):
        #    Bottom edge gets 12% of frame height — people enter from bottom
        #    in top-mounted CCTV cameras far more than from sides.
        #    Left/right/top use 5%.  All floors ensure a minimum pixel count
        #    so the check still works at unusual resolutions.
        #
        # 2. Aspect ratio (h/w < 1.5):  a full standing adult viewed from CCTV
        #    height is typically 2.5–3.5:1 (tall).  A head-only or
        #    head+shoulders crop from a top-down camera is ~0.8–1.4:1 (nearly
        #    square).  Any bbox that is more wide than ~1.5× its height is almost
        #    certainly a partial-body crop even when located in the frame centre.
        #
        # Partial detections:
        #   • Do NOT update existing gallery entries (noisy embeddings corrupt them)
        #   • DO match against gallery using a lenient threshold (edge_match_threshold)
        #   • CAN create new gallery entries (so the track remains visible)
        frame_h, frame_w = frame.shape[:2]
        MARGIN_X     = max(40, int(frame_w * 0.05))   # left / right — 5% of width
        MARGIN_Y_TOP = max(30, int(frame_h * 0.05))   # top — 5% of height
        MARGIN_Y_BOT = max(80, int(frame_h * 0.12))   # bottom — 12% (primary entry edge)

        global_tracks = []
        for track_info, track_feature in zip(local_tracks, features):
            x1, y1, x2, y2 = map(int, track_info["bbox"])
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            w_box = max(1, x2 - x1)
            h_box = max(1, y2 - y1)
            aspect = h_box / w_box  # full standing adult ≈ 2.5–3.5

            is_edge = (
                x1 < MARGIN_X or                    # left border
                x2 > frame_w - MARGIN_X or          # right border
                y1 < MARGIN_Y_TOP or                # top border
                y2 > frame_h - MARGIN_Y_BOT or      # bottom border (larger margin)
                aspect < 1.5                        # partial body (head-only / torso-only)
            )

            local_id = track_info["track_id"]
            
            with self.gallery.lock:
                already_mapped = local_id in self.gallery.local_to_global[camera_name]
                
            if already_mapped:
                global_id = self.gallery.find_or_create_global_id(
                    local_id, track_feature, camera_name,
                    active_local_ids,
                    update_gallery=not is_edge,  # partial detections never corrupt gallery
                    is_edge=is_edge,             # enables lenient threshold for matching
                    bbox=bbox,                   # spatial fallback + position tracking
                )
                global_tracks.append((global_id, track_info["bbox"]))
            else:
                # ── Tracklet Stabilization ──────────────
                # Accumulate frames for new tracks before deciding their Global ID
                cam_pending = self._pending_tracks[camera_name]
                if local_id not in cam_pending:
                    cam_pending[local_id] = {"features": [], "stable_count": 0, "bboxes": []}
                
                t_data = cam_pending[local_id]
                if np.any(track_feature != 0):
                    t_data["features"].append(track_feature.copy())
                    t_data["bboxes"].append(bbox)
                    if not is_edge:
                        t_data["stable_count"] += 1
                        
                # Check for maturity: 15 stable frames, or 45 frames fallback (e.g. forever blocked)
                if t_data["stable_count"] >= self.MIN_STABLE_FRAMES or len(t_data["features"]) >= 45:
                    valid_feats = t_data["features"]
                    if valid_feats:
                        mean_feat = np.mean(valid_feats, axis=0)
                        query_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-8)
                    else:
                        query_feat = track_feature
                        
                    global_id = self.gallery.find_or_create_global_id(
                        local_id, query_feat, camera_name,
                        active_local_ids,
                        update_gallery=True,  
                        is_edge=False,        # Query is now a stable mean
                        bbox=bbox,            
                    )
                    
                    # Backfill gallery to immediately populate robust historical views
                    if valid_feats:
                        with self.gallery.lock:
                            if global_id in self.gallery.persons:
                                person = self.gallery.persons[global_id]
                                step = max(1, len(valid_feats) // 10)
                                for f in valid_feats[::step]:
                                    person.add_feature(f, camera_name, is_stable=True)
                            
                    del cam_pending[local_id]
                    global_tracks.append((global_id, track_info["bbox"]))
                else:
                    global_tracks.append(("?", track_info["bbox"]))

        # ── Pre-departure buffer update ─────────────────────────────────────
        # Maintain a rolling buffer of the last 5 non-zero embeddings per local
        # track, updated EVERY FRAME including edge/partial frames.
        # Must run BEFORE departure detection so when a track disappears this
        # frame, the buffer already holds its last-seen appearance (near-door
        # side/back views) not the ghost-tracking garbage in _prev_frame_tracks.
        cam_buf = self._pre_departure_buffer[camera_name]
        for i in range(len(local_tracks)):
            track_id = local_tracks[i]["track_id"]
            feat = features[i]
            if np.any(feat != 0):               # skip zero-vectors (off-frame crops)
                if track_id not in cam_buf:
                    cam_buf[track_id] = []
                cam_buf[track_id].append(feat.copy())
                if len(cam_buf[track_id]) > 5:  # keep only the last 5
                    cam_buf[track_id] = cam_buf[track_id][-5:]

        # ── Departure detection ──────────────────────────────────────────────
        # Compare this frame's active local track IDs against the previous
        # frame's dict.  Any track ID present last frame but absent now has
        # just disappeared (BotSort dropped the track after track_buffer frames).
        # Use the pre-departure buffer (actual last-seen walking frames) instead
        # of the ghost tracking embedding in _prev_frame_tracks (stale+garbage).
        prev_tracks = self._prev_frame_tracks.get(camera_name, {})
        current_local_ids = {t["track_id"] for t in local_tracks}
        for departed_local_id, (departed_gid, departed_feat) in prev_tracks.items():
            if departed_local_id not in current_local_ids:
                # Retrieve and consume the pre-departure buffer for this track.
                departure_buf = cam_buf.pop(departed_local_id, None)
                if departure_buf:
                    # Store up to 5 buffered embeddings — captures the last real
                    # visible frames (side/back near door), not ghost predictions.
                    # These score 0.55-0.75 vs re-entry vs 0.38-0.52 for sitting.
                    self.gallery.record_departure(departed_gid, departure_buf)
                elif np.any(departed_feat != 0):
                    # Fallback: ghost-frame embedding only if buffer is empty
                    self.gallery.record_departure(departed_gid, [departed_feat])

        # Cleanup stale pending tracks that dropped before maturing
        cam_pending = self._pending_tracks[camera_name]
        pending_to_remove = [tid for tid in cam_pending if tid not in current_local_ids]
        for tid in pending_to_remove:
            del cam_pending[tid]

        # Update per-camera track state for next frame's departure detection
        self._prev_frame_tracks[camera_name] = {
            local_tracks[i]["track_id"]: (global_tracks[i][0], features[i])
            for i in range(len(local_tracks))
        }

        # Draw results on the enhanced frame so the visualisation shows the
        # brightness-corrected view that YOLO actually used for detection.
        annotated_frame = self.draw_tracks(enhanced_frame.copy(), global_tracks, camera_name)
        
        return global_tracks, annotated_frame
    
    def draw_tracks(
        self, 
        frame: np.ndarray, 
        tracks: List[Tuple[Union[int, str], np.ndarray]], 
        camera_name: str
    ) -> np.ndarray:
        """Draw tracking results on frame."""
        for global_id, bbox in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            if isinstance(global_id, str):
                color = (128, 128, 128)
            else:
                color = tuple(map(int, self.colors[global_id % len(self.colors)]))
            
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # ID label (show GLOBAL ID prominently)
            label = f'ID: {global_id}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            cv2.rectangle(
                frame, 
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1), 
                color, -1
            )
            cv2.putText(
                frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
        
        # Camera name label
        cv2.putText(
            frame, f'Camera: {camera_name}',
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        return frame
    
    def process_source(
        self,
        source: Union[str, int],
        camera_name: str = "Camera_1",
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        display: bool = False
    ) -> dict:
        """
        Process a single video source (file or RTSP).
        
        Args:
            source: Video file path, RTSP URL, or camera index
            camera_name: Name for this camera
            output_path: Path to save output video
            max_frames: Maximum frames to process
            display: Show live display (local only)
            
        Returns:
            Statistics dictionary
        """
        print(f"\n📹 Processing: {source}")
        print(f"   Camera name: {camera_name}")
        
        # Open source
        video_source = VideoSource(source)
        if not video_source.open():
            print(f"❌ Cannot open source: {source}")
            return {'error': 'Cannot open source'}
        
        info = video_source.get_info()
        print(f"   Resolution: {info['width']}x{info['height']}")
        print(f"   FPS: {info['fps']}")
        print(f"   Type: {'Stream' if info['is_stream'] else 'Video file'}")
        
        # Setup output writer
        out = None
        output_fps = 25  # default / stream fallback
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

            # For live streams, always use 25 FPS output.
            # For video files, honour the source FPS (already clamped 1-60).
            output_fps = 25 if info['is_stream'] else info['fps']

            # Use XVID + AVI: far more reliable on Windows than mp4v.
            # mp4v can produce unplayable files when the process ends early.
            # Automatically rewrite any .mp4 output path to .avi.
            avi_output_path = output_path
            if output_path.lower().endswith('.mp4'):
                avi_output_path = output_path[:-4] + '.avi'
                print(f"   ⚠️  Output changed to AVI for reliability: {avi_output_path}")

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(
                avi_output_path, fourcc, output_fps,
                (info['width'], info['height'])
            )
            if not out.isOpened():
                print("❌ VideoWriter failed to open. Output will not be saved.")
                out = None
            else:
                print(f"   Output: {avi_output_path}")
                print(f"   Output FPS: {output_fps}")
        
        # -----------------------------------------------------------------
        # [RESERVED FOR FUTURE USE] Adaptive Day / Night model switching
        # -----------------------------------------------------------------
        # If extreme darkness requires returning to a dual-model architecture,
        # uncomment the block below and the switching logic in the processing loop.
        #
        # _SWITCH_TO_NIGHT = 70    # go night if brightness drops below this
        # _SWITCH_TO_DAY   = 90    # go day  if brightness rises above this
        # _CHECK_EVERY     = 90    # re-evaluate every N frames
        # _HYSTERESIS      = 2     # consecutive threshold-crossing checks before switching
        #
        # _current_mode   = 'day'
        # _pending_mode   = 'day'
        # _pending_count  = 0
        # _brightness_now = 128.0  # assume bright until first measurement
        #
        # print(f"   DAY   model : {self.day_model_path}")
        # print(f"   NIGHT model : {self.night_model_path}")
        # print(f"   Conf={self.conf}  ImgSz={self.imgsz}")
        # print("   Starting in DAY mode (auto-switches based on brightness)")
        # local_tracker = YOLO(self.day_model_path)
        # -----------------------------------------------------------------

        print(f"   Loading YOLO fallback: {self.yolo_model_path}")
        print(f"   Conf={self.conf}  ImgSz={self.imgsz}")
        local_tracker = YOLO(self.yolo_model_path)
        
        # Processing statistics
        stats = {
            'camera_name': camera_name,
            'source': str(source),
            'frames_processed': 0,
            'total_detections': 0,
            'unique_persons': set(),
            'start_time': time.time()
        }
        
        # Frame-duplication clock for live streams.
        #
        # WHY it's needed:
        #   VideoWriter has no concept of real time.  It encodes whatever frames
        #   you hand it at the declared output_fps.  If the CPU only processes
        #   2 frames/second but output_fps=25, each "real" second produces only
        #   2 file-frames → plays at 25/2 = 12.5× speed.
        #
        # FIX:
        #   After processing each frame, calculate how much real time has passed
        #   since the last written frame.  Write that frame N = round(elapsed × fps)
        #   times so the output file contains the correct number of frames per
        #   real second (duplicating the last frame during slow processing
        #   gives a slight freeze-frame artefact but correct playback speed).
        last_frame_real_time = time.time()   # wall-clock of last written frame
        MAX_DUP = int(output_fps * 5)        # safety cap: never > 5 seconds of dups

        # Determine total frames for progress bar
        total = max_frames if max_frames else (
            info['total_frames'] if info['total_frames'] != 'Stream' else None
        )
        
        pbar = tqdm(total=total, desc=f"Processing {camera_name}")
        
        print("\n🎬 Starting processing... (Ctrl+C to stop)")
        
        try:
            while True:
                # Check max frames
                if max_frames and stats['frames_processed'] >= max_frames:
                    break
                
                # Read frame
                ret, frame = video_source.read()
                if not ret:
                    if video_source.is_stream:
                        continue  # Keep trying for streams
                    else:
                        break  # End of video file
                
                # Measure brightness once per frame (currently used for CLAHE scaling)
                _brightness_now = self._measure_brightness(frame)

                # -----------------------------------------------------------------
                # [RESERVED FOR FUTURE USE] Adaptive model switching
                # -----------------------------------------------------------------
                # if stats['frames_processed'] % _CHECK_EVERY == 0 and stats['frames_processed'] > 0:
                #     if _brightness_now < _SWITCH_TO_NIGHT:
                #         _candidate = 'night'
                #     elif _brightness_now > _SWITCH_TO_DAY:
                #         _candidate = 'day'
                #     else:
                #         _candidate = _current_mode  # hysteresis zone — hold current mode
                #
                #     if _candidate == _pending_mode:
                #         _pending_count += 1
                #     else:
                #         _pending_mode  = _candidate
                #         _pending_count = 1
                #
                #     if _pending_count >= _HYSTERESIS and _candidate != _current_mode:
                #         _current_mode  = _candidate
                #         _pending_count = 0
                #         _model_path = (self.day_model_path if _current_mode == 'day'
                #                        else self.night_model_path)
                #         print(f"\n🔄  Brightness={_brightness_now:.0f} → "
                #               f"Switching to {_current_mode.upper()} model: {_model_path}")
                #         local_tracker = YOLO(_model_path)
                # -----------------------------------------------------------------

                # Process frame (pass pre-computed brightness so CLAHE doesn't re-measure)
                tracks, annotated_frame = self.process_frame(
                    frame, camera_name, local_tracker, brightness=_brightness_now
                )
                
                # Update statistics
                stats['frames_processed'] += 1
                stats['total_detections'] += len(tracks)
                for global_id, _ in tracks:
                    stats['unique_persons'].add(global_id)
                
                # Add info overlay
                elapsed = time.time() - stats['start_time']
                fps = stats['frames_processed'] / elapsed if elapsed > 0 else 0
                
                info_lines = [
                    f"FPS: {fps:.1f}",
                    f"Frame: {stats['frames_processed']}",
                    f"Persons: {len(tracks)}",
                    f"Global IDs: {len(stats['unique_persons'])}"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(
                        annotated_frame, line,
                        (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                
                # Timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    annotated_frame, timestamp,
                    (info['width'] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                
                # Write output — frame-duplication for real-time playback
                if out:
                    now = time.time()
                    if info['is_stream']:
                        # How many output frames should this one real frame represent?
                        real_elapsed = now - last_frame_real_time
                        last_frame_real_time = now
                        dup_count = max(1, min(round(real_elapsed * output_fps), MAX_DUP))
                        for _ in range(dup_count):
                            out.write(annotated_frame)
                    else:
                        # Video file: write every processed frame (speed already correct)
                        out.write(annotated_frame)
                
                # Display (local only)
                if display:
                    cv2.imshow(camera_name, annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                pbar.update(1)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped by user")
        
        finally:
            pbar.close()
            video_source.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Final statistics
        elapsed = time.time() - stats['start_time']
        stats['elapsed_time'] = elapsed
        stats['average_fps'] = stats['frames_processed'] / elapsed if elapsed > 0 else 0
        stats['unique_persons'] = len(stats['unique_persons'])
        
        print(f"\n📊 Statistics for {camera_name}:")
        print(f"   Frames: {stats['frames_processed']}")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Avg FPS: {stats['average_fps']:.1f}")
        print(f"   Unique persons: {stats['unique_persons']}")
        
        return stats
    
    def process_multi_camera(
        self,
        cameras: List[Dict],
        output_dir: str = "outputs",
        max_frames_per_camera: Optional[int] = None,
        sequential: bool = False   # Default: parallel (all cameras at once)
    ) -> List[dict]:
        """
        Process multiple cameras simultaneously using one thread per camera.

        Args:
            cameras: List of camera configurations
                     [{"name": "Entrance", "source": "rtsp://..."}, ...]
            output_dir: Directory for output videos
            max_frames_per_camera: Max frames per camera (None = run until stopped)
            sequential: False (default) = all cameras run in parallel.
                        True = one camera at a time (legacy mode).

        Returns:
            List of statistics for each camera
        """
        print(f"\n{'='*60}")
        print(f"📹 MULTI-CAMERA TRACKING")
        print(f"   Cameras: {len(cameras)}")
        print(f"   Mode: {'Sequential' if sequential else 'Parallel (all cameras simultaneously)'}")
        print(f"   Output: {output_dir}")
        print(f"{'='*60}")

        os.makedirs(output_dir, exist_ok=True)

        if sequential:
            # ── Legacy mode: one camera at a time ─────────────────────────
            all_stats = []
            for i, cam_config in enumerate(cameras):
                name = cam_config.get('name', f'Camera_{i+1}')
                source = cam_config.get('source')
                print(f"\n{'='*60}")
                print(f"📷 Camera {i+1}/{len(cameras)}: {name}")
                print(f"{'='*60}")
                output_path = os.path.join(output_dir, f"{name}.mp4")
                stats = self.process_source(
                    source=source,
                    camera_name=name,
                    output_path=output_path,
                    max_frames=max_frames_per_camera
                )
                all_stats.append(stats)
        else:
            # ── Parallel mode: all cameras run simultaneously ──────────────
            # Each camera gets its own thread that calls process_source().
            # The shared GlobalPersonGallery and reid_model are already
            # thread-safe (Gallery has a Lock; ReID uses self._reid_lock).

            all_stats = [None] * len(cameras)   # pre-allocate result slots
            stop_event = threading.Event()       # signal all threads to stop

            def _camera_worker(index: int, cam_config: dict):
                """Thread target: process one camera stream."""
                name = cam_config.get('name', f'Camera_{index+1}')
                source = cam_config.get('source')
                output_path = os.path.join(output_dir, f"{name}.mp4")
                print(f"\n🚀 Starting camera thread: {name}")
                try:
                    stats = self.process_source(
                        source=source,
                        camera_name=name,
                        output_path=output_path,
                        max_frames=max_frames_per_camera
                    )
                except Exception as exc:
                    print(f"\n❌ Camera {name} thread error: {exc}")
                    stats = {'error': str(exc), 'camera_name': name}
                all_stats[index] = stats
                print(f"\n✅ Camera thread finished: {name}")

            # Launch one thread per camera
            threads = []
            for i, cam_config in enumerate(cameras):
                t = threading.Thread(
                    target=_camera_worker,
                    args=(i, cam_config),
                    daemon=True,
                    name=f"cam-{cam_config.get('name', i)}"
                )
                threads.append(t)
                t.start()

            print(f"\n🎬 All {len(cameras)} camera threads launched. Press Ctrl+C to stop all.")

            try:
                # Wait for all threads to finish
                for t in threads:
                    while t.is_alive():
                        t.join(timeout=1.0)   # check every second so Ctrl+C works
            except KeyboardInterrupt:
                print("\n\n🛑 Ctrl+C — stopping all camera threads...")
                # process_source() catches KeyboardInterrupt internally;
                # threads will finish their current frame and exit cleanly.
                for t in threads:
                    t.join(timeout=10.0)
                print("✅ All threads stopped.")

        # ── Summary ────────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("📊 MULTI-CAMERA SUMMARY")
        print(f"{'='*60}")
        
        gallery_stats = self.gallery.get_statistics()
        print(f"\n🌐 Global Gallery Statistics:")
        print(f"   Total unique persons: {gallery_stats['total_persons']}")
        print(f"   Cross-camera persons: {gallery_stats['cross_camera_persons']}")
        print(f"   Cameras processed: {gallery_stats['cameras']}")
        
        print(f"\n📹 Per-Camera Statistics:")
        for stats in all_stats:
            if 'error' not in stats:
                print(f"   {stats['camera_name']}:")
                print(f"      Frames: {stats['frames_processed']}")
                print(f"      Unique persons: {stats['unique_persons']}")
                print(f"      Avg FPS: {stats['average_fps']:.1f}")
        
        return all_stats
    
    def reset_gallery(self):
        """Reset the global person gallery (for new session)."""
        self.gallery = GlobalPersonGallery(
            similarity_threshold=self.gallery.similarity_threshold
        )
        print("✅ Global gallery reset")
    
    def get_gallery_info(self) -> dict:
        """Get global gallery information."""
        return self.gallery.get_statistics()


def main():
    """Demo usage"""
    print("=" * 60)
    print("🎯 MULTI-CAMERA PERSON TRACKING SYSTEM")
    print("=" * 60)

    # Use OSNet by default (auto-downloads, no local .pth needed)
    reid_model_path = 'osnet_x1_0'

    # Fall back to custom model if OSNet name is overridden
    if reid_model_path not in ('osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x1_4'):
        if not os.path.exists(reid_model_path):
            print(f"❌ Model not found: {reid_model_path}")
            print("   Use --model osnet_x1_0  or  supply a valid .pth path")
            return
    
    # Initialize tracker
    tracker = MultiCameraTracker(reid_model_path)
    
    print("\n✅ Tracker ready!")
    print("\n" + "=" * 60)
    print("📋 USAGE EXAMPLES")
    print("=" * 60)
    
    print("""
# Process single video file:
tracker.process_source(
    source="video.mp4",
    camera_name="Entrance",
    output_path="output.mp4"
)

# Process RTSP stream:
tracker.process_source(
    source="rtsp://user:pass@192.168.1.100:554/stream",
    camera_name="Lobby",
    output_path="lobby_output.mp4",
    max_frames=500
)

# Process multiple cameras (same person gets same ID!):
tracker.process_multi_camera([
    {"name": "Entrance", "source": "video1.mp4"},
    {"name": "Lobby", "source": "video2.mp4"},
    {"name": "Exit", "source": "video3.mp4"},
], output_dir="outputs/")

# Check global gallery:
print(tracker.get_gallery_info())
    """)
    
    return tracker


if __name__ == '__main__':
    main()
