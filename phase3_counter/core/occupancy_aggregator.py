"""
RoomOccupancyAggregator — Thread-safe room occupancy counter.

Collects ENTRY / EXIT events pushed by multiple parallel camera threads
and maintains a live, consistent view of room occupancy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stats maintained
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  current_occupancy  — people currently inside the room (floor = 0)
  total_entered      — cumulative entries since session start
  total_exited       — cumulative exits  since session start
  peak_occupancy     — highest occupancy value recorded this session
  peak_time          — wall-clock time when peak was reached
  per_door           — per-door breakdown: {door_id: {entered, exited}}

Occupancy Floor:
  current_occupancy is always ≥ 0.  If exits > entries (can happen when
  the system starts mid-day with people already inside), the count clips
  to 0 rather than going negative.  Use --initial-count N to pre-seed.
"""

import threading
import time
import datetime
from collections import defaultdict
from typing import Dict


class RoomOccupancyAggregator:
    """
    Thread-safe room occupancy aggregator.

    Args:
        initial_count (int): Starting occupancy. Use when the system is
                             launched mid-day with people already inside.
                             Defaults to 0.
    """

    def __init__(self, initial_count: int = 0):
        self._lock = threading.Lock()

        self.current_occupancy: int   = max(0, initial_count)
        self.total_entered:     int   = 0
        self.total_exited:      int   = 0
        self.peak_occupancy:    int   = max(0, initial_count)
        self.peak_time:         float = time.time()
        self.session_start:     float = time.time()

        # Per-door stats: door_id → {entered: int, exited: int}
        self._per_door: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'entered': 0, 'exited': 0}
        )

        # ──────────────────────────────────────────────────────────────────
        # OCCUPANCY ALERT THRESHOLD
        # Uncomment the line below and set your threshold value once
        # the client specifies their room capacity limit:
        #
        # self._alert_threshold = 20   # Alert when room has more than N people
        #
        # The _trigger_alert() method at the bottom of this file handles
        # what happens when the threshold is exceeded.
        # ──────────────────────────────────────────────────────────────────

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Public API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def add_event(self, event_type: str, door_id: str) -> None:
        """
        Record an ENTRY or EXIT event from any camera thread.

        Thread-safe — multiple camera threads can call this concurrently.

        Args:
            event_type (str): "ENTRY" or "EXIT"
            door_id    (str): Door identifier, e.g. "Door_A"
        """
        with self._lock:
            if event_type == 'ENTRY':
                self.current_occupancy += 1
                self.total_entered     += 1
                self._per_door[door_id]['entered'] += 1

                # Track peak occupancy
                if self.current_occupancy > self.peak_occupancy:
                    self.peak_occupancy = self.current_occupancy
                    self.peak_time      = time.time()

            elif event_type == 'EXIT':
                self.current_occupancy -= 1
                self.total_exited      += 1
                self._per_door[door_id]['exited'] += 1

                # Floor at 0 — no negative occupancy
                if self.current_occupancy < 0:
                    self.current_occupancy = 0

            # ──────────────────────────────────────────────────────────
            # Uncomment to enable occupancy alerts:
            # if hasattr(self, '_alert_threshold'):
            #     if self.current_occupancy > self._alert_threshold:
            #         self._trigger_alert()
            # ──────────────────────────────────────────────────────────

    def get_snapshot(self) -> Dict:
        """
        Return a thread-safe snapshot of current occupancy stats.

        Safe to call from any thread at any time.

        Returns:
            Dict with keys: current_occupancy, total_entered, total_exited,
                            peak_occupancy, per_door
        """
        with self._lock:
            return {
                'current_occupancy': self.current_occupancy,
                'total_entered'    : self.total_entered,
                'total_exited'     : self.total_exited,
                'peak_occupancy'   : self.peak_occupancy,
                'per_door'         : {k: dict(v) for k, v in self._per_door.items()},
            }

    def get_session_summary(self) -> Dict:
        """
        Return a full session summary dict for JSON log on shutdown.
        """
        with self._lock:
            elapsed = time.time() - self.session_start
            return {
                'session_start'    : datetime.datetime.fromtimestamp(self.session_start)
                                             .strftime('%Y-%m-%d %H:%M:%S'),
                'session_elapsed_s': round(elapsed, 1),
                'final_occupancy'  : self.current_occupancy,
                'total_entered'    : self.total_entered,
                'total_exited'     : self.total_exited,
                'peak_occupancy'   : self.peak_occupancy,
                'peak_time'        : datetime.datetime.fromtimestamp(self.peak_time)
                                             .strftime('%Y-%m-%d %H:%M:%S'),
                'per_door'         : {k: dict(v) for k, v in self._per_door.items()},
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Alert placeholder
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Uncomment and implement when the alert threshold is decided:
    #
    # def _trigger_alert(self):
    #     """
    #     Fire an occupancy alert.
    #     Called automatically when current_occupancy > _alert_threshold.
    #     Implement the notification mechanism here (print, sound, email, etc.)
    #     NOTE: Called while holding self._lock — keep it fast / non-blocking.
    #     """
    #     print(f"\n🚨  OCCUPANCY ALERT: {self.current_occupancy} people in room "
    #           f"(threshold: {self._alert_threshold})")
    #     # TODO: Add email / sound / REST API call / BMS trigger here
