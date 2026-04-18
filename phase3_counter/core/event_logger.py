"""
EventLogger — CSV + JSON logging for entry/exit events.

Writes two types of logs to the configured log directory:

  1. entry_exit_log_YYYY-MM-DD.csv
     ─────────────────────────────
     Real-time event log (appended row-by-row as events occur).
     Columns: timestamp, camera_name, door_id, door_name,
              event, track_id, occupancy_after

  2. session_summary_YYYY-MM-DD_HHMMSS.json
     ────────────────────────────────────────
     Written once on clean shutdown (Ctrl+C or end of video).
     Contains aggregate stats: total_entered, total_exited,
     peak_occupancy, per_door breakdown, session duration.

Thread-safe: multiple camera threads can call log_event() concurrently.
Buffered: writes are flushed every `flush_every` events (default 10)
          and always on shutdown.
"""

import os
import csv
import json
import threading
import datetime
from typing import Dict


# CSV column order — changing this breaks existing log files
_CSV_COLUMNS = [
    'timestamp',
    'camera_name',
    'door_id',
    'door_name',
    'event',
    'track_id',
    'occupancy_after',
]


class EventLogger:
    """
    Thread-safe CSV + JSON event logger.

    Args:
        log_dir     (str): Directory for log files. Created if not exists.
        flush_every (int): Flush CSV buffer every N events. Default 10.
    """

    def __init__(self, log_dir: str = 'phase3_counter/logs',
                 flush_every: int = 10):
        self._log_dir    = log_dir
        self._flush_every = flush_every
        self._lock       = threading.Lock()
        self._event_count = 0

        os.makedirs(log_dir, exist_ok=True)

        # Open today's CSV log file (append mode — survives restarts)
        today     = datetime.date.today().strftime('%Y-%m-%d')
        csv_path  = os.path.join(log_dir, f'entry_exit_log_{today}.csv')
        file_exists = os.path.isfile(csv_path)

        self._csv_file   = open(csv_path, 'a', newline='', encoding='utf-8')
        self._csv_writer = csv.DictWriter(self._csv_file,
                                          fieldnames=_CSV_COLUMNS,
                                          extrasaction='ignore')

        # Write header only if this is a brand-new file
        if not file_exists:
            self._csv_writer.writeheader()
            self._csv_file.flush()

        self._csv_path = csv_path
        print(f"📝 Event log: {csv_path}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Public API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def log_event(self, event_dict: Dict) -> None:
        """
        Write one ENTRY or EXIT event to the CSV log.

        Thread-safe — called from camera threads.

        Args:
            event_dict: Dict with keys matching _CSV_COLUMNS.
                        At minimum: timestamp, camera_name, door_id,
                        door_name, event, track_id, occupancy_after.
        """
        with self._lock:
            self._csv_writer.writerow(event_dict)
            self._event_count += 1

            # Flush periodically to ensure data survives a crash
            if self._event_count % self._flush_every == 0:
                self._csv_file.flush()

    def write_session_summary(self, summary: Dict) -> str:
        """
        Write the session summary JSON file on shutdown.

        Args:
            summary: Dict from RoomOccupancyAggregator.get_session_summary()

        Returns:
            Path to the written JSON file.
        """
        with self._lock:
            ts   = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
            path = os.path.join(self._log_dir, f'session_summary_{ts}.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"📊 Session summary saved: {path}")
            return path

    def close(self) -> None:
        """Flush and close the CSV log file. Call on shutdown."""
        with self._lock:
            self._csv_file.flush()
            self._csv_file.close()
            print(f"✅ Event log closed: {self._csv_path} "
                  f"({self._event_count} events logged)")
