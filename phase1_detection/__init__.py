"""
Phase 1: Person Detection Module

Uses YOLOv5 for real-time person detection.
"""

from .person_detector import PersonDetector, PersonDetectorLite, benchmark_detector

__all__ = ['PersonDetector', 'PersonDetectorLite', 'benchmark_detector']
