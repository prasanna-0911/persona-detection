# 🎯 Persona Detection System
> A complete person detection, re-identification, and tracking system built as an AI/ML internship project.

---

## 📋 Table of Contents
- [Features](#-features)
- [Performance](#-performance)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training](#-training)
- [Technologies](#-technologies)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Real-time Detection** | Detect persons in video frames using YOLOv5 |
| **Re-Identification** | Recognize same person across different camera views |
| **Consistent Tracking** | Maintain unique IDs using DeepSORT algorithm |
| **Occlusion Handling** | Track persons even when temporarily hidden |

---

## 📊 Performance

### Detection (Phase 1)
| Metric | Value |
|--------|-------|
| Speed | 30-50 FPS |
| Confidence | >75% |
| False Positive Rate | <5% |
| Crowded Scenes | ✅ Supported |
| Lighting Variations | ✅ Handled |

### Re-Identification (Phase 2)
| Metric | Value |
|--------|-------|
| Same Person Similarity | 0.912 |
| Different Person Similarity | 0.078 |
| Separation Gap | 0.834 |
| Training Loss | 0.0034 |
| Embedding Dimension | 128 |

### Tracking (Phase 3)
| Metric | Value |
|--------|-------|
| ID Consistency | ✅ Maintained |
| Occlusion Handling | ✅ Supported |
| Real-time Capable | ✅ Yes |

---

## 🏗 Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        TRACKING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Video        YOLOv5         Re-ID         DeepSORT            │
│   Frame  ──►  Detection  ──► Features  ──►  Tracking  ──► Output│
│                                                                 │
│             "Where are      "Who is       "Track ID:42          │
│              people?"        this?"        maintained"          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘```