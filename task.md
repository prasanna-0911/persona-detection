# Task: YOLO Conversion & Balanced Merge Scripts

## Scripts to Create
- [x] [training_scripts/convert_crowdhuman.py](file:///e:/persona-detection-main/training_scripts/convert_crowdhuman.py) — CrowdHuman .odgt → YOLO format
- [x] [training_scripts/convert_citypersons.py](file:///e:/persona-detection-main/training_scripts/convert_citypersons.py) — CityPersons Cityscapes JSON → YOLO format
- [x] [training_scripts/balanced_merge.py](file:///e:/persona-detection-main/training_scripts/balanced_merge.py) — Epoch-wise rotation balanced sampling & merge

## Also answer: YOLO Compatibility of other datasets
- [x] COCO 2017 → ✅ Ultralytics built-in auto-converter
- [x] NightOwls → ✅ COCO JSON format, same Ultralytics converter works
- [x] EuroCity Persons → ❌ Custom ECP JSON format, needs conversion script (write later)
- [x] WiderPerson → ❌ Custom text format, needs conversion script (write later)
- [x] CrowdHuman → ❌ .odgt format — writing now
- [x] CityPersons → ❌ Cityscapes JSON format — writing now

## Verification
- [x] Scripts written with robust path handling, tqdm progress, and clipping
- [x] YOLO compatibility of all datasets documented
