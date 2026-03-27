from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


@dataclass(frozen=True)
class DetectionSummary:
    source_path: Path
    has_detection: bool
    max_conf: float
    classes: tuple[int, ...]
    is_excluded: bool = False


def load_model(model: str | Path) -> Any:
    # Lazy import so `--help` works before deps are installed.
    from ultralytics import YOLO  # type: ignore

    return YOLO(str(model))


def iter_image_files(source: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if source.is_file():
        if source.suffix.lower() in exts:
            try:
                if source.stat().st_size > 0:
                    yield source
            except (OSError, PermissionError):
                pass
        return
    for p in sorted(source.rglob("*")):
        try:
            if p.is_file() and p.suffix.lower() in exts:
                if p.stat().st_size > 0:
                    yield p
        except (OSError, PermissionError):
            continue


def summarize_detection(
    result,
    conf_threshold: float,
    allowed_classes: Optional[Sequence[int]] = None,
    exclude_classes: Optional[Sequence[int]] = None,
) -> DetectionSummary:
    src = Path(result.path)
    if result.boxes is None or len(result.boxes) == 0:
        return DetectionSummary(source_path=src, has_detection=False, max_conf=0.0, classes=(), is_excluded=False)

    boxes = result.boxes
    confs = boxes.conf.detach().cpu().tolist()
    clss = boxes.cls.detach().cpu().tolist()

    allowed = set(allowed_classes) if allowed_classes is not None else None
    excluded_set = set(exclude_classes) if exclude_classes is not None else None
    
    kept: list[tuple[float, int]] = []
    all_detected_above_conf = set()

    for c, k in zip(confs, clss):
        ci = int(k)
        if c < conf_threshold:
            continue
        
        all_detected_above_conf.add(ci)
        
        if allowed is not None and ci not in allowed:
            continue
        kept.append((float(c), ci))

    # Logical Product: Exact match exclusion
    is_excluded = False
    if excluded_set is not None and all_detected_above_conf == excluded_set:
        is_excluded = True
        kept = []

    if not kept:
        return DetectionSummary(source_path=src, has_detection=False, max_conf=0.0, classes=(), is_excluded=is_excluded)

    max_conf = max(c for c, _ in kept)
    classes = tuple(sorted(set(ci for _, ci in kept)))
    return DetectionSummary(source_path=src, has_detection=True, max_conf=max_conf, classes=classes, is_excluded=False)


def predict_summaries(
    model: Any,
    sources: Iterable[Path],
    conf: float,
    iou: float = 0.7,
    imgsz: int = 960,
    device: Optional[str] = None,
    allowed_classes: Optional[Sequence[int]] = None,
    exclude_classes: Optional[Sequence[int]] = None,
):
    import sys
    for p in sources:
        try:
            results = model.predict(
                source=str(p),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
            # ultralytics returns a list (even for a single image)
            if not results:
                yield DetectionSummary(source_path=p, has_detection=False, max_conf=0.0, classes=(), is_excluded=False)
                continue
            yield summarize_detection(
                results[0],
                conf_threshold=conf,
                allowed_classes=allowed_classes,
                exclude_classes=exclude_classes
            )
        except Exception as e:
            sys.stderr.write(f"Error processing {p}: {e}\n")
            continue

