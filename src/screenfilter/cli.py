from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

from .yolo import iter_image_files, load_model, predict_summaries


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def cmd_train(args: argparse.Namespace) -> int:
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(args.model))
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        weight_decay=args.weight_decay,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    if args.resume:
        train_kwargs["resume"] = True
    model.train(**train_kwargs)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    source = Path(args.source)
    sources = list(iter_image_files(source))
    if not sources:
        raise SystemExit(f"No images found under: {source}")

    allowed_classes = args.classes if args.classes else None

    out_jsonl: Optional[Path] = Path(args.out_jsonl) if args.out_jsonl else None
    if out_jsonl is not None:
        _ensure_dir(out_jsonl.parent)

    rows = 0
    with (out_jsonl.open("w", encoding="utf-8") if out_jsonl else _nullcontext()) as f:
        for s in predict_summaries(
            model=model,
            sources=sources,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            allowed_classes=allowed_classes,
        ):
            row = {
                "path": str(s.source_path),
                "has_detection": s.has_detection,
                "max_conf": s.max_conf,
                "classes": list(s.classes),
            }
            if f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

    print(f"Processed {rows} image(s).")
    if out_jsonl:
        print(f"Wrote: {out_jsonl}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    source = Path(args.source)
    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    sources = list(iter_image_files(source))
    if not sources:
        raise SystemExit(f"No images found under: {source}")

    allowed_classes = args.classes if args.classes else None

    log_path = out_dir / "collect_log.jsonl"

    copied = 0
    kept = 0
    with log_path.open("w", encoding="utf-8") as log_f:
        for s in predict_summaries(
            model=model,
            sources=sources,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            allowed_classes=allowed_classes,
        ):
            row = {
                "path": str(s.source_path),
                "has_detection": s.has_detection,
                "max_conf": s.max_conf,
                "classes": list(s.classes),
            }
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if getattr(args, "verbose", False):
                print(
                    f"[collect] {s.source_path.name}: "
                    f"has_detection={s.has_detection} "
                    f"max_conf={s.max_conf:.3f} "
                    f"classes={list(s.classes)}"
                )

            if not s.has_detection:
                continue
            kept += 1
            dst = out_dir / s.source_path.name
            if dst.exists() and not args.overwrite:
                continue
            if getattr(args, "move", False):
                shutil.move(str(s.source_path), str(dst))
            else:
                shutil.copy2(s.source_path, dst)
            copied += 1

    op_name = "Moved" if getattr(args, "move", False) else "Copied"
    print(f"Kept {kept}/{len(sources)} image(s). {op_name} {copied} to: {out_dir}")
    print(f"Wrote log: {log_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="screenfilter")
    sub = p.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Train YOLOv8 on your dataset.")
    train.add_argument("--data", required=True, help="Dataset YAML (e.g. configs/messengers.yaml).")
    train.add_argument("--model", required=True, help="YOLOv8 model/weights (local path or model id).")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=100, help="Epochs to wait for no observable improvement for early stopping of training.")
    train.add_argument("--imgsz", type=int, default=960)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--weight-decay", type=float, default=0.0005, help="L2 regularization term.")
    train.add_argument("--device", default=None, help='e.g. "cpu", "0", "0,1"')
    train.add_argument("--project", default="runs/detect")
    train.add_argument("--name", default="train")
    train.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from checkpoint. Set --model to weights/last.pt from the same run (e.g. runs/detect/runs/detect/train/weights/last.pt).",
    )
    train.set_defaults(func=cmd_train)

    pred = sub.add_parser("predict", help="Run inference and optionally write JSONL summaries.")
    pred.add_argument("--model", required=True, help="Trained weights (e.g. runs/detect/train/weights/best.pt).")
    pred.add_argument("--source", required=True, help="Image file or folder of screenshots.")
    pred.add_argument("--conf", type=float, default=0.25)
    pred.add_argument("--iou", type=float, default=0.7)
    pred.add_argument("--imgsz", type=int, default=960)
    pred.add_argument("--device", default=None)
    pred.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Optional class ids to consider (e.g. --classes 0 3 for whatsapp+slack).",
    )
    pred.add_argument("--out-jsonl", default=None, help="Write per-image detection summary to JSONL.")
    pred.set_defaults(func=cmd_predict)

    col = sub.add_parser("collect", help="Copy only images that contain messenger detections.")
    col.add_argument("--model", required=True)
    col.add_argument("--source", required=True)
    col.add_argument("--out", required=True, help="Output folder to copy kept screenshots into.")
    col.add_argument("--conf", type=float, default=0.25)
    col.add_argument("--iou", type=float, default=0.7)
    col.add_argument("--imgsz", type=int, default=960)
    col.add_argument("--device", default=None)
    col.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Optional class ids to consider.",
    )
    col.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image detection summary (confidence scores, classes).",
    )
    col.add_argument("--overwrite", action="store_true")
    col.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    col.set_defaults(func=cmd_collect)

    return p


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

