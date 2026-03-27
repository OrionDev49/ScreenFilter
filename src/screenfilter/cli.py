from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

from .yolo import iter_image_files, load_model, predict_summaries


def cmd_web(args: argparse.Namespace) -> int:
    from .web import start_server

    start_server(host=args.host, port=args.port)
    return 0


def _ensure_dir(p: Path) -> None:

    p.mkdir(parents=True, exist_ok=True)


def _get_processed_paths(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    processed = set()
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "path" in data:
                    processed.add(data["path"])
            except json.JSONDecodeError:
                continue
    return processed


def _resolve_classes(
    model: Any,
    include_specs: Optional[list[str]],
    exclude_specs: Optional[list[str]] = None,
) -> Optional[list[int]]:
    if not include_specs and not exclude_specs:
        return None

    all_names = model.names  # dict: int -> str
    all_ids = set(all_names.keys())

    def parse_spec(specs: list[str]) -> set[int]:
        res = set()
        for s in specs:
            if s.isdigit():
                res.add(int(s))
                continue

            # String name or wildcard
            pattern = s.replace("*", ".*")
            try:
                regex = re.compile(f"^{pattern}$")
                for idx, name in all_names.items():
                    if regex.match(name):
                        res.add(idx)
            except re.error:
                continue
        return res

    included = parse_spec(include_specs) if include_specs else all_ids
    excluded = parse_spec(exclude_specs) if exclude_specs else set()

    final = sorted(list(included - excluded))
    return final


def cmd_train(args: argparse.Namespace) -> int:
    try:
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
    except Exception as e:
        sys.stderr.write(f"Error in train: {e}\n")
        return 1


def cmd_predict(args: argparse.Namespace) -> int:
    try:
        model = load_model(args.model)
        source = Path(args.source)
        sources = list(iter_image_files(source))
        if not sources:
            raise SystemExit(f"No images found under: {source}")

        allowed_classes = _resolve_classes(model, args.classes, None)
        exclude_classes = _resolve_classes(model, args.exclude_classes, None) if args.exclude_classes else None

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
                exclude_classes=exclude_classes,
            ):
                row = {
                    "path": str(s.source_path),
                    "has_detection": s.has_detection,
                    "is_excluded": s.is_excluded,
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
    except Exception as e:
        sys.stderr.write(f"Error in predict: {e}\n")
        return 1


def cmd_collect(args: argparse.Namespace) -> int:
    try:
        model = load_model(args.model)
        allowed_classes = _resolve_classes(model, args.classes, None)
        exclude_classes = _resolve_classes(model, args.exclude_classes, None) if args.exclude_classes else None
        conf = args.conf
        iou = args.iou
        imgsz = args.imgsz
        device = args.device
        overwrite = args.overwrite
        move = getattr(args, "move", False)
        verbose = getattr(args, "verbose", False)

        # Multi-directory mode
        if getattr(args, "src_base", None):
            src_base = Path(args.src_base)
            dst_base = Path(args.dst_base)
            mapping_str = args.map
            
            # Parse mapping table
            mapping = {}
            for line in mapping_str.strip().split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue
                src, dst = line.split(":", 1)
                mapping[src.strip()] = dst.strip()

            print(f"Starting multi-directory collection from {src_base} to {dst_base}...")
            
            date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
            total_copied = 0
            total_kept = 0
            total_images = 0

            for src_name, dst_name in mapping.items():
                src_dir = src_base / src_name
                if not src_dir.exists():
                    print(f"Warning: Source directory {src_dir} does not exist. Skipping.")
                    continue

                # Find all date directories: [Base Directory]/[$Src]/***/[YYYY-MM-DD]
                # Since *** can be deep, we use rglob
                date_dirs = [d for d in src_dir.rglob("*") if d.is_dir() and date_pattern.match(d.name)]
                
                if getattr(args, "date", None):
                    date_dirs = [d for d in date_dirs if d.name == args.date]
                
                for ddir in date_dirs:
                    date_str = ddir.name
                    target_out_dir = dst_base / dst_name / date_str
                    _ensure_dir(target_out_dir)

                    sources = list(iter_image_files(ddir))
                    if not sources:
                        continue
                    
                    total_images += len(sources)
                    log_path = target_out_dir / "collect_log.jsonl"
                    
                    if getattr(args, "resume", False):
                        processed = _get_processed_paths(log_path)
                        if processed:
                            sources = [s for s in sources if str(s) not in processed]
                            if not sources:
                                continue
                    
                    with log_path.open("a", encoding="utf-8") as log_f:
                        for s in predict_summaries(
                            model=model,
                            sources=sources,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            device=device,
                            allowed_classes=allowed_classes,
                            exclude_classes=exclude_classes,
                        ):
                            try:
                                row = {
                                    "path": str(s.source_path),
                                    "has_detection": s.has_detection,
                                    "is_excluded": s.is_excluded,
                                    "max_conf": s.max_conf,
                                    "classes": list(s.classes),
                                }
                                log_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                                if verbose:
                                    print(
                                        f"[collect] {s.source_path.name}: "
                                        f"has_detection={s.has_detection} "
                                        f"is_excluded={s.is_excluded} "
                                        f"max_conf={s.max_conf:.3f} "
                                        f"classes={list(s.classes)}"
                                    )

                                if s.is_excluded or not s.has_detection:
                                    continue
                                
                                total_kept += 1
                                dst_file = target_out_dir / s.source_path.name
                                if dst_file.exists() and not overwrite:
                                    continue
                                
                                if move:
                                    shutil.move(str(s.source_path), str(dst_file))
                                else:
                                    shutil.copy2(s.source_path, dst_file)
                                total_copied += 1
                            except Exception as e:
                                sys.stderr.write(f"Error collecting {s.source_path}: {e}\n")
                                continue

            op_name = "Moved" if move else "Copied"
            print(f"Finished multi-directory collection.")
            print(f"Kept {total_kept}/{total_images} image(s). {op_name} {total_copied} to destination base.")
            return 0

        # Single-directory mode
        source = Path(args.source)
        out_dir = Path(args.out)
        _ensure_dir(out_dir)

        sources = list(iter_image_files(source))
        if not sources:
            raise SystemExit(f"No images found under: {source}")

        log_path = out_dir / "collect_log.jsonl"

        log_mode = "w"
        if getattr(args, "resume", False):
            processed = _get_processed_paths(log_path)
            if processed:
                sources = [s for s in sources if str(s) not in processed]
                if not sources:
                    print(f"All files in {source} already processed. Nothing to do.")
                    return 0
                log_mode = "a"

        copied = 0
        kept = 0
        with log_path.open(log_mode, encoding="utf-8") as log_f:
            for s in predict_summaries(
                model=model,
                sources=sources,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                allowed_classes=allowed_classes,
                exclude_classes=exclude_classes,
            ):
                try:
                    row = {
                        "path": str(s.source_path),
                        "has_detection": s.has_detection,
                        "is_excluded": s.is_excluded,
                        "max_conf": s.max_conf,
                        "classes": list(s.classes),
                    }
                    log_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    if verbose:
                        print(
                            f"[collect] {s.source_path.name}: "
                            f"has_detection={s.has_detection} "
                            f"is_excluded={s.is_excluded} "
                            f"max_conf={s.max_conf:.3f} "
                            f"classes={list(s.classes)}"
                        )

                    if s.is_excluded or not s.has_detection:
                        continue
                    kept += 1
                    dst = out_dir / s.source_path.name
                    if dst.exists() and not overwrite:
                        continue
                    if move:
                        shutil.move(str(s.source_path), str(dst))
                    else:
                        shutil.copy2(s.source_path, dst)
                    copied += 1
                except Exception as e:
                    sys.stderr.write(f"Error collecting {s.source_path}: {e}\n")
                    continue

        op_name = "Moved" if move else "Copied"
        print(f"Kept {kept}/{len(sources)} image(s). {op_name} {copied} to: {out_dir}")
        print(f"Wrote log: {log_path}")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error in collect: {e}\n")
        return 1


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
        nargs="*",
        default=None,
        help="Optional class ids or names to consider (e.g. --classes 0 3 or slack/*).",
    )
    pred.add_argument(
        "--exclude-classes",
        nargs="*",
        default=None,
        help="Optional class ids or names that, if they match the detection exactly, exclude the entire image (Logical Product).",
    )
    pred.add_argument("--out-jsonl", default=None, help="Write per-image detection summary to JSONL.")
    pred.set_defaults(func=cmd_predict)

    col = sub.add_parser("collect", help="Copy only images that contain messenger detections.")
    col.add_argument("--model", required=True)
    col.add_argument("--source", default=None, help="Source folder of screenshots (single directory mode).")
    col.add_argument("--out", default=None, help="Output folder (single directory mode).")
    col.add_argument("--src-base", default=None, help="Source base directory (multi-directory mode).")
    col.add_argument("--dst-base", default=None, help="Destination base directory (multi-directory mode).")
    col.add_argument("--map", default=None, help="Matching table $Src:$Dst (multi-directory mode).")
    col.add_argument("--date", default=None, help="Target date YYYY-MM-DD (multi-directory mode).")
    col.add_argument("--conf", type=float, default=0.25)
    col.add_argument("--iou", type=float, default=0.7)
    col.add_argument("--imgsz", type=int, default=960)
    col.add_argument("--device", default=None)
    col.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class ids or names to consider.",
    )
    col.add_argument(
        "--exclude-classes",
        nargs="*",
        default=None,
        help="Optional class ids or names that, if they match the detection exactly, exclude the entire image (Logical Product).",
    )
    col.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image detection summary (confidence scores, classes).",
    )
    col.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination.")
    col.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    col.add_argument(
        "--resume",
        action="store_true",
        help="Skip already processed files by checking collect_log.jsonl.",
    )
    col.set_defaults(func=cmd_collect)

    web = sub.add_parser("web", help="Start the ScreenFilter Web UI.")
    web.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    web.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    web.set_defaults(func=cmd_web)

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

