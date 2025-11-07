#!/usr/bin/env python3
"""
medgemma_colposcopy_batch_test.py

Batch-evaluate Google/medgemma-4b-it on a folder with colposcopy images organized in subfolders.

Expected folder structure (default):
/home/phd2/Documenti/colposcopy_data/
    NEG/   -> negative (label 0)
    G1/    -> LSIL (label 1)
    G2/    -> HSIL/cancer (label 2)

Outputs:
- results.csv (one row per image: filepath,true_label,pred_label,raw_output)
- metrics.txt (accuracy, balanced accuracy, kappa, classification report)
- confusion_matrix.png

Usage:
    python3 medgemma_colposcopy_batch_test.py --data_dir /home/phd2/Documenti/colposcopy_data

Dependencies:
    pip install transformers torch pillow huggingface_hub scikit-learn pandas matplotlib tqdm

Notes:
- The script will try to use CUDA if available. If you need to force CPU, pass --device cpu.
- If the HF model requires authentication, set the env var HUGGINGFACE_HUB_TOKEN or run `huggingface_hub login("<TOKEN>")`.

"""

import re
import argparse
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import pipeline
from huggingface_hub import login as hf_login
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score)
import matplotlib.patheffects as pe
from sklearn.metrics import ConfusionMatrixDisplay


def build_pipeline(model_name: str, device: str, dtype=torch.bfloat16):
    # choose torch dtype (use bfloat16 only on CUDA devices that support it)
    device_id = 0 if device.startswith("cuda") else -1
    kwargs = {"model": model_name, "device": device_id}
    
    # set dtype only if on CUDA
    if device.startswith("cuda"):
        kwargs["dtype"] = dtype
    pipe = pipeline(
        "image-text-to-text",
        use_fast=True,
        **kwargs,
    )

    return pipe

def extract_text_from_output(output):
    """
    Robust extractor that handles the shape you showed:
    [ { "generated_text": [ {"role":"system",...}, {"role":"user",...}, {"role":"assistant","content":"2\n"} ] } ]
    and other nested/list/dict shapes.
    Returns a cleaned string ('' if nothing found).
    """
    def walk(obj):
        """Return list of text fragments found in obj."""
        if obj is None:
            return []
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, (int, float, bool)):
            return [str(obj)]
        if isinstance(obj, (list, tuple)):
            parts = []
            for item in obj:
                parts.extend(walk(item))
            return parts
        if isinstance(obj, dict):
            parts = []
            # Priority: if there's a top-level generated_text that's a list of chat turns
            if "generated_text" in obj:
                gen = obj["generated_text"]
                # If generated_text is a list of role/content dicts (chat-like)
                if isinstance(gen, list):
                    # find assistant items first (most likely to contain final response)
                    for turn in gen:
                        if isinstance(turn, dict) and turn.get("role") == "assistant":
                            # content may be string, dict, or list
                            cont = turn.get("content")
                            parts.extend(walk(cont))
                    # if we didn't find assistant content, fall back to walking the whole generated_text
                    if parts:
                        return parts
                    parts.extend(walk(gen))
                    return parts
                else:
                    # handle generated_text being string/dict
                    parts.extend(walk(gen))
            # prioritized common textual keys
            for key in ("text", "answer", "prediction", "output_text", "response", "content"):
                if key in obj:
                    parts.extend(walk(obj[key]))
            # if still empty, recursively walk all values
            if not parts:
                for v in obj.values():
                    parts.extend(walk(v))
            return parts
        # fallback
        return [str(obj)]

    parts = walk(output)
    merged = " ".join([p for p in parts if p])
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def parse_prediction(text: str):
    """
    Robust parse for 0/1/2. Works for '2', '2\\n', '2.', 'Answer: 2', 'assistant: 2', words like 'HSIL',
    and last-resort digit picking.
    """
    if not text:
        return None
    s = text.strip()

    # If the whole cleaned string is exactly a single digit 0-2, return it
    if re.fullmatch(r"[0-2]", s):
        return int(s)

    # common simple patterns: starting digit, digit + punctuation, or a line containing only the digit
    m = re.search(r"\b([0-2])\b", s)
    if m:
        return int(m.group(1))

    # word heuristics
    s_low = s.lower()
    if any(w in s_low for w in ["negative", "neg", "no", "normal"]):
        return 0
    if any(w in s_low for w in ["lsil", "low", "g1", "grade 1", "grade1"]):
        return 1
    if any(w in s_low for w in ["hsil", "high", "g2", "grade 2", "grade2", "cancer"]):
        return 2

    # last resort: last digit 0-2 in string
    digits = re.findall(r"([0-2])", s)
    if digits:
        return int(digits[-1])

    return None


def collect_image_paths(data_dir: Path, allowed_dirs=None):
    img_paths = []
    labels = []
    if allowed_dirs is None:
        allowed_dirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
    for class_dir in allowed_dirs:
        dir_path = data_dir / class_dir
        if not dir_path.exists():
            logging.warning("Class directory not found: %s", dir_path)
            continue
        EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}
        # force lowercase extensions to check
        for ext_name in EXTENSIONS:
            # Generate the patterns for both lowercase and uppercase
            patterns = [f"*.{ext_name}", f"*.{ext_name.upper()}"]
            for pattern in patterns:
                for p in dir_path.glob(pattern):
                    img_paths.append(p)
                    labels.append(class_dir)
    return img_paths, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/phd2/Documenti/colposcopy_data/images_split_only/test",
                        help="Path to data directory containing subfolders (NEG,G1,G2)")
    parser.add_argument("--model", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                        help="Device to run on: cuda or cpu")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (optional). If provided, will call login().")
    parser.add_argument("--out_dir", type=str, default="./medgemma_results",
                        help="Directory to save CSV, metrics, and plots")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # label mapping: adjust if your folder names differ
    dir_to_label = {"NEG": 0, "G1": 1, "G2": 2}

    if args.hf_token:
        try:
            hf_login(args.hf_token)
        except Exception as e:
            logging.warning("HF login failed: %s", e)

    device_str = args.device.lower()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; switching to cpu")
        device_str = "cpu"

    logging.info("Initializing pipeline (model=%s, device=%s)", args.model, device_str)
    pipe = build_pipeline(args.model, device_str)

    allowed_dirs = list(dir_to_label.keys())
    img_paths, dir_labels = collect_image_paths(data_dir, allowed_dirs=allowed_dirs)

    if len(img_paths) == 0:
        logging.error("No images found in %s. Check folder structure and extensions.", data_dir)
        return

    logging.info("Found %d images across %d classes", len(img_paths), len(set(dir_labels)))

    records = []

    for p, class_dir in tqdm(list(zip(img_paths, dir_labels)), desc="Processing images", total=len(img_paths)):
        true_label = dir_to_label.get(class_dir, None)
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            logging.warning("Could not open %s: %s", p, e)
            continue

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert gynecologist."}]},
            {"role": "user", "content": [
                {"type": "text", "text": "Look at this colposcopy image with acetic acid. Classify it as 0 (negative) or 1 (LSIL) or 2 (HSIL/cancer). Respond only with the class."},
                {"type": "image", "image": img}
            ]}
        ]

        out = pipe(text=messages, max_new_tokens=30)   # your existing call is fine
        logging.debug("pipeline repr: %s", repr(out))
        raw_text = extract_text_from_output(out)       # uses the new extractor
        pred = parse_prediction(raw_text)

        records.append({
            "filepath": str(p),
            "class_dir": class_dir,
            "true_label": true_label,
            "pred_label": pred,
            "raw_output": raw_text,
            "raw_output_repr": repr(out),
        })

    df = pd.DataFrame.from_records(records)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    logging.info("Saved per-image results to %s", csv_path)

    # drop rows with None predictions for metric calculation
    df_valid = df[df["pred_label"].notna()].copy()
    if df_valid.empty:
        logging.error("No valid predictions to compute metrics.")
        return

    y_true = df_valid["true_label"].astype(int).to_numpy()
    y_pred = df_valid["pred_label"].astype(int).to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    class_report = classification_report(y_true, y_pred, target_names=["NEG", "G1", "G2"], digits=4)

    # save metrics
    metrics_text = (
        f"n_samples: {len(df_valid)}\n"
        f"accuracy: {acc:.4f}\n"
        f"balanced_accuracy: {bal_acc:.4f}\n"
        f"cohen_kappa: {kappa:.4f}\n"
        f"f1_macro: {f1_macro:.4f}\n\n"
        f"classification_report:\n{class_report}\n"
    )
    (out_dir / "metrics.txt").write_text(metrics_text)
    logging.info("Saved metrics to %s", out_dir / "metrics.txt")

    # plot confusion matrix
    try:
        fig_path = out_dir / "confusion_matrix.png"
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                display_labels=["NEG(0)", "G1(1)", "G2(2)"])
        # Plot with custom parameters
        disp.plot(cmap=plt.cm.viridis, colorbar=True) 
        plt.savefig(fig_path, dpi=200)
        plt.close()
        logging.info("Saved confusion matrix to %s", fig_path)

    except Exception as e:
        logging.warning("Could not plot confusion matrix: %s", e)


    # 2. Plot the matrix using the dedicated class
    

    print("\n--- SUMMARY ---")
    print(metrics_text)
    print(f"Per-image results saved at: {csv_path}")
    print(f"Confusion matrix image (if generated) saved at: {out_dir / 'confusion_matrix.png'}")
    print("--- End ---")


if __name__ == '__main__':
    main()
