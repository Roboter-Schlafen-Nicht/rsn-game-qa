"""
End-to-end helper for HELP v2:

1) Generate data_help_v2.yaml that merges original + twin_dataset.
2) Launch a YOLO detect train run.
3) Run a quick predict on a sample image to sanity-check boxes.
"""

from pathlib import Path
import subprocess
import textwrap


# --- CONFIG -------------------------------------------------------------------

# Paths relative to project root
ORIG_DATA_YAML = Path("data.yaml")  # your current config
TWIN_IMG_DIR = Path("data/twin_dataset/images")
TWIN_LBL_DIR = Path("data/twin_dataset/labels")

# New YAML to create
V2_YAML = Path("data_help_v2.yaml")

# Base model to fine-tune from
BASE_MODEL = Path("runs/detect/train12/weights/best.pt")

# Training settings
EPOCHS = 50
IMGSZ = 640
RUN_NAME = "train_help_v2_xpu"

# Optional: one debug image to run predict on after training
DEBUG_IMAGE = None  # e.g. Path("debug_help.png")


# --- HELPERS ------------------------------------------------------------------


def load_orig_data_yaml(path: Path) -> str:
    """Load the original data.yaml as text."""
    if not path.exists():
        raise FileNotFoundError(f"Original data YAML not found: {path}")
    return path.read_text()


def make_v2_yaml(orig_yaml_text: str) -> str:
    """
    Create a v2 YAML text that appends twin_dataset/images to the train set.

    This assumes your original data.yaml uses the standard Ultralytics format, e.g.:

        path: ...
        train: path/to/images
        val: path/to/val

    We simply add data/twin_dataset/images as an extra train source.
    """
    twin_path_str = str(TWIN_IMG_DIR).replace("\\", "/")

    # Very simple approach: we prepend a small note and then override train
    # by appending twin_dataset as a second entry.
    header = textwrap.dedent(
        f"""
        # Auto-generated v2 data config.
        # Original: {ORIG_DATA_YAML}
        # Twin dataset merged from: {TWIN_IMG_DIR}
        """
    ).lstrip()

    # Naive parse: look for the 'train:' line and turn it into a list plus twin_dataset.
    lines = orig_yaml_text.splitlines()
    new_lines = []
    train_handled = False

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("train:") and not train_handled:
            indent = line[: len(line) - len(stripped)]
            # Original train path
            orig_train = stripped.split(":", 1)[1].strip()
            # Build a list form including twin_dataset
            new_lines.append(f"{indent}train:\n")
            new_lines.append(f"{indent}  - {orig_train}\n")
            new_lines.append(f"{indent}  - {twin_path_str}\n")
            train_handled = True
        else:
            new_lines.append(line + "\n")

    return header + "".join(new_lines)


def run_cmd(cmd: list[str]) -> None:
    """Run a shell command and stream output."""
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


# --- MAIN ---------------------------------------------------------------------


def main() -> None:
    # 1) Build data_help_v2.yaml
    if not TWIN_IMG_DIR.exists():
        raise FileNotFoundError(f"Twin dataset images not found: {TWIN_IMG_DIR}")
    if not TWIN_LBL_DIR.exists():
        raise FileNotFoundError(f"Twin dataset labels not found: {TWIN_LBL_DIR}")

    orig_yaml_text = load_orig_data_yaml(ORIG_DATA_YAML)
    v2_yaml_text = make_v2_yaml(orig_yaml_text)
    V2_YAML.write_text(v2_yaml_text, encoding="utf-8")
    print(f"[HELP V2] Wrote merged YAML to {V2_YAML}")

    # 2) Train YOLO v2
    if not BASE_MODEL.exists():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")

    train_cmd = [
        "yolo",
        "detect",
        "train",
        f"model={str(BASE_MODEL)}",
        f"data={str(V2_YAML)}",
        f"epochs={EPOCHS}",
        f"imgsz={IMGSZ}",
        f"name={RUN_NAME}",
    ]
    run_cmd(train_cmd)

    # 3) Quick predict sanity check (optional)
    if DEBUG_IMAGE is not None and Path(DEBUG_IMAGE).exists():
        best_path = Path("runs") / "detect" / RUN_NAME / "weights" / "best.pt"
        if best_path.exists():
            predict_cmd = [
                "yolo",
                "detect",
                "predict",
                f"model={str(best_path)}",
                f"source={str(DEBUG_IMAGE)}",
                f"imgsz={IMGSZ}",
                "conf=0.25",
                "save=True",
            ]
            run_cmd(predict_cmd)
        else:
            print(f"[HELP V2] Warning: best.pt not found at {best_path}, skipping predict.")
    else:
        print("[HELP V2] No DEBUG_IMAGE configured, skipping predict.")


if __name__ == "__main__":
    main()
