from pathlib import Path
from reader import convert_root_to_sparse_h5

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "offline_step" / "out"

TREENAME = "HitNtuple"

COMMON_CFG = {
    "pixel_size_xy": 5,
    "voxel_size_z": 0.1,
    "smear_sigma": 0.0,
    "radius": 500,
    "length_z": 1000,
    "submap_size_xy": 100,
    "events_per_group": 4,
}

def convert_folder(input_dir: Path, prefix: str = "") -> None:
    root_files = sorted(input_dir.glob("*.root"))
    if not root_files:
        print(f"[!] No ROOT files found in {input_dir}")
        return

    print(f"[+] Found {len(root_files)} ROOT files in {input_dir}")

    for fp in root_files:
        stem = fp.stem
        out_name = f"{prefix}{stem}.h5"
        out_path = OUT_DIR / out_name

        print(f"[+] Converting {fp.name} -> {out_path.name}")

        convert_root_to_sparse_h5(
            filename=str(fp),
            treename=TREENAME,
            output_h5=str(out_path),
            **COMMON_CFG
        )

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    convert_folder(DATA_DIR / "others")
    convert_folder(DATA_DIR / "low-p")

    print("[+] Done.")

if __name__ == "__main__":
    main()