"""
Module: root_to_hdf5_runner
---------------------------
This script loads a configuration YAML file and runs the ROOT-to-HDF5
conversion pipeline using `convert_root_to_sparse_h5`.

The workflow:
1. Parse command-line arguments.
2. Load YAML configuration.
3. Invoke conversion routine with parameters from the config.

Typical usage:
    python root_to_hdf5_runner.py --config_path demo_readout.yaml
"""

import argparse
import yaml
from typing import Any, Dict

from reader import convert_root_to_sparse_h5


def load_config(yaml_path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the configuration file is malformed.
    """
    try:
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax in config file: {yaml_path}") from e


def main() -> None:
    """
    Entry point for the ROOT-to-HDF5 conversion process.

    Parses command-line arguments, loads configuration settings,
    and invokes the conversion routine.
    """
    parser = argparse.ArgumentParser(
        description="Convert ROOT hit-maps into sparse HDF5 format using YAML configuration."
    )
    parser.add_argument(
        "--config_path",
        default="demo_readout.yaml",
        help="Path to the YAML configuration file (default: demo_readout.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    # Perform conversion
    convert_root_to_sparse_h5(
        filename=config["filename"],
        treename=config["treename"],
        output_h5=config["output_h5"],
        pixel_size_xy=config["pixel_size_xy"],
        voxel_size_z=config["voxel_size_z"],
        smear_sigma=config["smear_sigma"],
        radius=config["radius"],
        length_z=config["length_z"],
        submap_size_xy=config["submap_size_xy"],
        events_per_group=config["events_per_group"],
    )


if __name__ == "__main__":
    main()