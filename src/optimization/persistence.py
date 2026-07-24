"""Strategy config persistence — save/load ``StrategyConfig`` to/from JSON.

So an AI-optimised config can be saved, reviewed (human-in-the-loop),
and loaded into the live runner with confidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.strategies.base import StrategyConfig


def save_config(config: StrategyConfig, path: str | Path) -> None:
    """Persist a ``StrategyConfig`` to a JSON file.

    Parameters
    ----------
    config : StrategyConfig
        The configuration to serialise.
    path : str or Path
        Destination file path (will be overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(config)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    import logging
    logger = logging.getLogger(__name__)
    logger.info("StrategyConfig saved to %s", path)


def load_config(path: str | Path) -> StrategyConfig:
    """Load a ``StrategyConfig`` from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file created by ``save_config``.

    Returns
    -------
    StrategyConfig
        The deserialised configuration.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the JSON is malformed or missing required fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    required = {"entry_threshold", "exit_threshold", "stop_loss_pct", "take_profit_pct"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(
            f"Config file {path} is missing required fields: {missing}"
        )

    # Build config from the dict, allowing extra fields
    config = StrategyConfig(
        entry_threshold=float(data.get("entry_threshold", 0.5)),
        exit_threshold=float(data.get("exit_threshold", 0.3)),
        max_position_pct=float(data.get("max_position_pct", 0.10)),
        min_position_pct=float(data.get("min_position_pct", 0.01)),
        stop_loss_pct=float(data.get("stop_loss_pct", 0.02)),
        take_profit_pct=float(data.get("take_profit_pct", 0.05)),
        weight=float(data.get("weight", 1.0)),
        extra=data.get("extra", {}),
    )
    return config
