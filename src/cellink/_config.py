"""Find YAML configs, given either by the name of a config shipped with cellink or as a file path."""

from pathlib import Path

import yaml

PACKAGED_CONFIG_DIRS = (
    Path(__file__).parent / "resources" / "config",
    Path(__file__).parent / "tl" / "external" / "config",
)


def packaged_configs() -> dict[str, Path]:
    """Map the name of every config shipped with cellink to its file, e.g. ``"ldsc_docker"``."""
    return {path.stem: path for config_dir in PACKAGED_CONFIG_DIRS for path in sorted(config_dir.glob("*.yaml"))}


def resolve_config_path(config: str | Path) -> Path:
    """
    Return the YAML file a config argument refers to.

    Parameters
    ----------
    config : str or pathlib.Path
        Either the name of a config shipped with cellink (e.g. ``"1000genomes"``,
        ``"ldsc_docker"``, ``"vep_offline"``; see :func:`packaged_configs`), or a path to a
        YAML file. A bare name, i.e. one without a directory and without a ``.yaml`` suffix,
        is looked up among the packaged configs first and then as ``<name>.yaml`` in the
        working directory. Anything else is treated as a path.

    Returns
    -------
    pathlib.Path
        The config file.

    Raises
    ------
    FileNotFoundError
        If ``config`` names neither a packaged config nor an existing file.
    """
    path = Path(config)
    if len(path.parts) == 1 and path.suffix not in (".yaml", ".yml"):
        packaged = packaged_configs()
        if path.name in packaged:
            return packaged[path.name]
        for candidate in (path.with_name(f"{path.name}.yaml"), path):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"{str(config)!r} is neither a config shipped with cellink nor a file in the working directory. "
            f"Shipped configs: {', '.join(sorted(packaged))}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {str(config)!r}")
    return path


def load_config(config: str | Path | None = None, config_dict: dict | None = None) -> dict:
    """Return ``config_dict`` if given, else load the YAML config ``config`` refers to (a shipped name or a path)."""
    if config_dict is not None:
        return config_dict
    with open(resolve_config_path(config)) as f:
        return yaml.safe_load(f)
