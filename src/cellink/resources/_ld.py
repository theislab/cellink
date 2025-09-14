import shutil
import tarfile

import pandas as pd

from cellink.resources._utils import _download_file, _load_config, get_data_home


def _extract_or_refresh(tgz_path, extract_path, refresh=False):
    """Extract a tar.gz file if the directory doesn't exist or refresh=True."""
    if refresh and extract_path.exists():
        for item in extract_path.iterdir():
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)

    if not any(p for p in extract_path.iterdir() if p != tgz_path):
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=extract_path)

        contents = list(extract_path.iterdir())
        if len(contents) == 2 and contents[1].is_dir():
            for item in contents[1].iterdir():
                shutil.move(str(item), str(extract_path))
            contents[1].rmdir()


def get_1000genomes_ld_scores(
    config_path="./cellink/resources/config/1000genomes.yaml",
    population="EUR",
    data_home=None,
    return_path=False,
    refresh=False,
):
    """Download and extract precomputed LD scores for a population."""
    data_home = get_data_home(data_home)
    DATA = data_home / f"1000genomes_ld_scores_{population}"
    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)
    if population not in config["ld_scores"]:
        raise ValueError("population must be one of {'EUR', 'EAS'}")

    prefix = config["ld_scores"]["prefix"]
    tgz_path = DATA / config["ld_scores"][population]["filename"]

    _download_file(config["ld_scores"][population]["url"], tgz_path, checksum=None)
    _extract_or_refresh(tgz_path, DATA, refresh=refresh)

    if return_path:
        return DATA, prefix

    annots, ldscores = [], []

    for chrom in range(1, 23):
        annot_file = DATA / f"{prefix}{chrom}.annot.gz"
        if annot_file.exists():
            df_annot = pd.read_csv(annot_file, sep="\t")
            df_annot["chrom"] = chrom
            annots.append(df_annot)

        ld_file = DATA / f"{prefix}{chrom}.l2.ldscore.gz"
        if ld_file.exists():
            df_ld = pd.read_csv(ld_file, sep="\t")
            df_ld["chrom"] = chrom
            ldscores.append(df_ld)

    annot = pd.concat(annots, ignore_index=True)
    ldscores = pd.concat(ldscores, ignore_index=True)

    return annot, ldscores, prefix


def get_1000genomes_ld_weights(
    config_path="./cellink/resources/config/1000genomes.yaml",
    population="EUR",
    data_home=None,
    return_path=False,
    refresh=False,
):
    """Download and extract precomputed LD weights for a population."""
    data_home = get_data_home(data_home)
    DATA = data_home / f"1000genomes_ld_weights_{population}"
    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)
    if population not in config["ld_weights"]:
        raise ValueError("population must be one of {'EUR', 'EAS'}")

    prefix = config["ld_weights"]["prefix"]
    tgz_path = DATA / config["ld_weights"][population]["filename"]

    _download_file(config["ld_weights"][population]["url"], tgz_path, checksum=None)
    _extract_or_refresh(tgz_path, DATA, refresh=refresh)

    if return_path:
        return DATA, prefix

    weights = []

    for chrom in range(1, 23):
        weight_file = DATA / f"{prefix}{chrom}.l2.ldscore.gz"
        if weight_file.exists():
            df_weight = pd.read_csv(weight_file, sep="\t")
            df_weight["chrom"] = chrom
            weights.append(df_weight)

    weights = pd.concat(weights, ignore_index=True)

    return annot, weights


if __name__ == "__main__":
    annot, ldscores, prefix = get_1000genomes_ld_scores(population="EUR")
    annot, ldscores, prefix = get_1000genomes_ld_scores(population="EAS")
    annot, weights, prefix = get_1000genomes_ld_weights(population="EUR")
    annot, weights, prefix = get_1000genomes_ld_weights(population="EAS")
