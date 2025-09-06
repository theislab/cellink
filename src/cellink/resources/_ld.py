import tarfile
import pandas as pd
from cellink.resources._utils import get_data_home, _download_file, _run, _load_config

def get_1000genomes_ld_scores(config_path="./cellink/resources/config/1000genomes.yaml", population="EUR", data_home=None, return_path=False):
    """
    Download and extract precomputed LD scores for Europeans (EUR) or East Asians (EAS).

    Parameters
    ----------
    population : {"EUR", "EAS"}
        Population to download LD scores for.
    data_home : str | Path
        Directory for local cache.
    return_path : bool
        If True, return only the path to the extracted files (for LDSC).
        If False, return loaded pandas objects.

    Returns
    -------
    If return_path=False:
        dict
            {
                "annot": DataFrame (all chromosomes),
                "ldscores": DataFrame (all chromosomes),
                "M": DataFrame with counts per chromosome,
                "M_5_50": DataFrame with counts per chromosome
            }
    If return_path=True:
        pathlib.Path
            Path to directory containing extracted LD files.
    """

    data_home = get_data_home(data_home)
    DATA = data_home / f"1000genomes_ld_{population}"

    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)

    if population not in config["ld"]:
        raise ValueError("population must be one of {'EUR', 'EAS'}")

    tgz_path = DATA / config["ld"][population]["filename"]

    _download_file(config["ld"][population]["url"], tgz_path, checksum=None)

    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=DATA)

    if return_path:
        return DATA

    annots, ldscores, M_vals, M_5_50_vals = [], [], [], []

    for chrom in range(1, 23):
        annot_file = DATA / f"baselineLD.{chrom}.annot.gz"
        if annot_file.exists():
            df_annot = pd.read_csv(annot_file, sep="\t")
            df_annot["chrom"] = chrom
            annots.append(df_annot)

        ld_file = DATA / f"baselineLD.{chrom}.l2.ldscore.gz"
        if ld_file.exists():
            df_ld = pd.read_csv(ld_file, sep="\t")
            df_ld["chrom"] = chrom
            ldscores.append(df_ld)

    annot = pd.concat(annots, ignore_index=True)
    ldscores = pd.concat(ldscores, ignore_index=True)

    return annot, ldscores

if __name__ == "__main__":
    annot, ldscores = get_1000genomes_ld_scores(population="EUR")
    annot, ldscores = get_1000genomes_ld_scores(population="EAS")
