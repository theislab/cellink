import numpy as np
import pandas as pd
import dask.array as da

def generate_bim_fam():

    print("khsdgkhs")

def to_plink(dask_genotype_array, bim_df, fam_df, output_prefix):
    """
    Export genotype data in Dask array format to PLINK binary format.

    Parameters:
    - dask_genotype_array: dask.array.Array
        Genotype data (individuals x SNPs). Values must be 0, 1, 2, or np.nan for missing.
    - bim_df: pandas.DataFrame
        DataFrame containing SNP information with columns: ['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'].
    - fam_df: pandas.DataFrame
        DataFrame containing individual information with columns: ['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENOTYPE'].
    - output_prefix: str
        Prefix for the output PLINK files (.bed, .bim, .fam).
    """

    num_individuals, num_snps = dask_genotype_array.shape
    if len(bim_df) != num_snps:
        raise ValueError("Number of SNPs in BIM file does not match genotype matrix.")
    if len(fam_df) != num_individuals:
        raise ValueError("Number of individuals in FAM file does not match genotype matrix.")

    if len(dask_genotype_array.chunks) != 2:
        raise ValueError("Dask array is not 2D. Please ensure the input is (individuals x SNPs).")

    if dask_genotype_array.chunks[0][0] == 1 or dask_genotype_array.chunks[1][0] == 1:
        print("Rechunking genotype array to ensure proper chunk alignment.")
        dask_genotype_array = dask_genotype_array.rechunk((10, 1000))  # Adjust based on your data size ###################

    bim_file = f"{output_prefix}.bim"
    bim_df.to_csv(bim_file, sep="\t", index=False, header=False)

    fam_file = f"{output_prefix}.fam"
    fam_df.to_csv(fam_file, sep="\t", index=False, header=False)

    bed_file = f"{output_prefix}.bed"
    with open(bed_file, "wb") as bed:
        bed.write(bytearray([108, 27, 1]))

        for delayed_chunk_row in dask_genotype_array.to_delayed():
            for delayed_chunk in delayed_chunk_row:
                chunk = delayed_chunk.compute()
                if len(chunk.shape) != 2:
                    raise ValueError(f"Chunk is not 2D. Got shape: {chunk.shape}")

                chunk = np.nan_to_num(chunk, nan=3).astype(np.uint8)

                bed_data = []
                for row in chunk:
                    packed_row = []
                    for i in range(0, len(row), 4):
                        genotypes = row[i:i + 4]
                        byte = 0
                        for j, genotype in enumerate(genotypes):
                            byte |= (genotype << (j * 2))
                        packed_row.append(byte)
                    bed_data.extend(packed_row)

                bed.write(bytearray(bed_data))

    print(f"Exported: {output_prefix}.bed, {output_prefix}.bim, {output_prefix}.fam")

if __name__ == "__main__":
    genotype_data = da.random.randint(0, 3, size=(100, 10000))
    genotype_data = genotype_data.rechunk((10, 10000))


    bim_data = pd.DataFrame({
        "CHR": [1] * 10000,
        "SNP": [f"rs{i}" for i in range(1, 10001)],
        "CM": [0] * 10000,
        "BP": range(1, 10001),
        "A1": ["A"] * 10000,
        "A2": ["G"] * 10000,
    })

    fam_data = pd.DataFrame({
        "FID": [f"F{i}" for i in range(1, 101)],
        "IID": [f"I{i}" for i in range(1, 101)],
        "PID": [0] * 100,
        "MID": [0] * 100,
        "SEX": [0] * 100,
        "PHENOTYPE": [-9] * 100,
    })

    export_to_plink(genotype_data, bim_data, fam_data, "output/genotype")


