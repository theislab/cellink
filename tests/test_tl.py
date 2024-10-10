from pathlib import Path

import pytest

def test_():
    from cellink.tl._simulate_genotype_data import simulate_genotype_data_msprime, simulate_genotype_data_numpy

    simulate_genotype_data_msprime(10, 20)
    geno = simulate_genotype_data_numpy(10, 20)

    from cellink.pp._variant_qc import variant_qc

    variant_qc(geno, inplace=False)

    from cellink.tl._encode_genotype_data import one_hot_encode_genotypes, dosage_per_strand

    one_hot_encode_genotypes(geno)
    dosage_per_strand(geno)
