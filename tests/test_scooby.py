import numpy as np
import pytest

from cellink.tl.external._scooby import (
    KNOWN_SCOOBY_CHECKPOINTS,
    UCSC_TO_ENSEMBL_CHR_MAP,
    ScoobyRunner,
    build_scooby_embedding,
    configure_scooby_runner,
    get_scooby_runner,
)


def test_scooby_runner_device_resolution():
    assert ScoobyRunner(device="cpu").resolve_device() == "cpu"
    assert ScoobyRunner(device="cuda").resolve_device() == "cuda"
    assert ScoobyRunner(device="auto").resolve_device() in {"cpu", "cuda"}


def test_configure_and_get_scooby_runner_singleton():
    runner = configure_scooby_runner(device="cpu")
    assert get_scooby_runner() is runner
    assert isinstance(runner, ScoobyRunner)


def test_known_scooby_checkpoints_registry():
    assert KNOWN_SCOOBY_CHECKPOINTS["lauradmartens/onek1k-scooby"] == {
        "cell_emb_dim": 10,
        "n_tracks": 2,
        "modality": "rna",
        "use_transform_borzoi_emb": True,
    }
    assert all({"cell_emb_dim", "n_tracks", "modality"} <= set(v) for v in KNOWN_SCOOBY_CHECKPOINTS.values())


def test_ucsc_to_ensembl_chr_map():
    assert UCSC_TO_ENSEMBL_CHR_MAP["chr1"] == "1"
    assert UCSC_TO_ENSEMBL_CHR_MAP["chrX"] == "X"
    assert "chrY" not in UCSC_TO_ENSEMBL_CHR_MAP  # not used by this project's real sequences.bed
    assert len(UCSC_TO_ENSEMBL_CHR_MAP) == 23  # chr1-22 + chrX


def _sim_count_adata(n_obs=200, n_vars=50, zero_count_frac=0.0, seed=0):
    """A small, genuinely non-negative raw-count-like AnnData (Poisson
    counts). `build_scooby_embedding` (its `normalize_total`+`log1p`
    pipeline assumes real non-negative counts, exactly like every real
    single-cell h5ad in this project). Optionally zeroes out a fraction of
    cells entirely, to exercise the zero-count-cell handling below with
    realistic, deterministic data rather than relying on `sim_adata()`'s
    unrelated randomness."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    X = rng.poisson(lam=5.0, size=(n_obs, n_vars)).astype(np.float64)
    n_zero = int(n_obs * zero_count_frac)
    if n_zero:
        X[:n_zero] = 0
    return ad.AnnData(X=X)


def test_build_scooby_embedding_shape_and_alignment():
    adata = _sim_count_adata(n_obs=200, n_vars=50)
    n_comps = 8
    emb_df = build_scooby_embedding(adata, n_comps=n_comps, use_hvg=True, n_top_genes=30)

    assert list(emb_df.columns) == ["embedding"]
    assert emb_df.shape[0] == adata.n_obs
    assert (emb_df.index == adata.obs_names).all()
    assert emb_df["embedding"].iloc[0].shape == (n_comps,)
    assert emb_df["embedding"].iloc[0].dtype == np.float32


def test_build_scooby_embedding_no_hvg():
    adata = _sim_count_adata(n_obs=200, n_vars=50)
    emb_df = build_scooby_embedding(adata, n_comps=4, use_hvg=False)
    assert emb_df["embedding"].iloc[0].shape == (4,)


def test_build_scooby_embedding_zero_count_cells_get_zero_vector_not_dropped():
    """Real, not-hypothetical failure mode this function must handle: some
    cells have zero total counts (unfiltered real data)"""
    adata = _sim_count_adata(n_obs=100, n_vars=30, zero_count_frac=0.3, seed=1)
    n_zero = 30
    assert np.ravel(np.asarray(adata.X.sum(axis=1)))[:n_zero].sum() == 0  # sanity-check the fixture itself

    emb_df = build_scooby_embedding(adata, n_comps=6, use_hvg=False)

    assert emb_df.shape[0] == adata.n_obs  # no cells dropped
    zero_embs = np.stack(emb_df["embedding"].iloc[:n_zero].to_numpy())
    assert np.all(zero_embs == 0)
    nonzero_embs = np.stack(emb_df["embedding"].iloc[n_zero:].to_numpy())
    assert not np.all(nonzero_embs == 0)  # real cells still get a real (non-trivial) embedding


def test_scooby_names_raise_friendly_error_without_heavy_deps():
    import cellink.tl.external as ext

    for name in ("train_scooby", "train_scooby_multiome", "score_variant_effects_scooby"):
        try:
            getattr(ext, name)
        except ImportError as e:
            assert "scooby" in str(e) or "embpy" in str(e)


def test_score_variant_effects_scooby_requires_embpy_independently():
    scooby_or_torch_missing = False
    try:
        import scooby  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        scooby_or_torch_missing = True

    if scooby_or_torch_missing:
        pytest.skip("scooby/torch not installed in this environment, can't isolate the embpy-specific error path")

    try:
        import embpy  # noqa: F401

        pytest.skip("embpy is installed in this environment, can't test the missing-embpy error path")
    except ImportError:
        pass

    from cellink.tl.external._scooby import score_variant_effects_scooby

    with pytest.raises(ImportError, match="embpy"):
        score_variant_effects_scooby(
            snp=None, chromosome_sequence="", model_path_or_name="lauradmartens/onek1k-scooby", cell_embeddings=None
        )
