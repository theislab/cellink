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


def test_from_pretrained_materialized_keeps_non_persistent_buffers(tmp_path):
    """_from_pretrained_materialized gives real tensors for non-persistent buffers and the checkpoint's weights."""
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from cellink.tl.external._scooby import _from_pretrained_materialized

    class _Config:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class _Model(torch.nn.Module):
        config_class = _Config

        def __init__(self, config, width=3):
            super().__init__()
            self.lin = torch.nn.Linear(width, 2)
            self.register_buffer("positions", torch.arange(width, dtype=torch.float32), persistent=False)

    reference = _Model(_Config(), width=3)
    safetensors_torch.save_file(reference.state_dict(), str(tmp_path / "model.safetensors"))

    model = _from_pretrained_materialized(_Model, str(tmp_path), width=3)
    assert not any(t.is_meta for t in [*model.parameters(), *model.buffers()])
    torch.testing.assert_close(model.lin.weight, reference.lin.weight)
    torch.testing.assert_close(model.positions, torch.arange(3, dtype=torch.float32))

    with pytest.raises(ValueError, match="does not match"):
        _from_pretrained_materialized(_Model, str(tmp_path), width=4)


def test_load_scooby_wrapper_bypasses_from_pretrained(tmp_path):
    """_load_scooby_wrapper builds the model from the wrapper's resolved checkpoint settings, without from_pretrained."""
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from cellink.tl.external._scooby import ScoobyRunner, _load_scooby_wrapper

    class _Config:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class _Scooby(torch.nn.Module):
        config_class = _Config

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("from_pretrained must not be used")

        def __init__(self, config, cell_emb_dim, embedding_dim=1920, n_tracks=2, **kwargs):
            super().__init__()
            self.init_kwargs = {"cell_emb_dim": cell_emb_dim, "n_tracks": n_tracks, **kwargs}
            self.lin = torch.nn.Linear(cell_emb_dim, n_tracks)

    class _Wrapper:
        def __init__(self, name, device=None, cell_emb_dim=None, n_tracks=None):
            self.model_name, self.model, self.device = name, None, None
            self.cell_emb_dim, self.n_tracks = cell_emb_dim or 4, n_tracks or 3
            self.embedding_dim, self.use_transform_borzoi_emb = 1920, True

        def load(self, device):
            raise AssertionError("ScoobyWrapper.load must not be used")

    reference = _Scooby(_Config(), cell_emb_dim=4, n_tracks=3)
    safetensors_torch.save_file(reference.state_dict(), str(tmp_path / "model.safetensors"))

    class _Runner(ScoobyRunner):
        def get_scooby_class(self):
            return _Scooby

    wrapper = _load_scooby_wrapper(_Wrapper, str(tmp_path), _Runner(device="cpu"))
    assert wrapper.device == "cpu" and not wrapper.model.training
    assert wrapper.model.init_kwargs["use_transform_borzoi_emb"] is True
    assert wrapper.model.init_kwargs["return_center_bins_only"] is True
    torch.testing.assert_close(wrapper.model.lin.weight, reference.lin.weight)
