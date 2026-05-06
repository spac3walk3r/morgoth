from configya import YAMLConfig

structure = {}

structure["pygcn"] = dict(port=8099)
structure["luigi"] = dict(n_workers=16)
structure["multinest"] = dict(
    n_cores=8, path_to_python="/home/balrog/.environs/test_3.12/bin/python"
)

structure["download"] = dict(
    trigdat=dict(
        v00=dict(interval=5, max_time=1800),
        v01=dict(interval=5, max_time=1800),
        v02=dict(interval=5, max_time=1800),
    ),
    tte=dict(
        v00=dict(interval=5, max_time=7200),
        v01=dict(interval=5, max_time=7200),
        v02=dict(interval=5, max_time=7200),
    ),
    cspec=dict(
        v00=dict(interval=5, max_time=7200),
        v01=dict(interval=5, max_time=7200),
        v02=dict(interval=5, max_time=7200),
    ),
)

structure["upload"] = dict(
    report=dict(interval=2, max_time=1800),
    plot=dict(interval=5, max_time=1800),
    datafile=dict(interval=5, max_time=1800),
)

# DRM backend configuration (classic vs monica)
structure["drm_backend"] = dict(
    kind="classic",   # classic | monica
    monica=dict(
        # Legacy single-path and model_set (kept for backward compatibility)
        model_path="",
        model_set=dict(
            dense256="",
            dense128="",
            bottleneck64="",
        ),
        selected_model="",

        # DB and input edges (still used for native models; for out-in models we use per-family edges below)
        db_path="",
        nai_in_edges="",
        bgo_in_edges="",

        # Per-family/per-detector out-in models and out-edge files
        # Shared NaI model (optional fallback if side-specific not provided)
        nai_model="",            # checkpoint for NaI out-in model (all NaIs)
        # Side-specific NaI models (preferred): NAI_00..NAI_05 use nai_side0_model; NAI_06..NAI_11 use nai_side1_model
        nai_side0_model="",      # checkpoint for NAI_00..NAI_05
        nai_side1_model="",      # checkpoint for NAI_06..NAI_11
        bgo00_model="",          # checkpoint for BGO_00 out-in model
        bgo01_model="",          # checkpoint for BGO_01 out-in model

        # Optional out-edge files (not needed for trigdat; kept for CSPEC/TTE compatibility)
        nai_out_edges="",
        bgo00_out_edges="",
        bgo01_out_edges="",

        # Runtime
        device="cpu",
        batch_size=4096
    )
)

class MorgothConfig(YAMLConfig):
    def __init__(self):
        super(MorgothConfig, self).__init__(
            structure=structure,
            #config_path="~/.morgoth",
            config_path="/home/abacelj/.morgoth",
            config_name="morgoth_config.yml",
        )

morgoth_config = MorgothConfig()
