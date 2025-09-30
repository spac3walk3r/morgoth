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
        nai_model="",           # checkpoint for NaI out-in model
        bgo00_model="",         # checkpoint for BGO_00 out-in model
        bgo01_model="",         # checkpoint for BGO_01 out-in model
        nai_out_edges="",       # npy edges for NaI EBOUNDS
        bgo00_out_edges="",     # npy edges for BGO_00 EBOUNDS
        bgo01_out_edges="",     # npy edges for BGO_01 EBOUNDS

        # Runtime
        device="cpu",
        batch_size=4096
    )
)

class MorgothConfig(YAMLConfig):
    def __init__(self):
        super(MorgothConfig, self).__init__(
            structure=structure,
            config_path="~/.morgoth",
            config_name="morgoth_config.yml",
        )

morgoth_config = MorgothConfig()