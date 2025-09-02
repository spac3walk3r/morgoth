from configya import YAMLConfig


structure = {}

structure["pygcn"] = dict(port=8099)
structure["luigi"] = dict(n_workers=16)
structure["multinest"] = dict(
    n_cores=8, path_to_python="/home/balrog/.environs/test_3.9.11.2/bin/python"
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

structure["drm_backend"] = dict(
      kind="classic",   # classic | monica
      monica=dict(
          model_path="",
          db_path="",
          nai_in_edges="",
          bgo_in_edges="",
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
