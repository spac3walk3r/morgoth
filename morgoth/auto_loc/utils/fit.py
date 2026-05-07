import os
import glob
import shutil
import time

import gbm_drm_gen as drm
import matplotlib.pyplot as plt
import numpy as np
import yaml
from gbm_drm_gen.io.balrog_drm import BALROG_DRM
from threeML import *
from threeML.utils.data_builders.fermi.gbm_data import GBMTTEFile
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion
from threeML.utils.time_series.event_list import EventListWithDeadTime
from morgoth.utils.trig_reader import TrigReader

import gbmgeometry

from morgoth.utils.file_utils import if_dir_containing_file_not_existing_then_make

# Set the global NumPy seed as a precaution
SEED = 12345
np.random.seed(SEED) 

_gbm_detectors = (
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "na",
    "nb",
    "b0",
    "b1",
)

try:
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        time.sleep(rank * 0.5)
    else:
        using_mpi = False
except:
    using_mpi = False
base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")

def _id_to_name(i: int) -> str:
    if i <= 9:
        return f"n{i}"
    if i == 10:
        return "na"
    if i == 11:
        return "nb"
    if i == 12:
        return "b0"
    if i == 13:
        return "b1"
    raise ValueError(f"Unknown detector id: {i}")

class MultinestFitTrigdat(object):
    def __init__(
        self,
        grb_name,
        version,
        trigdat_file,
        bkg_fit_yaml_file,
        time_selection_yaml_file,
    ):
        """
        Initalize MultinestFit for Balrog
        :param grb_name: Name of GRB
        :param version: Version of data
        :param bkg_fit_yaml_file: Path to bkg fit yaml file
        """
        # Basic input
        self._grb_name = grb_name
        self._version = version
        self._bkg_fit_yaml_file = bkg_fit_yaml_file
        self._time_selection_yaml_file = time_selection_yaml_file

        # Load yaml information
        with open(self._bkg_fit_yaml_file, "r") as f:
            data = yaml.safe_load(f)
            self._use_dets = np.array(_gbm_detectors)[np.array(data["use_dets"])]

            self._bkg_fit_files = data["bkg_fit_files"]

        with open(self._time_selection_yaml_file, "r") as f:
            data = yaml.safe_load(f)
            self._active_time = (
                f"{data['active_time']['start']}-{data['active_time']['stop']}"
            )
            self._fine = data["fine"]

        self._trigdat_file = trigdat_file

        self._set_plugins()
        self._define_model()

    def _set_plugins(self):
        """
        Set the plugins using the saved background hdf5 files (trigdat path).
        When drm_backend.kind == "monica", build plugins with MonicaDRMGenTrig;
        otherwise use the classic TrigReader.to_plugin().
        """
        from morgoth.configuration import morgoth_config

        # Restore background polynomials into TrigReader
        success_restore = False
        tries = 0
        while not success_restore:
            try:
                trig_reader = TrigReader(
                    self._trigdat_file,
                    fine=False,  # trigdat is coarse
                    verbose=False,
                    restore_poly_fit=self._bkg_fit_files,
                )
                success_restore = True
                tries = 0
            except Exception:
                time.sleep(1)
                tries += 1
                if tries == 50:
                    raise AssertionError("Can not restore trigdat background fit...")

        # Set active interval from YAML
        active_time = f"{self._active_time_start}-{self._active_time_stop}"
        trig_reader.set_active_time_interval(active_time)

        # Branch: Monica vs classic
        kind = os.getenv("MORGOTH_DRM_BACKEND_KIND", str(morgoth_config["drm_backend"]["kind"]))
        use_monica = kind.strip().lower() == "monica"
        if not use_monica:
            # Classic: let TrigReader build plugins internally (uses DRMGenTrig)
            trig_data = trig_reader.to_plugin(*self._use_dets)
            self._data_list = DataList(*trig_data)
            return

        # Monica branch: build per-detector plugins manually
        from morgoth.monica_backend.drmgen import MonicaDRMGenTrig
        # Pull Monica config
        cfg_node = morgoth_config["drm_backend"]["monica"]
        def cfg_get(key, default=None):
            try:
                v = cfg_node[key]
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    return default
                return v
            except Exception:
                return default
        model_path   = cfg_get("model_path")  # native 70×64; leave empty if using family out-in
        db_path      = cfg_get("db_path")
        nai_in_edges = cfg_get("nai_in_edges")
        bgo_in_edges = cfg_get("bgo_in_edges")
        device       = cfg_get("device", "cpu")
        batch_size   = int(cfg_get("batch_size", 4096))
        missing = [n for n, val in [("db_path", db_path), ("nai_in_edges", nai_in_edges), ("bgo_in_edges", bgo_in_edges)]
                if val is None]
        if missing:
            raise RuntimeError(f"Monica trigdat requires config keys: {', '.join(missing)} in drm_backend.monica")

        det_bl = []
        # Mean of active time for response time stamp (matches your TTE logic)
        rsp_time = (float(self._active_time_start) + float(self._active_time_stop)) / 2.0

        # Note: trig_reader._time_series is a dict of TimeSeriesBuilder per detector (short names "n0".. "b1")
        # We will use those builders to create SpectrumLike plugins and attach Monica responses.
        for det in self._use_dets:
            # Get the per-detector TimeSeriesBuilder object
            ts = trig_reader._time_series[det]

            # Ensure the active interval is set for this builder
            ts.set_active_time_interval(active_time)

            # Build Monica trigdat response for this detector
            rsp = MonicaDRMGenTrig(
                det_name=det,                          # short name "n7"/"b0" is fine; class maps internally
                trigdat_file=self._trigdat_file,
                model_path=str(model_path or ""),      # optional for native; ignored if using family out-in
                db_path=str(db_path),
                nai_in_edges=str(nai_in_edges),
                bgo_in_edges=str(bgo_in_edges),
                device=str(device),
                batch_size=batch_size,
                occult=True,
            )

            # Convert the time series to a SpectrumLike and wrap in a BALROG-like with Monica DRM
            sl = ts.to_spectrumlike()
            bl = drm.BALROGLike.from_spectrumlike(sl, rsp_time, rsp, free_position=True)
            det_bl.append(bl)

        # Package into a DataList
        self._data_list = DataList(*det_bl)

    def _define_model(self, spectrum="cpl"):
        """
        Define a Model for the fit
        :param spectrum: Which spectrum type should be used (cpl, band, pl, sbpl or solar_flare)
        """
        # data_list=comm.bcast(data_list, root=0)
        if spectrum == "cpl":
            # we define the spectral model
            cpl = Cutoff_powerlaw()
            cpl.K.max_value = 10**4
            cpl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
            cpl.xc.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
            cpl.index.set_uninformative_prior(Uniform_prior)
            # we define a point source model using the spectrum we just specified
            self._model = Model(PointSource("GRB_cpl_", 0.0, 0.0, spectral_shape=cpl))

        elif spectrum == "band":
            band = Band()
            band.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1200)
            band.alpha.set_uninformative_prior(Uniform_prior)
            band.xp.prior = Log_uniform_prior(lower_bound=10, upper_bound=1e4)
            band.beta.set_uninformative_prior(Uniform_prior)

            self._model = Model(PointSource("GRB_band", 0.0, 0.0, spectral_shape=band))

        elif spectrum == "pl":
            pl = Powerlaw()
            pl.K.max_value = 10**4
            pl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
            pl.index.set_uninformative_prior(Uniform_prior)
            # we define a point source model using the spectrum we just specified
            self._model = Model(PointSource("GRB_pl", 0.0, 0.0, spectral_shape=pl))

        elif spectrum == "sbpl":
            sbpl = SmoothlyBrokenPowerLaw()
            sbpl.K.min_value = 1e-5
            sbpl.K.max_value = 1e4
            sbpl.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1e4)
            sbpl.alpha.set_uninformative_prior(Uniform_prior)
            sbpl.beta.set_uninformative_prior(Uniform_prior)
            sbpl.break_energy.min_value = 1
            sbpl.break_energy.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
            self._model = Model(PointSource("GRB_sbpl", 0.0, 0.0, spectral_shape=sbpl))

        elif spectrum == "solar_flare":
            # broken powerlaw
            bpl = Broken_powerlaw()
            bpl.K.max_value = 10**5
            bpl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**5)
            bpl.xb.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
            bpl.alpha.set_uninformative_prior(Uniform_prior)
            bpl.beta.set_uninformative_prior(Uniform_prior)

            # thermal brems
            tb = Thermal_bremsstrahlung_optical_thin()
            tb.K.max_value = 1e5
            tb.K.min_value = 1e-5
            tb.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=10**5)
            tb.kT.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e4)
            tb.Epiv.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e4)

            # combined
            total = bpl + tb

            self._model = Model(
                PointSource("Solar_flare", 0.0, 0.0, spectral_shape=total)
            )
        else:
            raise Exception("Use valid model type: cpl, pl, sbpl, band or solar_flare")

    def fit(self):
        """
        Fit the model to data using multinest
        :return:
        """

        # define bayes object with model and data_list
        self._bayes = BayesianAnalysis(self._model, self._data_list)
        # wrap for ra angle
        wrap = [0] * len(self._model.free_parameters)
        wrap[0] = 1

        # define temp chain save path
        self._temp_chains_dir = os.path.join(
            base_dir, self._grb_name, f"c_trig_{self._version}"
        )
        chain_path = os.path.join(self._temp_chains_dir, f"trigdat_{self._version}_")

        # Make temp chains folder if it does not exists already
        if not os.path.exists(self._temp_chains_dir):
            os.mkdir(os.path.join(self._temp_chains_dir))

        # use multinest to sample the posterior
        # set main_path+trigger to whatever you want to use

        self._bayes.set_sampler("multinest", share_spectrum=True)
        self._bayes.sampler.setup(
            n_live_points=500, chain_name=chain_path, wrapped_params=wrap, verbose=True
        )
        self._bayes.sample()

    def save_fit_result(self):
        """
        Save the fits result to '{base_dir}/{grb_name}/{report_type}/{version}/trigdat_{version}_loc_results.fits'
        :return:
        """
        fit_result_name = f"trigdat_{self._version}_loc_results.fits"
        fit_result_path = os.path.join(
            base_dir, self._grb_name, "trigdat", self._version, fit_result_name
        )
        os.makedirs(os.path.dirname(fit_result_path), exist_ok=True)

        if using_mpi:
            if rank == 0:
                self._bayes.restore_median_fit()
                self._bayes.results.write_to(fit_result_path, overwrite=True)

        else:
            self._bayes.restore_median_fit()
            self._bayes.results.write_to(fit_result_path, overwrite=True)

    def move_chains_dir(self):
        """
        Move temp chains directory to sub-folder '{base_dir}/{grb_name}/{report_type}/{version}/chains'
        :return:
        """
        if using_mpi:
            if rank == 0:
                chains_dir_store = os.path.join(
                    base_dir, self._grb_name, "trigdat", self._version, "chains"
                )
                shutil.move(self._temp_chains_dir, chains_dir_store)
        else:
            chains_dir_store = os.path.join(
                base_dir, self._grb_name, "trigdat", self._version, "chains"
            )
            os.makedirs(os.path.dirname(chains_dir_store), exist_ok=True)
            shutil.move(self._temp_chains_dir, chains_dir_store)

    def create_spectrum_plot(self):
        """
        Create the spectral plot to show the fit results for all used dets
        :return:
        """
        plot_name = f"{self._grb_name}_spectrum_plot_trigdat_{self._version}.png"
        plot_path = os.path.join(
            base_dir, self._grb_name, "trigdat", self._version, "plots", plot_name
        )

        color_dict = {
            "n0": "#FF9AA2",
            "n1": "#FFB7B2",
            "n2": "#FFDAC1",
            "n3": "#E2F0CB",
            "n4": "#B5EAD7",
            "n5": "#C7CEEA",
            "n6": "#DF9881",
            "n7": "#FCE2C2",
            "n8": "#B3C8C8",
            "n9": "#DFD8DC",
            "na": "#D2C1CE",
            "nb": "#6CB2D1",
            "b0": "#58949C",
            "b1": "#4F9EC4",
        }

        color_list = []
        for d in self._use_dets:
            color_list.append(color_dict[d])

        set = plt.get_cmap("Set1")
        color_list = set.colors

        if using_mpi:
            if rank == 0:
                if_dir_containing_file_not_existing_then_make(plot_path)

                try:
                    spectrum_plot = display_spectrum_model_counts(
                        self._bayes, data_colors=color_list, model_colors=color_list
                    )
                    ca = spectrum_plot.get_axes()[0]
                    ls = ca.lines
                    max_val = 0
                    for l in ls:
                        if max(l.get_ydata()) > max_val:
                            max_val = sorted(l.get_ydata())[-2]

                    y_lims = ca.get_ylim()
                    if y_lims[0] < 10e-6:
                        ca.set_ylim(bottom=10e-6)
                    if y_lims[1] > 10e6:
                        if max_val <= 10e6:
                            ca.set_ylim(top=max_val * 10)
                        else:
                            ca.set_ylim(top=max_val * 10e2)
                    for c in spectrum_plot.get_axes():
                        c.set_xlim(10, 30000)
                    spectrum_plot.savefig(plot_path, bbox_inches="tight")

                except Exception as e:
                    print(f"No spectral plot possible:\n{e}")

        else:
            if_dir_containing_file_not_existing_then_make(plot_path)

            try:
                spectrum_plot = display_spectrum_model_counts(
                    self._bayes, data_colors=color_list, model_colors=color_list
                )
                ca = spectrum_plot.get_axes()[0]
                ls = ca.lines
                max_val = 0
                for l in ls:
                    if max(l.get_ydata()) > max_val:
                        max_val = sorted(l.get_ydata())[-2]

                y_lims = ca.get_ylim()
                if y_lims[0] < 10e-6:
                    ca.set_ylim(bottom=10e-6)
                if y_lims[1] > 10e6:
                    if max_val <= 10e6:
                        ca.set_ylim(top=max_val * 10)
                    else:
                        ca.set_ylim(top=max_val * 10e2)
                for c in spectrum_plot.get_axes():
                    c.set_xlim(10, 30000)
                spectrum_plot.savefig(plot_path, bbox_inches="tight")
            except Exception as e:
                print(f"No spectral plot plot possible:\n{e}")

class MultinestFitTTE(object):
    def __init__(
        self,
        grb_name,
        version,
        trigdat_file,
        bkg_fit_yaml_file,
        time_selection_yaml_file,
    ):
        """
        Initalize MultinestFit for Balrog
        :param grb_name: Name of GRB
        :param version: Version of data
        :param bkg_fit_yaml_file: Path to bkg fit yaml file
        """
        # Basic input
        self._grb_name = grb_name
        self._version = version
        self._bkg_fit_yaml_file = bkg_fit_yaml_file
        self._time_selection_yaml_file = time_selection_yaml_file

        # Load yaml information
        with open(self._bkg_fit_yaml_file, "r") as f:
            data = yaml.safe_load(f)

            # Map indices to detector names; keep only those with background files present
            raw_use = data.get("use_dets", [])
            def _as_name(x):
                if isinstance(x, int) or (isinstance(x, str) and x.isdigit()):
                    return _id_to_name(int(x))
                return str(x)
            use_names = [_as_name(x) for x in raw_use]

            bkg_files = data.get("bkg_fit_files", {})
            use_names = [d for d in use_names if d in bkg_files]

            self._use_dets = use_names
            self._bkg_fit_files = bkg_files

        with open(self._time_selection_yaml_file, "r") as f:
            data = yaml.safe_load(f)

            self._active_time_start = data["active_time"]["start"]
            self._active_time_stop = data["active_time"]["stop"]

        self._trigdat_file = trigdat_file

        self._set_plugins()
        self._define_model()

    def _resolve_bkg(self):
        """
        Resolve a single background YAML (regardless of backend-specific folders)
        and normalize HDF5 paths so both backends can reuse the same background fits.
        This overrides self._use_dets and self._bkg_fit_files.
        """
        # Candidate YAMLs (in priority order)
        candidates = [
            self._bkg_fit_yaml_file,
            os.path.join(base_dir, self._grb_name, "tte", self._version, "bkg_fit_tte.yml"),
            os.path.join(base_dir, self._grb_name, "tte", "drmgen", "bkg_fit_tte_drmgen.yml"),
            os.path.join(base_dir, self._grb_name, "tte", "monica-nn", "bkg_fit_tte_monica-nn.yml"),
        ]
        data = None
        yaml_path = None
        for p in candidates:
            if p and os.path.isfile(p):
                try:
                    with open(p, "r") as f:
                        data = yaml.safe_load(f)
                    yaml_path = p
                    break
                except Exception:
                    pass
        if data is None:
            raise FileNotFoundError(f"Could not find a background YAML among: {candidates}")

        yaml_dir = os.path.dirname(yaml_path)

        # Normalize detector names
        raw_use = data.get("use_dets", []) or list(data.get("bkg_fit_files", {}).keys())
        def _as_name(x):
            if isinstance(x, int) or (isinstance(x, str) and x.isdigit()):
                return _id_to_name(int(x))
            return str(x)
        use_names = [_as_name(x) for x in raw_use]

        # Normalize HDF5 paths: make absolute and ensure file exists
        in_map = data.get("bkg_fit_files", {})
        resolved_map = {}

        # Candidate directories (in decreasing priority)
        candidate_dirs = [
            yaml_dir,
            os.path.join(base_dir, self._grb_name, "tte", self._version, "bkg_files"),
            os.path.join(base_dir, self._grb_name, "tte", "bkg_files"),
            os.path.join(base_dir, self._grb_name, "tte", "drmgen", "bkg_files"),
            os.path.join(base_dir, self._grb_name, "tte", "monica-nn", "bkg_files"),
        ]

        for det_key, p in in_map.items():
            det = _as_name(det_key)
            # Build path candidates
            candidates_p = []
            if p:
                if os.path.isabs(p):
                    candidates_p.append(p)
                else:
                    candidates_p.append(os.path.normpath(os.path.join(yaml_dir, p)))
            # try by basename in canonical locations
            base = os.path.basename(p) if p else f"bkg_det_{det}.h5"
            for ddir in candidate_dirs:
                candidates_p.append(os.path.join(ddir, base))
            # pick first existing
            resolved = next((pp for pp in candidates_p if os.path.isfile(pp)), None)
            if resolved:
                resolved_map[det] = resolved

        # Keep only dets with a resolved background file
        use_names = [d for d in use_names if d in resolved_map]
        if not use_names:
            # fallback to all resolved keys if use_dets was unusable
            use_names = sorted(resolved_map.keys())
        self._use_dets = use_names
        self._bkg_fit_files = resolved_map

    def _set_plugins(self):
        """
        Set the plugins using the saved background hdf5 files
        :return:
        """
        from morgoth.configuration import morgoth_config

        kind = os.getenv("MORGOTH_DRM_BACKEND_KIND", str(morgoth_config["drm_backend"]["kind"]))
        use_monica = kind.strip().lower() == "monica"
        if use_monica:
            from morgoth.monica_backend.drmgen import MonicaDRMGen
            # Extract config safely (configya Node)
            cfg_node = morgoth_config["drm_backend"]["monica"]
            def cfg_get(key, default=None):
                try:
                    v = cfg_node[key]
                    if v is None or (isinstance(v, str) and v.strip() == ""):
                        return default
                    return v
                except Exception:
                    return default
            model_path      = cfg_get("model_path")
            nai_model       = cfg_get("nai_model")
            nai_side0_model = cfg_get("nai_side0_model")
            nai_side1_model = cfg_get("nai_side1_model")
            bgo00_model     = cfg_get("bgo00_model")
            bgo01_model     = cfg_get("bgo01_model")

            db_path      = cfg_get("db_path")
            nai_in_edges = cfg_get("nai_in_edges")
            bgo_in_edges = cfg_get("bgo_in_edges")
            device       = cfg_get("device", "cpu")
            batch_size   = int(cfg_get("batch_size", 4096))

            missing = [n for n, val in [
                ("db_path", db_path),
                ("nai_in_edges", nai_in_edges),
                ("bgo_in_edges", bgo_in_edges),
            ] if val is None]

            if missing:
                raise RuntimeError(
                    f"Monica backend requires config keys: {', '.join(missing)} in drm_backend.monica"
                )

            any_model = any([
                model_path,
                nai_model,
                nai_side0_model,
                nai_side1_model,
                bgo00_model,
                bgo01_model,
            ])

            if not any_model:
                raise RuntimeError(
                    "Monica backend requires at least one model path in drm_backend.monica "
                    "(model_path or detector/family-specific model entries)"
                )

        def _resolve_gbm_file(datdir, stem):
            versions = ["v03", "v02", "v01", "v00"]
            exts = [".fit", ".fit.gz", ".pha", ".pha.gz", ".rsp2"]
            for v in versions:
                for ext in exts:
                    p = os.path.join(datdir, f"{stem}_{v}{ext}")
                    if os.path.isfile(p):
                        return p
            g = glob.glob(os.path.join(datdir, f"{stem}_v*"))
            return g[0] if g else None

        det_ts = []
        det_rsp = []

        # Resolve background YAML and normalize bkg file paths (backend-agnostic)
        self._resolve_bkg()

        datdir = os.path.join(base_dir, self._grb_name, "tte", "data")
        grb_trig = self._grb_name.replace("GRB", "bn", 1)

        for det in self._use_dets:
            tte_file = _resolve_gbm_file(datdir, f"glg_tte_{det}_{grb_trig}")
            cspec_file = _resolve_gbm_file(datdir, f"glg_cspec_{det}_{grb_trig}")
            if tte_file is None or cspec_file is None:
                raise RuntimeError(f"Missing TTE/CSPEC for {det} in {datdir}")

            if use_monica:
                rsp = MonicaDRMGen(
                    det_name=det,
                    trigdat_file=self._trigdat_file,
                    cspecfile=cspec_file,
                    model_path=str(model_path),
                    db_path=str(db_path),
                    nai_in_edges=str(nai_in_edges),
                    bgo_in_edges=str(bgo_in_edges),
                    device=str(device),
                    batch_size=batch_size,
                )
            else:
                rsp = drm.DRMGenTTE(
                    tte_file=tte_file,
                    trigdat=self._trigdat_file,
                    mat_type=2,
                    cspecfile=cspec_file,
                    occult=True,
                )

            det_rsp.append(rsp)

            # Time Series
            gbm_tte_file = GBMTTEFile(tte_file)

            event_list = EventListWithDeadTime(
                arrival_times=gbm_tte_file.arrival_times - gbm_tte_file.trigger_time,
                measurement=gbm_tte_file.energies,
                n_channels=gbm_tte_file.n_channels,
                start_time=gbm_tte_file.tstart - gbm_tte_file.trigger_time,
                stop_time=gbm_tte_file.tstop - gbm_tte_file.trigger_time,
                dead_time=gbm_tte_file.deadtime,
                first_channel=0,
                instrument=gbm_tte_file.det_name,
                mission=gbm_tte_file.mission,
                verbose=True,
            )

            success_restore = False
            tries = 0
            while not success_restore:
                try:
                    ts = TimeSeriesBuilder(
                        det,
                        event_list,
                        response=BALROG_DRM(rsp, 0.0, 0.0),
                        unbinned=False,
                        verbose=True,
                        container_type=BinnedSpectrumWithDispersion,
                        restore_poly_fit=self._bkg_fit_files.get(det),
                    )

                    success_restore = True
                    tries = 0
                except Exception:
                    time.sleep(1)
                    tries += 1
                    if tries == 50:
                        raise AssertionError("Can not restore background fit...")

            ts.set_active_time_interval(
                f"{self._active_time_start}-{self._active_time_stop}"
            )
            det_ts.append(ts)

        # Mean of active time
        rsp_time = (float(self._active_time_start) + float(self._active_time_stop)) / 2

        # Spectrum Like
        det_sl = []
        # set up energy range
        for series in det_ts:
            if series._name != "b0" and series._name != "b1":
                sl = series.to_spectrumlike()
                sl.set_active_measurements("8.1-700")
                det_sl.append(sl)
            else:
                sl = series.to_spectrumlike()
                sl.set_active_measurements("350-25000")
                det_sl.append(sl)

        # Make Balrog Like
        det_bl = []
        for i, det in enumerate(self._use_dets):
            det_bl.append(
                drm.BALROGLike.from_spectrumlike(
                    det_sl[i], rsp_time, det_rsp[i], free_position=True
                )
            )

        self._data_list = DataList(*det_bl)

    def _define_model(self, spectrum="band"):
        """
        Define a Model for the fit
        :param spectrum: Which spectrum type should be used (cpl, band, pl, sbpl or solar_flare)
        """
        # data_list=comm.bcast(data_list, root=0)
        if spectrum == "cpl":
            # we define the spectral model
            cpl = Cutoff_powerlaw()
            cpl.K.max_value = 10**4
            cpl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
            cpl.xc.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
            cpl.index.set_uninformative_prior(Uniform_prior)
            # we define a point source model using the spectrum we just specified
            self._model = Model(PointSource("GRB_cpl_", 0.0, 0.0, spectral_shape=cpl))

        elif spectrum == "band":
            band = Band()
            band.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1200)
            band.alpha.set_uninformative_prior(Uniform_prior)
            band.xp.prior = Log_uniform_prior(lower_bound=10, upper_bound=1e4)
            band.beta.set_uninformative_prior(Uniform_prior)

            self._model = Model(PointSource("GRB_band", 0.0, 0.0, spectral_shape=band))

        elif spectrum == "pl":
            pl = Powerlaw()
            pl.K.max_value = 10**4
            pl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
            pl.index.set_uninformative_prior(Uniform_prior)
            # we define a point source model using the spectrum we just specified
            self._model = Model(PointSource("GRB_pl", 0.0, 0.0, spectral_shape=pl))

        elif spectrum == "sbpl":
            sbpl = SmoothlyBrokenPowerLaw()
            sbpl.K.min_value = 1e-5
            sbpl.K.max_value = 1e4
            sbpl.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1e4)
            sbpl.alpha.set_uninformative_prior(Uniform_prior)
            sbpl.beta.set_uninformative_prior(Uniform_prior)
            sbpl.break_energy.min_value = 1
            sbpl.break_energy.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
            self._model = Model(PointSource("GRB_sbpl", 0.0, 0.0, spectral_shape=sbpl))

        elif spectrum == "solar_flare":
            # broken powerlaw
            bpl = Broken_powerlaw()
            bpl.K.max_value = 10**5
            bpl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**5)
            bpl.xb.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
            bpl.alpha.set_uninformative_prior(Uniform_prior)
            bpl.beta.set_uninformative_prior(Uniform_prior)

            # thermal brems
            tb = Thermal_bremsstrahlung_optical_thin()
            tb.K.max_value = 1e5
            tb.K.min_value = 1e-5
            tb.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=10**5)
            tb.kT.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e4)
            tb.Epiv.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e4)

            # combined
            total = bpl + tb

            self._model = Model(
                PointSource("Solar_flare", 0.0, 0.0, spectral_shape=total)
            )
        else:
            raise Exception("Use valid model type: cpl, pl, sbpl, band or solar_flare")

    def fit(self):
        """
        Fit the model to data using multinest
        :return:
        """

        # define bayes object with model and data_list
        self._bayes = BayesianAnalysis(self._model, self._data_list)
        # wrap for ra angle
        wrap = [0] * len(self._model.free_parameters)
        wrap[0] = 1

        # define temp chain save path
        self._temp_chains_dir = os.path.join(
            base_dir, self._grb_name, f"c_tte_{self._version}"
        )
        chain_path = os.path.join(self._temp_chains_dir, f"tte_{self._version}_")

        # Make temp chains folder if it does not exist already
        if not os.path.exists(self._temp_chains_dir):
            os.mkdir(os.path.join(self._temp_chains_dir))

        # use multinest to sample the posterior
        # set main_path+trigger to whatever you want to use

        self._bayes.set_sampler("multinest", share_spectrum=True)

        self._bayes.sampler.setup(
            n_live_points=400, chain_name=chain_path, wrapped_params=wrap, verbose=True, seed=SEED
        )
        #self._bayes.sampler.setup(
        #    n_live_points=400, chain_name=chain_path, wrapped_params=wrap, importance_nested_sampling=True, verbose=True, seed=SEED
        #)
        self._bayes.sample()

    def save_fit_result(self):
        """
        Save the fits result to '{base_dir}/{grb_name}/{report_type}/{version}/tte_{version}_loc_results.fits'
        :return:
        """
        fit_result_name = f"tte_{self._version}_loc_results.fits"
        fit_result_path = os.path.join(
            base_dir, self._grb_name, "tte", self._version, fit_result_name
        )
        os.makedirs(os.path.dirname(fit_result_path), exist_ok=True)

        if using_mpi:
            if rank == 0:
                self._bayes.restore_median_fit()
                self._bayes.results.write_to(fit_result_path, overwrite=True)

        else:
            self._bayes.restore_median_fit()
            self._bayes.results.write_to(fit_result_path, overwrite=True)

    def move_chains_dir(self):
        """
        Move temp chains directory to sub-folder '{base_dir}/{grb_name}/{report_type}/{version}/chains'
        :return:
        """
        if using_mpi:
            if rank == 0:
                chains_dir_store = os.path.join(
                    base_dir, self._grb_name, "tte", self._version, "chains"
                )
                shutil.move(self._temp_chains_dir, chains_dir_store)
        else:
            chains_dir_store = os.path.join(
                base_dir, self._grb_name, "tte", self._version, "chains"
            )
            os.makedirs(os.path.dirname(chains_dir_store), exist_ok=True)
            shutil.move(self._temp_chains_dir, chains_dir_store)

    def create_spectrum_plot(self):
        """
        Create the spectral plot to show the fit results for all used dets
        :return:
        """
        plot_name = f"{self._grb_name}_spectrum_plot_tte_{self._version}.png"
        plot_path = os.path.join(
            base_dir, self._grb_name, "tte", self._version, "plots", plot_name
        )

        color_dict = {
            "n0": "#FF9AA2",
            "n1": "#FFB7B2",
            "n2": "#FFDAC1",
            "n3": "#E2F0CB",
            "n4": "#B5EAD7",
            "n5": "#C7CEEA",
            "n6": "#DF9881",
            "n7": "#FCE2C2",
            "n8": "#B3C8C8",
            "n9": "#DFD8DC",
            "na": "#D2C1CE",
            "nb": "#6CB2D1",
            "b0": "#58949C",
            "b1": "#4F9EC4",
        }

        color_list = []
        for d in self._use_dets:
            color_list.append(color_dict[d])

        set = plt.get_cmap("Set1")
        color_list = set.colors

        if using_mpi:
            if rank == 0:
                if_dir_containing_file_not_existing_then_make(plot_path)

                try:
                    spectrum_plot = display_spectrum_model_counts(
                        self._bayes, data_colors=color_list, model_colors=color_list
                    )
                    ca = spectrum_plot.get_axes()[0]
                    ls = ca.lines
                    max_val = 0
                    for l in ls:
                        if max(l.get_ydata()) > max_val:
                            max_val = sorted(l.get_ydata())[-2]

                    y_lims = ca.get_ylim()
                    if y_lims[0] < 10e-6:
                        ca.set_ylim(bottom=10e-6)
                    if y_lims[1] > 10e6:
                        if max_val <= 10e6:
                            ca.set_ylim(top=max_val * 10)
                        else:
                            ca.set_ylim(top=max_val * 10e2)

                    for c in spectrum_plot.get_axes():
                        c.set_xlim(10, 30000)
                    spectrum_plot.savefig(plot_path, bbox_inches="tight")
                except Exception as e:
                    print(f"No spectral plot possible:\n{e}")

        else:
            if_dir_containing_file_not_existing_then_make(plot_path)

            try:
                spectrum_plot = display_spectrum_model_counts(
                    self._bayes, data_colors=color_list, model_colors=color_list
                )
                ca = spectrum_plot.get_axes()[0]
                ls = ca.lines
                max_val = 0
                for l in ls:
                    if max(l.get_ydata()) > max_val:
                        max_val = sorted(l.get_ydata())[-2]

                y_lims = ca.get_ylim()
                if y_lims[0] < 10e-6:
                    ca.set_ylim(bottom=10e-6)
                if y_lims[1] > 10e6:
                    if max_val <= 10e6:
                        ca.set_ylim(top=max_val * 10)
                    else:
                        ca.set_ylim(top=max_val * 10e2)

                for c in spectrum_plot.get_axes():
                    c.set_xlim(10, 30000)
                spectrum_plot.savefig(plot_path, bbox_inches="tight")
            except Exception as e:
                print(f"No spectral plot possible:\n{e}")