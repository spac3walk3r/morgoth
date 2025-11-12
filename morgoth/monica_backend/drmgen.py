import os
import time
import atexit
import csv
import numpy as np
import torch
from astropy.coordinates import SkyCoord
import astropy.units as u
from gbmgeometry.gbm_frame import GBMFrame
import gbmgeometry

from drm_monica.remap import remap_70to_outin, build_remap_precompute, remap_apply_precomputed
from drm_monica.db_reader import load_energy_axes, load_atm_grid_info
from drm_monica.io.cspec import read_cspec_out_edges

from gbm_drm_gen.matrix_functions import geocoords
from gbm_drm_gen.utils.geometry import is_occulted

try:
    from morgoth.configuration import morgoth_config
except Exception:
    morgoth_config = None

def cfg_get(node, key, default=""):
    try:
        v = node[key]
        if v is None:
            return default
        s = str(v).strip()
        return s if s else default
    except Exception:
        return default

DET_ORIENT_DEG = {
    "NAI_00": (45.89, 20.58), "NAI_01": (45.11, 45.31), "NAI_02": (58.44, 90.21),
    "NAI_03": (314.87, 45.24), "NAI_04": (303.15, 90.27), "NAI_05": (3.35, 89.97),
    "NAI_06": (224.93, 20.43), "NAI_07": (224.62, 46.18), "NAI_08": (236.61, 89.97),
    "NAI_09": (135.19, 45.55), "NAI_10": (123.73, 90.42), "NAI_11": (183.74, 90.32),
    "BGO_00": (0.00, 90.00),   "BGO_01": (180.00, 90.00),
}

def _short_to_long(det_short: str) -> str:
    s = det_short.lower()
    if s.startswith("n"):
        if s == "na":
            return "NAI_10"
        if s == "nb":
            return "NAI_11"
        idx = int(s[1])
        return f"NAI_0{idx}"
    if s == "b0":
        return "BGO_00"
    if s == "b1":
        return "BGO_01"
    raise ValueError(f"Unknown detector short name: {det_short}")

def _long_to_group(long_name: str) -> str:
    if long_name.startswith("NAI_"):
        idx = int(long_name.split("_")[1])
        return f"n{idx}" if idx < 10 else ("na" if idx == 10 else "nb")
    if long_name == "BGO_00":
        return "b0"
    if long_name == "BGO_01":
        return "b1"
    raise ValueError(f"Cannot map to DB group: {long_name}")

def _azel_to_unit(az_deg: float, el_deg: float) -> np.ndarray:
    a = np.deg2rad(float(az_deg))
    e = np.deg2rad(float(el_deg))
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], dtype=np.float64)

def _det_normal(long_name: str) -> np.ndarray:
    az, zen = DET_ORIENT_DEG[long_name]
    el = 90.0 - zen
    return _azel_to_unit(az, el)

def _nearest_index(val: float, centers: np.ndarray) -> int:
    return int(np.argmin(np.abs(centers - val)))

def _get_mpi_rank() -> str:
    for k in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "MPI_RANKID", "SLURM_PROCID"):
        v = os.getenv(k)
        if v is not None:
            return v
    return "0"

class _MonicaNetNative(torch.nn.Module):
    def __init__(self, in_dim: int, det_count: int, hidden_dims=(256,256,256), det_emb_dim: int = 8):
        super().__init__()
        self.emb = torch.nn.Embedding(det_count, det_emb_dim) if det_count > 1 else None
        feat_in = in_dim + (det_emb_dim if self.emb is not None else 0)
        h1, h2, h3 = hidden_dims
        self.fc1 = torch.nn.Linear(feat_in, h1)
        self.fc2 = torch.nn.Linear(h1, h2)
        self.fc3 = torch.nn.Linear(h2, h3)
        self.act = torch.nn.ReLU()
        self.eff = torch.nn.Linear(h3, 70)
        self.shp = torch.nn.Linear(h3, 70 * 64)
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim=-1)
    @torch.no_grad()
    def forward(self, x, det_id=None):
        if self.emb is not None and det_id is not None:
            x = torch.cat([x, self.emb(det_id)], dim=-1)
        h = self.act(self.fc1(x)); h = self.act(self.fc2(h)); h = self.act(self.fc3(h))
        eff = self.softplus(self.eff(h))
        shp = self.softmax(self.shp(h).view(-1, 70, 64))
        return eff.unsqueeze(-1) * shp  # [B,70,64]

class _MonicaNetOutIn(torch.nn.Module):
    def __init__(self, in_dim: int, det_count: int, n_out: int, n_in: int,
                 hidden_dims=(256,256,256), det_emb_dim: int = 8):
        super().__init__()
        self.emb = torch.nn.Embedding(det_count, det_emb_dim) if det_count > 1 else None
        feat_in = in_dim + (det_emb_dim if self.emb is not None else 0)
        h1, h2, h3 = hidden_dims
        self.fc1 = torch.nn.Linear(feat_in, h1)
        self.fc2 = torch.nn.Linear(h1, h2)
        self.fc3 = torch.nn.Linear(h2, h3)
        self.act = torch.nn.ReLU()
        self.out_lin = torch.nn.Linear(h3, n_out * n_in)
        self.n_out = n_out; self.n_in = n_in
        self.softplus = torch.nn.Softplus()
    @torch.no_grad()
    def forward(self, x, det_id=None):
        if self.emb is not None and det_id is not None:
            x = torch.cat([x, self.emb(det_id)], dim=-1)
        h = self.act(self.fc1(x)); h = self.act(self.fc2(h)); h = self.act(self.fc3(h))
        y = self.softplus(self.out_lin(h))
        return y.view(-1, self.n_out, self.n_in)

class _MonicaNetOutInLowRank(torch.nn.Module):
    """
    Low-rank out-in head: y = Softplus( A (B h) ), B: h3->R, A: R->(n_out*n_in).
    """
    def __init__(self, in_dim: int, det_count: int, n_out: int, n_in: int,
                 hidden_dims=(256,256,128), rank: int = 96, det_emb_dim: int = 8):
        super().__init__()
        assert rank > 0
        self.emb = torch.nn.Embedding(det_count, det_emb_dim) if det_count > 1 else None
        feat_in = in_dim + (det_emb_dim if self.emb is not None else 0)
        h1, h2, h3 = hidden_dims
        self.fc1 = torch.nn.Linear(feat_in, h1)
        self.fc2 = torch.nn.Linear(h1, h2)
        self.fc3 = torch.nn.Linear(h2, h3)
        self.act = torch.nn.ReLU()
        self.B = torch.nn.Linear(h3, rank, bias=False)
        self.A = torch.nn.Linear(rank, n_out * n_in, bias=True)
        self.n_out = n_out; self.n_in = n_in
        self.softplus = torch.nn.Softplus()
    @torch.no_grad()
    def forward(self, x, det_id=None):
        if self.emb is not None and det_id is not None:
            x = torch.cat([x, self.emb(det_id)], dim=-1)
        h = self.act(self.fc1(x)); h = self.act(self.fc2(h)); h = self.act(self.fc3(h))
        z = self.B(h)
        y = self.softplus(self.A(z))
        return y.view(-1, self.n_out, self.n_in)

class _ModelStore:
    def __init__(self):
        self._store = {}  # key -> (model, det_to_idx, cfg)
    def _key(self, arch_tag: str, ckpt_path: str) -> tuple:
        return (arch_tag, os.path.abspath(ckpt_path))
    def get(self, arch_tag: str, ckpt_path: str):
        return self._store.get(self._key(arch_tag, ckpt_path))
    def put(self, arch_tag: str, ckpt_path: str, model, det_to_idx, cfg):
        self._store[self._key(arch_tag, ckpt_path)] = (model, det_to_idx, cfg)

_MODEL_STORE = _ModelStore()

class _BatchGroup:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.instances = []          # list[MonicaDRMGen]
        self._last_key = None        # (pose_id tuple, az_deg, el_deg)
        self._last_outputs = None    # list of per-instance matrices (R)
    def add(self, inst):
        if inst not in self.instances:
            self.instances.append(inst)
    @torch.no_grad()
    def predict_all(self, az_deg: float, el_deg: float):
        # Build cache key that includes pose_id tuple so time changes invalidate cache
        key = (tuple(inst._pose_id for inst in self.instances), float(az_deg), float(el_deg))
        if self._last_key is not None and key == self._last_key and self._last_outputs is not None:
            for inst, R in zip(self.instances, self._last_outputs):
                inst._matrix = R
                inst._n_calls += 1
            return
        # Occult: test once (az/el are scalars here)
        inst0 = self.instances[0]
        occulted = inst0._occult and is_occulted(float(az_deg), float(el_deg), inst0._scpos)
        if occulted:
            outs = []
            for inst in self.instances:
                Nout = len(inst._out_edges) - 1
                Nin  = len(inst._in_edges) - 1
                R = np.zeros((Nout, Nin), dtype=np.float64, order="C")
                inst._matrix = R
                inst._n_calls += 1
                outs.append(R)
            self._last_key = key
            self._last_outputs = outs
            return
        # Build features and det_ids
        feats = []
        det_ids = []
        t_feat0 = time.perf_counter()
        for inst in self.instances:
            x = inst._build_features(float(az_deg), float(el_deg)).astype(np.float32)
            feats.append(x)
            det_ids.append(inst._det_id if inst._det_id is not None else 0)
        t_feat = time.perf_counter() - t_feat0
        X = torch.from_numpy(np.stack(feats, axis=0))
        D = torch.tensor(det_ids, dtype=torch.long) if getattr(self.model, "emb", None) is not None else None
        # Forward once
        t_model0 = time.perf_counter()
        Y = self.model(X, D)
        t_model = time.perf_counter() - t_model0
        outs = []
        for bi, inst in enumerate(self.instances):
            if inst._target_mode == "outin":
                R = Y[bi].cpu().numpy().astype(np.float64, copy=False)  # (N_out,N_in)
                inst._matrix = np.ascontiguousarray(R, dtype=np.float64)
                inst._t_feat += t_feat / max(len(self.instances), 1)
                inst._t_model += t_model / max(len(self.instances), 1)
                inst._n_calls += 1
                outs.append(inst._matrix)
            else:
                M70 = Y[bi].cpu().numpy().astype(np.float64, copy=False)  # (70,64)
                t_remap0 = time.perf_counter()
                if inst._pre is not None:
                    M_inout = remap_apply_precomputed(M70, inst._pre)
                    R = M_inout.T
                else:
                    R = remap_70to_outin(inst._in_edges, inst._out_edges, inst._e_in, inst._epx_lo, inst._epx_hi, M70)
                t_remap = time.perf_counter() - t_remap0
                inst._matrix = np.ascontiguousarray(R, dtype=np.float64)
                inst._t_feat += t_feat / max(len(self.instances), 1)
                inst._t_model += t_model / max(len(self.instances), 1)
                inst._t_remap += t_remap
                inst._n_calls += 1
                outs.append(inst._matrix)
        self._last_key = key
        self._last_outputs = outs

class _Batcher:
    def __init__(self):
        self.groups = {}  # key=id(shared_model) -> _BatchGroup
    def register(self, inst):
        key = id(inst._model)
        grp = self.groups.get(key)
        if grp is None:
            grp = _BatchGroup(inst._model)
            self.groups[key] = grp
        grp.add(inst)
    def run(self, inst, az_deg: float, el_deg: float):
        grp = self.groups.get(id(inst._model))
        if grp is None:
            return inst._single_forward(az_deg, el_deg)
        return grp.predict_all(az_deg, el_deg)

_BATCHER = _Batcher()

class MonicaDRMGen:

    def __init__(self, det_name, trigdat_file, cspecfile, model_path,
                 db_path, nai_in_edges, bgo_in_edges, device="cpu", batch_size=4096, occult=True):
        self.det_short = det_name
        self.det_long = _short_to_long(det_name)
        self.det_group = _long_to_group(self.det_long)
        self.trigdat_file = trigdat_file
        self.cspecfile = cspecfile
        self.model_path_arg = model_path
        self.db_path = db_path
        self.nai_in_edges_path_arg = nai_in_edges
        self.bgo_in_edges_path_arg = bgo_in_edges
        self.device = device
        self.batch_size = int(batch_size)
        self._occult = bool(occult)

        self._initialized = False
        self._matrix = None

        self._in_edges = None
        self._out_edges = None

        self._pose_id = 0

        self._t_feat = 0.0
        self._t_model = 0.0
        self._t_remap = 0.0
        self._n_calls = 0
        self._profile = (os.getenv("MONICA_PROFILE", "0") == "1")
        self._profile_csv = os.getenv("MONICA_PROFILE_CSV", "").strip()
        self._profile_csv_dir = os.getenv("MONICA_PROFILE_CSV_DIR", "").strip()
        self._run_tag = os.getenv("MONICA_RUN_TAG", "").strip()
        self._model_label_env = os.getenv("MONICA_MODEL_LABEL", "").strip()
        self._rank = _get_mpi_rank()
        base = os.path.basename(self.trigdat_file)
        self._bn = ""
        for tok in base.replace(".", "_").split("_"):
            if tok.startswith("bn") and len(tok) >= 11:
                self._bn = tok
                break

        self._e_in = None
        self._epx_lo = None
        self._epx_hi = None
        self._pre = None

        self._model = None
        self._target_mode = None
        self._det_to_idx = None
        self._det_id = None
        self._det_id_t = None

        self._det_n = None
        self._pos = None
        self._quat = None
        self._scpos = None
        self._theta_cent = None
        self._lat_cent = None
        self._phi_cent = None

    def _lazy_init(self):
        if self._initialized:
            return

        # Torch threads (per process)
        try:
            th = int(os.getenv("MONICA_THREADS", "0")) or None
        except Exception:
            th = None
        if th is not None and th > 0:
            try:
                torch.set_num_threads(th)
            except Exception:
                pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        cfgm = None
        if morgoth_config is not None:
            try:
                cfgm = morgoth_config["drm_backend"]["monica"]
            except Exception:
                cfgm = None

        # Geometry helpers
        self._det_n = _det_normal(self.det_long)
        self._pos = gbmgeometry.PositionInterpolator.from_trigdat(trigdat_file=self.trigdat_file)
        self._update_sc_pose(0.0)

        th_edge, lat_edge, phi_edge = load_atm_grid_info(self.db_path, self.det_group)
        self._theta_cent = 0.5 * (th_edge[:-1] + th_edge[1:])
        self._lat_cent   = 0.5 * (lat_edge[:-1] + lat_edge[1:])
        self._phi_cent   = 0.5 * (phi_edge[:-1] + phi_edge[1:])

        # Resolve out-in model paths and edges per detector
        ckpt_outin, in_edges_path, out_edges_path = (None, None, None)
        try:
            if self.det_long.startswith("NAI_"):
                # Side-specific selection: NAI_00..NAI_05 -> side0, NAI_06..NAI_11 -> side1
                idx = int(self.det_long.split("_")[1])  # 0..11
                if idx <= 5:
                    ckpt_outin = cfg_get(cfgm, "nai_side0_model") or cfg_get(cfgm, "nai_model")
                else:
                    ckpt_outin = cfg_get(cfgm, "nai_side1_model") or cfg_get(cfgm, "nai_model")
                # Input edges for NaIs
                in_edges_path = cfg_get(cfgm, "nai_in_edges")
                # Trigdat out edges are fixed; we already set self._out_edges via get_trigdat_out_edges
                out_edges_path = None
            elif self.det_long == "BGO_00":
                ckpt_outin = cfg_get(cfgm, "bgo00_model")
                in_edges_path = cfg_get(cfgm, "bgo_in_edges")
                out_edges_path = None
            elif self.det_long == "BGO_01":
                ckpt_outin = cfg_get(cfgm, "bgo01_model")
                in_edges_path = cfg_get(cfgm, "bgo_in_edges")
                out_edges_path = None
        except Exception:
            pass
        use_outin = bool(ckpt_outin and os.path.isfile(ckpt_outin))

        if use_outin:
            # Load edges for this detector
            self._in_edges = np.load(in_edges_path).astype(np.float64)
            self._out_edges = np.load(out_edges_path).astype(np.float64)

            # Load checkpoint, config, and build the right head
            ckpt = torch.load(ckpt_outin, map_location="cpu")
            cfg = ckpt.get("config", {})
            in_dim = int(cfg.get("in_dim", 12))
            n_in  = int(cfg.get("n_in", len(self._in_edges) - 1))
            n_out = int(cfg.get("n_out", len(self._out_edges) - 1))
            hidden_dims = tuple(int(x) for x in cfg.get("hidden_dims", [256, 256, 256]))
            lowrank = bool(cfg.get("lowrank", False))
            rank = int(cfg.get("rank", 0) or 96)

            det_to_idx = ckpt.get("det_to_idx", None)
            if det_to_idx is None:
                names = [f"NAI_0{i}" for i in range(10)] + ["NAI_10", "NAI_11", "BGO_00", "BGO_01"]
                det_to_idx = {n: i for i, n in enumerate(names)}

            arch_tag = "outin_lowrank" if lowrank else "outin_dense"
            ms_entry = _MODEL_STORE.get(arch_tag, ckpt_outin)
            if ms_entry is None:
                if lowrank:
                    model = _MonicaNetOutInLowRank(in_dim=in_dim, det_count=len(det_to_idx),
                                                   n_out=n_out, n_in=n_in,
                                                   hidden_dims=hidden_dims, rank=rank)
                else:
                    model = _MonicaNetOutIn(in_dim=in_dim, det_count=len(det_to_idx),
                                            hidden_dims=hidden_dims, n_out=n_out, n_in=n_in)
                model.load_state_dict(ckpt["model"])
                # Optional inference transforms once
                if os.getenv("MONICA_COMPILE", "").strip():
                    try:
                        model = torch.compile(model, mode=os.getenv("MONICA_COMPILE").strip())
                    except Exception:
                        pass
                if os.getenv("MONICA_DQ", "0") == "1":
                    try:
                        from torch.ao.quantization import quantize_dynamic
                        model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                    except Exception:
                        pass
                if os.getenv("MONICA_JIT", "0") == "1":
                    try:
                        ex_x = torch.randn(1, in_dim, dtype=torch.float32)
                        ex_det = torch.zeros(1, dtype=torch.long)
                        model = torch.jit.trace(model, (ex_x, ex_det))
                    except Exception:
                        pass
                model = model.to(self.device).eval()
                _MODEL_STORE.put(arch_tag, ckpt_outin, model, det_to_idx, cfg)
                ms_entry = (model, det_to_idx, cfg)
            model, det_to_idx, cfg = ms_entry

            self._model = model
            self._target_mode = "outin"
            self._det_to_idx = det_to_idx
            self._det_id = int(det_to_idx[self.det_long])
            self._det_id_t = torch.tensor([self._det_id], dtype=torch.long)

            self._model_label = self._model_label_env or (
                f"outin_lr{rank}_{hidden_dims[0]}-{hidden_dims[1]}-{hidden_dims[2]}" if lowrank
                else f"outin_{hidden_dims[0]}-{hidden_dims[1]}-{hidden_dims[2]}"
            )

        else:
            # Native path (share single native model across dets)
            self._out_edges = read_cspec_out_edges(self.cspecfile).astype(np.float64)
            if self.det_long.startswith("NAI_"):
                in_edges = self.nai_in_edges_path_arg or (cfg_get(cfgm, "nai_in_edges") if cfgm else "")
            else:
                in_edges = self.bgo_in_edges_path_arg or (cfg_get(cfgm, "bgo_in_edges") if cfgm else "")
            if not in_edges:
                raise RuntimeError("Input edges path for family not provided")
            self._in_edges = np.load(in_edges).astype(np.float64)

            e_in, _, epx_lo, epx_hi = load_energy_axes(self.db_path, self.det_group)
            self._e_in = e_in.astype(np.float64)
            self._epx_lo = epx_lo.astype(np.float64)
            self._epx_hi = epx_hi.astype(np.float64)

            ckpt_path = self.model_path_arg or (cfg_get(cfgm, "model_path") if cfgm else "")
            if (not ckpt_path) or (not os.path.isfile(ckpt_path)):
                if cfgm:
                    name = os.environ.get("MONICA_MODEL") or cfg_get(cfgm, "selected_model")
                    if name:
                        ms = cfgm.get("model_set", {})
                        ckpt_path = str(ms.get(name, "")).strip()
            if (not ckpt_path) or (not os.path.isfile(ckpt_path)):
                raise RuntimeError(f"Monica native model checkpoint not found: {ckpt_path}")

            ms_entry = _MODEL_STORE.get("native", ckpt_path)
            if ms_entry is None:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                cfg = ckpt.get("config", {})
                in_dim = int(cfg.get("in_dim", 12))
                hidden_dims = tuple(int(x) for x in cfg.get("hidden_dims", [256,256,256]))
                det_to_idx = ckpt.get("det_to_idx", None)
                if det_to_idx is None:
                    names = [f"NAI_0{i}" for i in range(10)] + ["NAI_10", "NAI_11", "BGO_00", "BGO_01"]
                    det_to_idx = {n: i for i, n in enumerate(names)}
                model = _MonicaNetNative(in_dim=in_dim, det_count=len(det_to_idx), hidden_dims=hidden_dims)
                model.load_state_dict(ckpt["model"])
                # Optional transforms once
                if os.getenv("MONICA_COMPILE", "").strip():
                    try:
                        model = torch.compile(model, mode=os.getenv("MONICA_COMPILE").strip())
                    except Exception:
                        pass
                if os.getenv("MONICA_DQ", "0") == "1":
                    try:
                        from torch.ao.quantization import quantize_dynamic
                        model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                    except Exception:
                        pass
                if os.getenv("MONICA_JIT", "0") == "1":
                    try:
                        ex_x = torch.randn(1, in_dim, dtype=torch.float32)
                        ex_det = torch.zeros(1, dtype=torch.long)
                        model = torch.jit.trace(model, (ex_x, ex_det))
                    except Exception:
                        pass
                model = model.to(self.device).eval()
                _MODEL_STORE.put("native", ckpt_path, model, det_to_idx, cfg)
                ms_entry = (model, det_to_idx, cfg)
            model, det_to_idx, cfg = ms_entry

            self._model = model
            self._target_mode = "native"
            self._det_to_idx = det_to_idx
            self._det_id = int(det_to_idx[self.det_long])
            self._det_id_t = torch.tensor([self._det_id], dtype=torch.long)
            hidden_dims = tuple(int(x) for x in cfg.get("hidden_dims", [256,256,256]))
            self._model_label = self._model_label_env or f"native_{hidden_dims[0]}-{hidden_dims[1]}-{hidden_dims[2]}"

            try:
                self._pre = build_remap_precompute(
                    e_in=self._e_in, epx_lo=self._epx_lo, epx_hi=self._epx_hi,
                    target_in_edges=self._in_edges, target_out_edges=self._out_edges
                )
            except Exception:
                self._pre = None

        if self._profile:
            atexit.register(self._write_profile_csv)

        _BATCHER.register(self)
        self._initialized = True

    def _update_sc_pose(self, t: float):
        q = self._pos.quaternion(t)
        sc = self._pos.sc_pos(t)
        self._quat = np.asarray(q, dtype=np.float64)
        self._scpos = np.asarray(sc, dtype=np.float64)

    def _earth_geo_az_el(self) -> tuple[float, float]:
        q0, q1, q2, q3 = self._quat
        scx = np.array([
            q0*q0 - q1*q1 - q2*q2 + q3*q3,
            2.0*(q0*q1 + q3*q2),
            2.0*(q0*q2 - q3*q1),
        ], dtype=np.float64)
        scy = np.array([
            2.0*(q0*q1 - q3*q2),
            -q0*q0 + q1*q1 - q2*q2 + q3*q3,
            2.0*(q1*q2 + q3*q0),
        ], dtype=np.float64)
        scz = np.array([
            2.0*(q0*q2 + q3*q1),
            2.0*(q1*q2 - q3*q0),
            -q0*q0 - q1*q1 + q2*q2 + q3*q3,
        ], dtype=np.float64)
        geodir = np.array([-scx.dot(self._scpos), -scy.dot(self._scpos), -scz.dot(self._scpos)], dtype=np.float64)
        geodir /= (np.linalg.norm(geodir) + 1e-12)
        geo_az = np.arctan2(geodir[1], geodir[0])
        if geo_az < 0.0:
            geo_az += 2.0 * np.pi
        r_xy = np.hypot(geodir[0], geodir[1])
        geo_el = np.arctan2(r_xy, geodir[2])
        return float(np.rad2deg(geo_az)), float(np.rad2deg(geo_el))

    def _build_features(self, src_az_deg: float, src_el_deg: float) -> np.ndarray:
        geo_az_deg, geo_el_deg = self._earth_geo_az_el()
        theta_geo = 90.0 - geo_el_deg
        phi_geo = geo_az_deg
        theta_src = 90.0 - src_el_deg
        phi_src = src_az_deg

        thg = float(np.deg2rad(theta_geo))
        phg = float(np.deg2rad(phi_geo))
        ths = float(np.deg2rad(theta_src))
        phs = float(np.deg2rad(phi_src))
        gx, gy, gz, sl = geocoords(thg, phg, ths, phs)

        gz = np.asarray(gz, dtype=float).ravel()
        sl = np.asarray(sl, dtype=float).ravel()
        cos_lat = np.clip(gz.dot(sl), -1.0, 1.0)
        lat_deg = 180.0 - np.rad2deg(np.arccos(cos_lat))

        theta_c = float(self._theta_cent[_nearest_index(theta_geo, self._theta_cent)])
        lat_c   = float(self._lat_cent[_nearest_index(lat_deg, self._lat_cent)])
        phi_wrapped = (phi_geo + 360.0) % 360.0
        phi_c   = float(self._phi_cent[_nearest_index(phi_wrapped, self._phi_cent)])

        s = _azel_to_unit(src_az_deg, src_el_deg)
        cof = float(np.clip(self._det_n.dot(s), -1.0, 1.0))
        ctn = float(np.cos(np.deg2rad(theta_c)))

        def _trig(deg):
            r = np.deg2rad(deg)
            return np.sin(r), np.cos(r)

        saz, caz = _trig(src_az_deg)
        sel, cel = _trig(src_el_deg)
        sth, cth = _trig(theta_c)
        slt, clt = _trig(lat_c)
        sph, cph = _trig(phi_c)
        feat = np.array([saz, caz, sel, cel, sth, cth, slt, clt, sph, cph, cof, ctn], dtype=np.float32)
        return feat

    def set_time(self, t: float):
        self._lazy_init()
        self._time = float(t)
        self._update_sc_pose(self._time)
        self._pose_id += 1

    def set_location(self, ra_deg: float, dec_deg: float):
        self._lazy_init()
        if self._occult and is_occulted(float(ra_deg), float(dec_deg), self._scpos):
            Nout = len(self._out_edges) - 1
            Nin = len(self._in_edges) - 1
            self._matrix = np.zeros((Nout, Nin), dtype=np.float64, order="C")
            return
        loc_icrs = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
        frame = GBMFrame(quaternion_1=self._quat[0],
                         quaternion_2=self._quat[1],
                         quaternion_3=self._quat[2],
                         quaternion_4=self._quat[3],
                         sc_pos_X=self._scpos[0],
                         sc_pos_Y=self._scpos[1],
                         sc_pos_Z=self._scpos[2])
        loc_sat = loc_icrs.transform_to(frame)
        az_deg = float((loc_sat.lon.deg + 360.0) % 360.0)
        el_deg = float(loc_sat.lat.deg)
        self.set_location_direct_sat_coord(az_deg, el_deg)

    def _single_forward(self, az_deg: float, el_deg: float):
        t0 = time.perf_counter()
        x = self._build_features(float(az_deg), float(el_deg)).reshape(1, -1).astype(np.float32)
        self._t_feat += time.perf_counter() - t0
        x_t = torch.from_numpy(x)
        det_id = self._det_id_t
        t1 = time.perf_counter()
        y = self._model(x_t, det_id).squeeze(0).cpu().numpy()
        self._t_model += time.perf_counter() - t1
        if self._target_mode == "outin":
            self._matrix = np.ascontiguousarray(y.astype(np.float64), dtype=np.float64)
        else:
            t2 = time.perf_counter()
            M70 = y.astype(np.float64, copy=False)
            if self._pre is not None:
                M_inout = remap_apply_precomputed(M70, self._pre)
                R = M_inout.T
            else:
                R = remap_70to_outin(self._in_edges, self._out_edges, self._e_in, self._epx_lo, self._epx_hi, M70)
            self._t_remap += time.perf_counter() - t2
            self._matrix = np.ascontiguousarray(R, dtype=np.float64)
        self._n_calls += 1

    def set_location_direct_sat_coord(self, az_deg: float, el_deg: float):
        self._lazy_init()
        _BATCHER.run(self, float(az_deg), float(el_deg))

    def _write_profile_csv(self):
        if not self._profile or self._n_calls == 0:
            return
        total = self._t_feat + self._t_model + self._t_remap
        if total <= 0:
            return
        if self._profile_csv:
            out_path = self._profile_csv
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        else:
            d = self._profile_csv_dir or os.getcwd()
            os.makedirs(d, exist_ok=True)
            tag = self._bn or "bn_unknown"
            fname = f"profile_{tag}_{self.det_long}_rank{self._rank}.csv"
            out_path = os.path.join(d, fname)

        mean_feat_ms = self._t_feat / self._n_calls * 1000.0
        mean_model_ms = self._t_model / self._n_calls * 1000.0
        mean_remap_ms = self._t_remap / self._n_calls * 1000.0
        share_feat = self._t_feat / total
        share_model = self._t_model / total
        share_remap = self._t_remap / total

        header = ["bn","det","rank","run_tag","model_label",
                  "calls","mean_feat_ms","mean_model_ms","mean_remap_ms",
                  "share_feat","share_model","share_remap"]
        row = [self._bn, self.det_long, self._rank, self._run_tag, (self._model_label_env or getattr(self, "_model_label", "")),
               int(self._n_calls), mean_feat_ms, mean_model_ms, mean_remap_ms,
               share_feat, share_model, share_remap]
        try:
            newf = not os.path.exists(out_path)
            with open(out_path, "a", newline="") as f:
                w = csv.writer(f)
                if newf:
                    w.writerow(header)
                w.writerow(row)
        except Exception:
            print(f"[Monica profile] {row}")

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._lazy_init()
            Nout = int(len(self._out_edges) - 1) if self._out_edges is not None else 0
            Nin  = int(len(self._in_edges)  - 1) if self._in_edges  is not None else 0
            if Nout > 0 and Nin > 0:
                self._matrix = np.zeros((Nout, Nin), dtype=np.float64, order="C")
            else:
                self._matrix = np.zeros((1, 1), dtype=np.float64, order="C")
        return self._matrix

    @property
    def ebin_edge_in(self) -> np.ndarray:
        self._lazy_init()
        return self._in_edges

    @property
    def ebin_edge_out(self) -> np.ndarray:
        self._lazy_init()
        return self._out_edges

    @property
    def ebounds(self) -> np.ndarray:
        self._lazy_init()
        return self._out_edges

    @property
    def monte_carlo_energies(self) -> np.ndarray:
        self._lazy_init()
        return self._in_edges
    


from drm_monica.io.trigdat import get_trigdat_out_edges, get_trigdat_in_edges

class MonicaDRMGenTrig:
    def __init__(self, det_name, trigdat_file, model_path,
                 db_path, nai_in_edges, bgo_in_edges,
                 device="cpu", batch_size=4096, occult=True):
        self.det_short = det_name
        self.det_long = _short_to_long(det_name)
        self.det_group = _long_to_group(self.det_long)
        self.trigdat_file = trigdat_file
        self.model_path_arg = model_path
        self.db_path = db_path
        self.nai_in_edges_path_arg = nai_in_edges
        self.bgo_in_edges_path_arg = bgo_in_edges
        self.device = device
        self.batch_size = int(batch_size)
        self._occult = bool(occult)

        self._initialized = False
        self._matrix = None

        self._in_edges = None   # trigdat photon edges
        self._out_edges = None  # trigdat 8-channel edges

        self._pose_id = 0
        self._t_feat = 0.0
        self._t_model = 0.0
        self._t_remap = 0.0
        self._n_calls = 0
        self._profile = (os.getenv("MONICA_PROFILE", "0") == "1")
        self._profile_csv = os.getenv("MONICA_PROFILE_CSV", "").strip()
        self._profile_csv_dir = os.getenv("MONICA_PROFILE_CSV_DIR", "").strip()
        self._run_tag = os.getenv("MONICA_RUN_TAG", "").strip()
        self._model_label_env = os.getenv("MONICA_MODEL_LABEL", "").strip()
        self._rank = _get_mpi_rank()

        base = os.path.basename(self.trigdat_file)
        self._bn = ""
        for tok in base.replace(".", "_").split("_"):
            if tok.startswith("bn") and len(tok) >= 11:
                self._bn = tok
                break

        # DB axes (for native remap)
        self._e_in = None
        self._epx_lo = None
        self._epx_hi = None
        self._pre = None  # remap precompute

        # Model and detector mapping
        self._model = None
        self._target_mode = None
        self._det_to_idx = None
        self._det_id = None
        self._det_id_t = None

        # Geometry
        self._det_n = None
        self._pos = None
        self._quat = None
        self._scpos = None
        self._theta_cent = None
        self._lat_cent = None
        self._phi_cent = None

    def _lazy_init(self):
        if self._initialized:
            return

        # Torch threads
        try:
            th = int(os.getenv("MONICA_THREADS", "0")) or None
        except Exception:
            th = None
        if th is not None and th > 0:
            try:
                torch.set_num_threads(th)
            except Exception:
                pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        # Detector normal and geometry
        self._det_n = _det_normal(self.det_long)
        self._pos = gbmgeometry.PositionInterpolator.from_trigdat(trigdat_file=self.trigdat_file)
        self._update_sc_pose(0.0)

        # Load atm grid centers from DB (used to snap features)
        th_edge, lat_edge, phi_edge = load_atm_grid_info(self.db_path, self.det_group)
        self._theta_cent = 0.5 * (th_edge[:-1] + th_edge[1:])
        self._lat_cent   = 0.5 * (lat_edge[:-1] + lat_edge[1:])
        self._phi_cent   = 0.5 * (phi_edge[:-1] + phi_edge[1:])

        # Trigdat edges
        self._out_edges = get_trigdat_out_edges(self.det_long).astype(np.float64)
        # Input photon edges from npy (family-specific)
        if self.det_long.startswith("NAI_"):
            in_edges = self.nai_in_edges_path_arg
        else:
            in_edges = self.bgo_in_edges_path_arg
        if not in_edges:
            raise RuntimeError("Trigdat requires family input edges: set nai_in_edges/bgo_in_edges")
        self._in_edges = np.load(in_edges).astype(np.float64)

        # Load DB energy axes (native path)
        e_in, _, epx_lo, epx_hi = load_energy_axes(self.db_path, self.det_group)
        self._e_in   = e_in.astype(np.float64)
        self._epx_lo = epx_lo.astype(np.float64)
        self._epx_hi = epx_hi.astype(np.float64)

        # Load model (native 70×64 or out-in 8×N_in)
        cfgm = None
        try:
            from morgoth.configuration import morgoth_config
            cfgm = morgoth_config["drm_backend"]["monica"]
        except Exception:
            cfgm = None

        ckpt_path = self.model_path_arg or (cfg_get(cfgm, "model_path") if cfgm else "")
        # Support family-specific out-in checkpoints if provided in config
        fam_outin = None
        if cfgm:
            if self.det_long.startswith("NAI_"):
                fam_outin = cfg_get(cfgm, "nai_model")
            elif self.det_long == "BGO_00":
                fam_outin = cfg_get(cfgm, "bgo00_model")
            elif self.det_long == "BGO_01":
                fam_outin = cfg_get(cfgm, "bgo01_model")

        use_outin = bool(fam_outin and os.path.isfile(fam_outin))
        if use_outin:
            ckpt = torch.load(fam_outin, map_location="cpu")
            cfg = ckpt.get("config", {})
            in_dim = int(cfg.get("in_dim", 12))
            n_in  = int(cfg.get("n_in", len(self._in_edges) - 1))
            n_out = int(cfg.get("n_out", len(self._out_edges) - 1))
            hidden_dims = tuple(int(x) for x in cfg.get("hidden_dims", [256,256,256]))
            lowrank = bool(cfg.get("lowrank", False))
            rank = int(cfg.get("rank", 0) or 96)

            det_to_idx = ckpt.get("det_to_idx", None)
            if det_to_idx is None:
                names = [f"NAI_0{i}" for i in range(10)] + ["NAI_10", "NAI_11", "BGO_00", "BGO_01"]
                det_to_idx = {n: i for i, n in enumerate(names)}

            arch_tag = "outin_lowrank" if lowrank else "outin_dense"
            ms_entry = _MODEL_STORE.get(arch_tag, fam_outin)
            if ms_entry is None:
                if lowrank:
                    model = _MonicaNetOutInLowRank(in_dim=in_dim, det_count=len(det_to_idx),
                                                   n_out=n_out, n_in=n_in,
                                                   hidden_dims=hidden_dims, rank=rank)
                else:
                    model = _MonicaNetOutIn(in_dim=in_dim, det_count=len(det_to_idx),
                                            hidden_dims=hidden_dims, n_out=n_out, n_in=n_in)
                model.load_state_dict(ckpt["model"])
                # Optional inference transforms
                if os.getenv("MONICA_COMPILE", "").strip():
                    try: model = torch.compile(model, mode=os.getenv("MONICA_COMPILE").strip())
                    except Exception: pass
                if os.getenv("MONICA_DQ", "0") == "1":
                    try:
                        from torch.ao.quantization import quantize_dynamic
                        model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                    except Exception: pass
                if os.getenv("MONICA_JIT", "0") == "1":
                    try:
                        ex_x = torch.randn(1, in_dim, dtype=torch.float32)
                        ex_det = torch.zeros(1, dtype=torch.long)
                        model = torch.jit.trace(model, (ex_x, ex_det))
                    except Exception: pass
                model = model.to(self.device).eval()
                _MODEL_STORE.put(arch_tag, fam_outin, model, det_to_idx, cfg)
                ms_entry = (model, det_to_idx, cfg)
            model, det_to_idx, cfg = ms_entry
            self._model = model
            self._target_mode = "outin"
            self._det_to_idx = det_to_idx
            self._det_id = int(det_to_idx[self.det_long])
            self._det_id_t = torch.tensor([self._det_id], dtype=torch.long)
            self._model_label = self._model_label_env or (
                f"outin_lr{rank}_{hidden_dims[0]}-{hidden_dims[1]}-{hidden_dims[2]}" if lowrank
                else f"outin_{hidden_dims[0]}-{hidden_dims[1]}-{hidden_dims[2]}"
            )
        else:
            # Native path
            if (not ckpt_path) or (not os.path.isfile(ckpt_path)):
                raise RuntimeError(f"Monica native model checkpoint not found: {ckpt_path}")
            ms_entry = _MODEL_STORE.get("native", ckpt_path)
            if ms_entry is None:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                cfg = ckpt.get("config", {})
                in_dim = int(cfg.get("in_dim", 12))
                hidden_dims = tuple(int(x) for x in cfg.get("hidden_dims", [256,256,256]))
                det_to_idx = ckpt.get("det_to_idx", None)
                if det_to_idx is None:
                    names = [f"NAI_0{i}" for i in range(10)] + ["NAI_10", "NAI_11", "BGO_00", "BGO_01"]
                    det_to_idx = {n: i for i, n in enumerate(names)}
                model = _MonicaNetNative(in_dim=in_dim, det_count=len(det_to_idx), hidden_dims=hidden_dims)
                model.load_state_dict(ckpt["model"])
                if os.getenv("MONICA_COMPILE", "").strip():
                    try: model = torch.compile(model, mode=os.getenv("MONICA_COMPILE").strip())
                    except Exception: pass
                if os.getenv("MONICA_DQ", "0") == "1":
                    try:
                        from torch.ao.quantization import quantize_dynamic
                        model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                    except Exception: pass
                if os.getenv("MONICA_JIT", "0") == "1":
                    try:
                        ex_x = torch.randn(1, in_dim, dtype=torch.float32)
                        ex_det = torch.zeros(1, dtype=torch.long)
                        model = torch.jit.trace(model, (ex_x, ex_det))
                    except Exception: pass
                model = model.to(self.device).eval()
                _MODEL_STORE.put("native", ckpt_path, model, det_to_idx, cfg)
                ms_entry = (model, det_to_idx, cfg)
            model, det_to_idx, cfg = ms_entry
            self._model = model
            self._target_mode = "native"
            self._det_to_idx = det_to_idx
            self._det_id = int(det_to_idx[self.det_long])
            self._det_id_t = torch.tensor([self._det_id], dtype=torch.long)
            hidden_dims = tuple(int(x) for x in cfg.get("hidden_dims", [256,256,256]))
            self._model_label = self._model_label_env or f"native_{hidden_dims[0]}-{hidden_dims[1]}-{hidden_dims[2]}"
            # Precompute remap for trigdat out edges
            try:
                self._pre = build_remap_precompute(
                    e_in=self._e_in, epx_lo=self._epx_lo, epx_hi=self._epx_hi,
                    target_in_edges=self._in_edges, target_out_edges=self._out_edges
                )
            except Exception:
                self._pre = None

        if self._profile:
            atexit.register(self._write_profile_csv)

        _BATCHER.register(self)
        self._initialized = True

    def _update_sc_pose(self, t: float):
        q = self._pos.quaternion(t)
        sc = self._pos.sc_pos(t)
        self._quat = np.asarray(q, dtype=np.float64)
        self._scpos = np.asarray(sc, dtype=np.float64)

    def _earth_geo_az_el(self) -> tuple[float, float]:
        q0, q1, q2, q3 = self._quat
        scx = np.array([
            q0*q0 - q1*q1 - q2*q2 + q3*q3,
            2.0*(q0*q1 + q3*q2),
            2.0*(q0*q2 - q3*q1),
        ], dtype=np.float64)
        scy = np.array([
            2.0*(q0*q1 - q3*q2),
            -q0*q0 + q1*q1 - q2*q2 + q3*q3,
            2.0*(q1*q2 + q3*q0),
        ], dtype=np.float64)
        scz = np.array([
            2.0*(q0*q2 + q3*q1),
            2.0*(q1*q2 - q3*q0),
            -q0*q0 - q1*q1 + q2*q2 + q3*q3,
        ], dtype=np.float64)
        geodir = np.array([-scx.dot(self._scpos), -scy.dot(self._scpos), -scz.dot(self._scpos)], dtype=np.float64)
        geodir /= (np.linalg.norm(geodir) + 1e-12)
        geo_az = np.arctan2(geodir[1], geodir[0])
        if geo_az < 0.0:
            geo_az += 2.0 * np.pi
        r_xy = np.hypot(geodir[0], geodir[1])
        geo_el = np.arctan2(r_xy, geodir[2])
        return float(np.rad2deg(geo_az)), float(np.rad2deg(geo_el))

    def _build_features(self, src_az_deg: float, src_el_deg: float) -> np.ndarray:
        geo_az_deg, geo_el_deg = self._earth_geo_az_el()
        theta_geo = 90.0 - geo_el_deg
        phi_geo = geo_az_deg
        theta_src = 90.0 - src_el_deg
        phi_src = src_az_deg

        thg = float(np.deg2rad(theta_geo))
        phg = float(np.deg2rad(phi_geo))
        ths = float(np.deg2rad(theta_src))
        phs = float(np.deg2rad(phi_src))
        gx, gy, gz, sl = geocoords(thg, phg, ths, phs)
        gz = np.asarray(gz, dtype=float).ravel()
        sl = np.asarray(sl, dtype=float).ravel()
        cos_lat = np.clip(gz.dot(sl), -1.0, 1.0)
        lat_deg = 180.0 - np.rad2deg(np.arccos(cos_lat))

        theta_c = float(self._theta_cent[_nearest_index(theta_geo, self._theta_cent)])
        lat_c   = float(self._lat_cent[_nearest_index(lat_deg, self._lat_cent)])
        phi_wrapped = (phi_geo + 360.0) % 360.0
        phi_c   = float(self._phi_cent[_nearest_index(phi_wrapped, self._phi_cent)])

        s = _azel_to_unit(src_az_deg, src_el_deg)
        cof = float(np.clip(self._det_n.dot(s), -1.0, 1.0))
        ctn = float(np.cos(np.deg2rad(theta_c)))

        def _trig(deg):
            r = np.deg2rad(deg)
            return np.sin(r), np.cos(r)
        saz, caz = _trig(src_az_deg)
        sel, cel = _trig(src_el_deg)
        sth, cth = _trig(theta_c)
        slt, clt = _trig(lat_c)
        sph, cph = _trig(phi_c)
        feat = np.array([saz, caz, sel, cel, sth, cth, slt, clt, sph, cph, cof, ctn], dtype=np.float32)
        return feat

    def set_time(self, t: float):
        self._lazy_init()
        self._time = float(t)
        self._update_sc_pose(self._time)
        self._pose_id += 1

    def set_location(self, ra_deg: float, dec_deg: float):
        self._lazy_init()
        if self._occult and is_occulted(float(ra_deg), float(dec_deg), self._scpos):
            Nout = len(self._out_edges) - 1
            Nin = len(self._in_edges) - 1
            self._matrix = np.zeros((Nout, Nin), dtype=np.float64, order="C")
            return
        loc_icrs = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
        frame = GBMFrame(quaternion_1=self._quat[0],
                         quaternion_2=self._quat[1],
                         quaternion_3=self._quat[2],
                         quaternion_4=self._quat[3],
                         sc_pos_X=self._scpos[0],
                         sc_pos_Y=self._scpos[1],
                         sc_pos_Z=self._scpos[2])
        loc_sat = loc_icrs.transform_to(frame)
        az_deg = float((loc_sat.lon.deg + 360.0) % 360.0)
        el_deg = float(loc_sat.lat.deg)
        self.set_location_direct_sat_coord(az_deg, el_deg)

    def set_location_direct_sat_coord(self, az_deg: float, el_deg: float):
        self._lazy_init()
        # Use shared batcher for multi-det calls at same geometry
        _BATCHER.run(self, float(az_deg), float(el_deg))

    def _single_forward(self, az_deg: float, el_deg: float):
        t0 = time.perf_counter()
        x = self._build_features(float(az_deg), float(el_deg)).reshape(1, -1).astype(np.float32)
        self._t_feat += time.perf_counter() - t0
        x_t = torch.from_numpy(x)
        det_id = self._det_id_t
        t1 = time.perf_counter()
        y = self._model(x_t, det_id).squeeze(0).cpu().numpy()
        self._t_model += time.perf_counter() - t1
        if self._target_mode == "outin":
            self._matrix = np.ascontiguousarray(y.astype(np.float64), dtype=np.float64)
        else:
            t2 = time.perf_counter()
            M70 = y.astype(np.float64, copy=False)
            if self._pre is not None:
                M_inout = remap_apply_precomputed(M70, self._pre)
                R = M_inout.T
            else:
                R = remap_70to_outin(self._in_edges, self._out_edges, self._e_in, self._epx_lo, self._epx_hi, M70)
            self._t_remap += time.perf_counter() - t2
            self._matrix = np.ascontiguousarray(R, dtype=np.float64)
        self._n_calls += 1

    def _write_profile_csv(self):
        if not self._profile or self._n_calls == 0:
            return
        total = self._t_feat + self._t_model + self._t_remap
        if total <= 0:
            return
        if self._profile_csv:
            out_path = self._profile_csv
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        else:
            d = self._profile_csv_dir or os.getcwd()
            os.makedirs(d, exist_ok=True)
            tag = self._bn or "bn_unknown"
            fname = f"profile_{tag}_{self.det_long}_trig_rank{self._rank}.csv"
            out_path = os.path.join(d, fname)

        mean_feat_ms = self._t_feat / self._n_calls * 1000.0
        mean_model_ms = self._t_model / self._n_calls * 1000.0
        mean_remap_ms = self._t_remap / self._n_calls * 1000.0
        share_feat = self._t_feat / total
        share_model = self._t_model / total
        share_remap = self._t_remap / total

        header = ["bn","det","rank","run_tag","model_label",
                  "calls","mean_feat_ms","mean_model_ms","mean_remap_ms",
                  "share_feat","share_model","share_remap"]
        row = [self._bn, self.det_long, self._rank, self._run_tag, (self._model_label_env or getattr(self, "_model_label", "")),
               int(self._n_calls), mean_feat_ms, mean_model_ms, mean_remap_ms,
               share_feat, share_model, share_remap]
        try:
            newf = not os.path.exists(out_path)
            with open(out_path, "a", newline="") as f:
                w = csv.writer(f)
                if newf:
                    w.writerow(header)
                w.writerow(row)
        except Exception:
            print(f"[Monica profile] {row}")

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._lazy_init()
            Nout = int(len(self._out_edges) - 1) if self._out_edges is not None else 0
            Nin  = int(len(self._in_edges)  - 1) if self._in_edges  is not None else 0
            if Nout > 0 and Nin > 0:
                self._matrix = np.zeros((Nout, Nin), dtype=np.float64, order="C")
            else:
                self._matrix = np.zeros((1, 1), dtype=np.float64, order="C")
        return self._matrix

    @property
    def ebin_edge_in(self) -> np.ndarray:
        self._lazy_init()
        return self._in_edges

    @property
    def ebin_edge_out(self) -> np.ndarray:
        self._lazy_init()
        return self._out_edges

    @property
    def ebounds(self) -> np.ndarray:
        self._lazy_init()
        return self._out_edges

    @property
    def monte_carlo_energies(self) -> np.ndarray:
        self._lazy_init()
        return self._in_edges
