import numpy as np
import torch
from astropy.io import fits

from drm_monica.remap import remap_70to_outin
from drm_monica.io.cspec import read_cspec_out_edges
from drm_monica.db_reader import load_energy_axes, load_atm_grid_info

from gbm_drm_gen.matrix_functions import geocoords

# Detector naming and normals (spacecraft frame) to compute cos(off-axis)
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
        if s == "na": return "NAI_10"
        if s == "nb": return "NAI_11"
        idx = int(s[1])
        return f"NAI_0{idx}"
    if s == "b0": return "BGO_00"
    if s == "b1": return "BGO_01"
    raise ValueError(f"Unknown detector short name: {det_short}")

def _long_to_group(long_name: str) -> str:
    if long_name.startswith("NAI_"):
        idx = int(long_name.split("_")[1])
        return f"n{idx}" if idx < 10 else ("na" if idx == 10 else "nb")
    if long_name == "BGO_00": return "b0"
    if long_name == "BGO_01": return "b1"
    raise ValueError(f"Cannot map to DB group: {long_name}")

def _azel_to_unit(az_deg: float, el_deg: float) -> np.ndarray:
    a = np.deg2rad(az_deg); e = np.deg2rad(el_deg)
    return np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)], dtype=np.float64)

def _det_normal(long_name: str) -> np.ndarray:
    az, zen = DET_ORIENT_DEG[long_name]
    # Convert zenith to elevation
    el = 90.0 - zen
    return _azel_to_unit(az, el)

def _nearest_index(val: float, centers: np.ndarray) -> int:
    return int(np.argmin(np.abs(centers - val)))

class _MonicaNet(torch.nn.Module):
    # Minimal Monica MLP for inference (matches training in drm-monica/scripts/train_baseline.py)
    def __init__(self, in_dim: int, det_count: int, det_emb_dim: int = 8, hidden: int = 256):
        super().__init__()
        self.emb = torch.nn.Embedding(det_count, det_emb_dim) if det_count > 1 else None
        feat_in = in_dim + (det_emb_dim if self.emb is not None else 0)
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(feat_in, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        )
        self.eff = torch.nn.Linear(hidden, 70)
        self.shp = torch.nn.Linear(hidden, 70*64)
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim=-1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, det_id: torch.Tensor | None):
        # x: (B, in_dim), det_id: (B,) long tensor or None
        if self.emb is not None and det_id is not None:
            emb = self.emb(det_id)
            x = torch.cat([x, emb], dim=-1)
        h = self.backbone(x)
        eff = self.softplus(self.eff(h))              # (B,70)
        shp = self.softmax(self.shp(h).view(-1, 70, 64))  # (B,70,64)
        y = eff.unsqueeze(-1) * shp                   # (B,70,64)
        return y

class MonicaDRMGen:
    """
    TTE-only Monica response adapter for BALROGLike/BALROG_DRM.

    Provides:
      - set_location_direct_sat_coord(az_deg, el_deg)
      - matrix property: [N_out, N_in] (out-in)
      - ebin edges accessors (if queried by callers)
    """
    def __init__(self,
                 det_name: str,
                 trigdat_file: str,
                 cspecfile: str,
                 model_path: str,
                 db_path: str,
                 nai_in_edges: str,
                 bgo_in_edges: str,
                 device: str = "cpu",
                 batch_size: int = 4096):
        self.det_short = det_name  # e.g., "n7", "na", "b0"
        self.det_long = _short_to_long(det_name)  # "NAI_07" ...
        self.det_group = _long_to_group(self.det_long)  # "n7" ...
        self.trigdat_file = trigdat_file
        self.cspecfile = cspecfile
        self.model_path = model_path
        self.db_path = db_path
        self.nai_in_edges_path = nai_in_edges
        self.bgo_in_edges_path = bgo_in_edges
        self.device = device
        self.batch_size = int(batch_size)

        # Lazy state
        self._initialized = False
        self._matrix = None

        # Expose edges (filled on init)
        self._in_edges = None
        self._out_edges = None

    def _lazy_init(self):
        if self._initialized:
            return

        # Load per-detector fixed edges
        self._out_edges = read_cspec_out_edges(self.cspecfile).astype(np.float64)
        if self.det_long.startswith("NAI_"):
            self._in_edges = np.load(self.nai_in_edges_path).astype(np.float64)
        else:
            self._in_edges = np.load(self.bgo_in_edges_path).astype(np.float64)

        # Load per-detector DB axes
        e_in, _, epx_lo, epx_hi = load_energy_axes(self.db_path, self.det_group)
        self._e_in = e_in.astype(np.float64)
        self._epx_lo = epx_lo.astype(np.float64)
        self._epx_hi = epx_hi.astype(np.float64)

        # Atmospheric grid edges -> centers (for snapping)
        th_edge, lat_edge, phi_edge = load_atm_grid_info(self.db_path, self.det_group)
        self._theta_cent = 0.5 * (th_edge[:-1] + th_edge[1:])
        self._lat_cent   = 0.5 * (lat_edge[:-1] + lat_edge[1:])
        self._phi_cent   = 0.5 * (phi_edge[:-1] + phi_edge[1:])

        # Detector normal for cos(off-axis)
        self._det_n = _det_normal(self.det_long)

        # Load trigdat geometry once (spacecraft quaternion and position)
        with fits.open(self.trigdat_file, memmap=False) as f:
            # Use the first TRIGRATE row (same convention as in Morgoth)
            self._quat = np.array(f["TRIGRATE"].data["SCATTITD"][0], dtype=np.float64)  # [q0,q1,q2,q3]
            self._scpos = np.array(f["TRIGRATE"].data["EIC"][0], dtype=np.float64)     # spacecraft position (ECEF-like)

        # Load Monica model checkpoint
        ckpt = torch.load(self.model_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        in_dim = int(cfg.get("in_dim", 12))
        det_to_idx = ckpt.get("det_to_idx", None)
        if det_to_idx is None:
            # Fallback mapping if absent
            # Order must match training; we map NAI_00..NAI_11,BGO_00,BGO_01 if present
            names = [f"NAI_0{i}" for i in range(10)] + ["NAI_10","NAI_11","BGO_00","BGO_01"]
            det_to_idx = {n: i for i, n in enumerate(names)}
        self._det_to_idx = det_to_idx
        self._det_id = int(det_to_idx[self.det_long])

        self._model = _MonicaNet(in_dim=in_dim, det_count=len(det_to_idx))
        self._model.load_state_dict(ckpt["model"])
        self._model.to(self.device).eval()

        self._initialized = True

    def _earth_geo_az_el(self) -> tuple[float, float]:
        """
        Compute Earth (nadir) az/el in spacecraft frame from trigdat quaternion and spacecraft position.
        Matches the convention used in drm_monica/scripts/build_atm_dist.geo_az_el_from_quat_pos.
        """
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
        # Direction to Earth center in spacecraft frame
        geodir = np.array([-scx.dot(self._scpos), -scy.dot(self._scpos), -scz.dot(self._scpos)], dtype=np.float64)
        geodir /= (np.linalg.norm(geodir) + 1e-12)
        geo_az = np.arctan2(geodir[1], geodir[0])  # radians
        if geo_az < 0.0:
            geo_az += 2.0*np.pi
        r_xy = np.hypot(geodir[0], geodir[1])
        geo_el = np.arctan2(r_xy, geodir[2])   # elevation
        return float(np.rad2deg(geo_az)), float(np.rad2deg(geo_el))

    def _build_features(self, src_az_deg: float, src_el_deg: float) -> np.ndarray:
        """
        12-dim feature vector used during training:
          [sin/cos src_az, sin/cos src_el,
           sin/cos theta_c, sin/cos lat_c, sin/cos phi_c,
           cos(off-axis), cos(theta_c)]
        with atmospheric (theta, lat, phi) snapped to nearest cell centers.
        """
        # Earth geometry in spacecraft frame
        geo_az_deg, geo_el_deg = self._earth_geo_az_el()

        # Compute latitude via classic geocoords (training convention)
        theta_geo = 90.0 - geo_el_deg
        phi_geo = geo_az_deg
        theta_src = 90.0 - src_el_deg
        phi_src = src_az_deg

        gx, gy, gz, sl = geocoords(np.deg2rad(theta_geo), np.deg2rad(phi_geo),
                                   np.deg2rad(theta_src), np.deg2rad(phi_src))
        gz = np.asarray(gz, dtype=float).ravel()
        sl = np.asarray(sl, dtype=float).ravel()
        cos_lat = np.clip(gz.dot(sl), -1.0, 1.0)
        lat_deg = 180.0 - np.rad2deg(np.arccos(cos_lat))

        # Snap to nearest atmospheric centers
        theta_c = float(self._theta_cent[_nearest_index(theta_geo, self._theta_cent)])
        lat_c   = float(self._lat_cent[_nearest_index(lat_deg, self._lat_cent)])
        # Wrap phi into [0, 360)
        phi_wrapped = (phi_geo + 360.0) % 360.0
        phi_c   = float(self._phi_cent[_nearest_index(phi_wrapped, self._phi_cent)])

        # cos(off-axis)
        s = _azel_to_unit(src_az_deg, src_el_deg)
        cof = float(np.clip(self._det_n.dot(s), -1.0, 1.0))

        # cos(theta_c) proxy
        ctn = float(np.cos(np.deg2rad(theta_c)))

        # trig features
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

    def set_location_direct_sat_coord(self, az_deg: float, el_deg: float):
        """
        Set current source direction (spacecraft-frame az/el in degrees), compute DRM via Monica (70x64 -> remap),
        and store as out-in [N_out, N_in] matrix in self._matrix.
        """
        self._lazy_init()

        # Build features for this det/direction
        x = self._build_features(float(az_deg), float(el_deg)).reshape(1, -1)
        x_t = torch.from_numpy(x).to(self.device)
        det_id = torch.tensor([self._det_id], dtype=torch.long, device=self.device)

        with torch.no_grad():
            M70 = self._model(x_t, det_id).squeeze(0).cpu().numpy().astype(np.float64)

        # Remap to requested edges (out-in orientation for Morgoth/3ML)
        R = remap_70to_outin(self._in_edges, self._out_edges,
                             self._e_in, self._epx_lo, self._epx_hi, M70)
        # Ensure contiguous C-order
        self._matrix = np.ascontiguousarray(R, dtype=np.float64)

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            raise RuntimeError("Call set_location_direct_sat_coord() before accessing matrix")
        return self._matrix

    # Optional helpers if callers query edges
    @property
    def ebin_edge_in(self) -> np.ndarray:
        self._lazy_init()
        return self._in_edges

    @property
    def ebin_edge_out(self) -> np.ndarray:
        self._lazy_init()
        return self._out_edges
        