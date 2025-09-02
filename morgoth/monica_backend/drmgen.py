import numpy as np

  class MonicaDRMGen:
      """
      TTE-only Monica response adapter for BALROG_DRM.
      Provides:
        - set_location_direct_sat_coord(az_deg, el_deg)
        - matrix property: [N_out, N_in] (out-in)
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
          self.det_name = det_name
          self.trigdat_file = trigdat_file
          self.cspecfile = cspecfile
          self.model_path = model_path
          self.db_path = db_path
          self.nai_in_edges_path = nai_in_edges
          self.bgo_in_edges_path = bgo_in_edges
          self.device = device
          self.batch_size = int(batch_size)

          # Lazy init of heavy objects (model, DB axes)
          self._initialized = False
          self._matrix = None

      def _lazy_init(self):
          if self._initialized:
              return
          # TODO: load Monica model, det mapping, DB axes (e_in, epx_lo, epx_hi),
          #       read CSPEC EBOUNDS into self._out_edges,
          #       set self._in_edges by family (NaI/BGO).
          self._in_edges = None
          self._out_edges = None
          self._initialized = True

      def set_location_direct_sat_coord(self, az_deg: float, el_deg: float):
          self._lazy_init()
          # TODO: compute Monica features from (az_deg, el_deg) and trigdat geometry,
          #       run model -> 70x64, remap to (self._in_edges, self._out_edges),
          #       store as self._matrix (out-in).
          raise NotImplementedError("MonicaDRMGen not wired yet")

      @property
      def matrix(self) -> np.ndarray:
          if self._matrix is None:
              raise RuntimeError("Call set_location_direct_sat_coord first")
          return self._matrix