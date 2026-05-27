import numpy as np

from . import SimBackend
from .models.xrt_bioxas_model import build_beamline, build_histRGB, run_process

class XRTBIOXASBackend(SimBackend):
    """XRT ray-tracing simulation backend.

    Uses the XRT package to perform realistic ray-tracing to simulate the BIOXAS beamline.
    The beamline object (created by build_beamline) contains all simulation elements (mirrors, screens, etc.).
    """

    def __init__(self, noise: bool = False, beamline=None):
        """
        Initialize XRT backend.

        Parameters
        ----------
        noise : bool
            Whether to add noise to the simulated image.
        beamline : object or None
            If provided, use this beamline object (from build_beamline) for simulation.
            If None, a new beamline will be created automatically.
        """
        super().__init__()
        self._beamline = beamline  # Use user-supplied or auto-created beamline
        self._limits = [[-0.6, 0.6], [-0.45, 0.45]]
        self._noise = noise

    def _ensure_beamline(self):
        """Build XRT beamline if not already built."""
        if self._beamline is None:
            self._beamline = build_beamline()  # All simulation objects are attributes of this beamline

    async def generate_beam(self) -> np.ndarray:
        """Generate beam using XRT ray-tracing.

        Returns
        -------
        np.ndarray
            2D numpy array with shape (300, 400)
        """
        self._ensure_beamline()

        # Get mirror radii from devices
        mirror_radii_meridional = await self.mirror_radii_meridional()

        # Get information for DBHR devices (pitch and roll for each mirror)
        dbhr_info = await self._get_dbhr_information()

        self._beamline.DBHR1.extraPitch = dbhr_info[0] 
        self._beamline.DBHR1.extraRoll = dbhr_info[2]
        self._beamline.DBHR2.extraPitch = dbhr_info[1]
        self._beamline.DBHR2.extraRoll = dbhr_info[3]

        # Update XRT beamline mirror parameters
        self._beamline.Mirror1.R = mirror_radii_meridional[0]  # Vertical mirror Meridional radius
        self._beamline.Mirror2.R = mirror_radii_meridional[1]  # Horizontal mirror Meridional radius
        
        # Run ray tracing using the beamline and its objects
        outDict = run_process(self._beamline)
        lb = outDict["SampleScreen_local"]  # Get ray data at the sample screen


        # Build histogram from ray data
        hist2d, _, _ = build_histRGB(lb, lb, limits=self._limits, isScreen=True, shape=[400, 300])
        image = hist2d

        # Add noise if requested
        if self._noise:
            image += 1e-3 * np.abs(np.random.standard_normal(size=image.shape))

        return image

    async def mirror_radii_meridional(self) -> list[float]:
        """Get KB mirror radii from registered devices.

        Returns
        -------
        list[float]
            [R1, R2] where R1 is first mirror (vertical), R2 is second mirror (horizontal)
        """
        radii = [None, None] 

        for name, device in self._device_states.items():
            if device["type"] == "ToroidalMirror":
                state = await self._get_device_state(name)
                mirror_index = state["mirror_index"]
                radii[mirror_index] = state["radius_meridional"]

        return radii
            
    async def _get_dbhr_information(self) -> list[float,float]:
        pitch = [None, None]
        yaw = [None, None]  
        for name, device in self._device_states.items():
            if device["type"] == "dbhm_xrt":
                state = await self._get_device_state(name)
                mirror_index = state["optic_index"]
                pitch[mirror_index] = state["extraPitch"]
                yaw[mirror_index] = state["extraRoll"]
        return pitch + yaw


__all__ = ["XRTBIOXASBackend"]