"""KB mirror devices for XRTBackend."""

from ophyd_async.core import StandardReadable, soft_signal_rw
from ophyd_async.core import StandardReadableFormat as Format

from ...backends import SimBackend


class ToroidalMirror(StandardReadable):
    """KB mirror with curvature radius control (for XRTBackend).

    Exposes a single radius parameter that directly controls the XRT mirror R value.
    Used with XRTBioXAS for ray-tracing simulation.

    Args:
        backend: Simulation backend (should be XRTBioXAS)
        mirror_index: 0 for first mirror (vertical), 1 for second mirror (horizontal)
        initial_radius_meridional: Initial curvature radius in mm
        name: Device name
    """

    def __init__(
        self,
        backend: SimBackend,
        mirror_index: int,
        radius_meridional: float = 30000.0,
        name: str = "",
    ):
        self._backend = backend
        self._mirror_index = mirror_index

        # Curvature radius signal
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.radius_meridional = soft_signal_rw(float, radius_meridional)

        super().__init__(name=name)

        # Register with backend
        backend.register_device(
            device_name=name,
            device_type="ToroidalMirror",
            get_state_callback=self._get_state,
        )

    async def _get_state(self) -> dict:
        """Get current mirror state for backend (async)."""
        return {
            "mirror_index": self._mirror_index,
            "radius_meridional": await self.radius_meridional.get_value(),
        }


__all__ = ["ToroidalMirror"]
