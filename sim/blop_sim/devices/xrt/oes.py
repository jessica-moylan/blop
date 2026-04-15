"""Base Optic Device mirror devices for XRTBackend."""

from ophyd_async.core import StandardReadable, soft_signal_rw
from ophyd_async.core import StandardReadableFormat as Format

from ...backends import SimBackend


class DBHR(StandardReadable):
    """base optic Device mirror with curvature radius control (for XRTBackend).

    Args:
        backend: Simulation backend (should be XRTBackend)
        center: The third value of the center
        name: Device name
    """

    def __init__(
        self,
        backend: SimBackend,
        optic_index: int,
        center: float = 0,
        name: str = "",
    ):
        self._backend = backend
        self._optic_index = optic_index

        # Curvature radius signal
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.center = soft_signal_rw(float, center)

        super().__init__(name=name)

        # Register with backend
        backend.register_device(
            device_name=name,
            device_type="oes_xrt",
            get_state_callback=self._get_state,
        )

    async def _get_state(self) -> dict:
        """Get current mirror state for backend (async)."""
        return {
            "optic_index": self._optic_index,
            "center": await self.center.get_value(),
        }


__all__ = ["DBHR"]
