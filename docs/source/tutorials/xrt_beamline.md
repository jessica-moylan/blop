---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: docs
  language: python
  name: python3
---

```{code-cell} ipython3
import logging
from pathlib import PurePath

import numpy as np
import matplotlib.pyplot as plt
from tiled.client.container import Container
from bluesky.callbacks import best_effort
from bluesky_tiled_plugins import TiledWriter
from bluesky.run_engine import RunEngine
from event_model import RunRouter
from tiled.client import from_uri
from tiled.server import SimpleTiledServer
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider

from blop.ax import Agent, RangeDOF, Objective
from blop.protocols import EvaluationFunction

from blop_sim.backends.models.xrt_kb_model import build_histRGB
import numpy as np
import cv2


import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.materials.elemental as rmatsel
import xrt.backends.raycing.materials.compounds as rmatsco
import xrt.backends.raycing.materials.crystals as rmatscr
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun



# Simulation back-end for the BioXAS-Main beamline
from blop_sim.backends.simple import SimBackend
from blop_sim.devices.xrt import DBHR, Slit, KBMirror
from blop_sim.devices import DetectorDevice

import pdb

logging.getLogger("httpx").setLevel(logging.WARNING)
plt.ion()

DETECTOR_STORAGE = "/tmp/blop/bioxas"
```

### Tiled Server and Run Engine

```{code-cell} ipython3
tiled_server = SimpleTiledServer(readable_storage=[DETECTOR_STORAGE])
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)

def bec_factory(name, doc):
    bec = best_effort.BestEffortCallback()
    bec.disable_plots()
    return [bec], []

rr = RunRouter([bec_factory])

RE = RunEngine({})
RE.subscribe(rr)
RE.subscribe(tiled_writer)
```

```{code-cell} ipython3
CVD = rmats.Material(
    elements=['C'],
    quantities=[1.0],
    kind=r"plate",
    rho=3.52,
    name=r"CVD")

Rh = rmats.Material(
    elements=['Rh'],
    quantities=[1.0],
    kind=r"mirror",
    rho=12.41,
    name=r"Rh")

Si220 = rmats.CrystalSi(
    a=5.4307717932001225,
    hkl=[2, 2, 0],
    d=1.9200677810242166,
    V=160.17128543981727,
    elements=['Si'],
    quantities=[1.0],
    name=r"Si220",
    kind=r"crystal")

CVDcoating = rmats.Material(
    elements=['C'],
    quantities=[1.0],
    kind=r"mirror",
    rho=3.52,
    name=r"CVDcoating")

Si = rmats.Material(
    elements=['Si'],
    quantities=[1.0],
    kind=r"mirror",
    rho=2.33,
    name=r"Si")

RhOnSi = rmats.Coated(
    coating=Rh,
    surfaceRoughness=2,
    substrate=Si,
    name=r"RhOnSi")

CVDonSi = rmats.Coated(
    coating=CVDcoating,
    surfaceRoughness=2,
    substrate=Si,
    name=r"CVDonSi")

Si220harm = rmats.CrystalHarmonics(
    Nmax=2,
    name=r"Si220harm",
    hkl=[2, 2, 0],
    a=5.41949,
    b=5.41949,
    c=5.41949,
    alpha=1.5707963267948966,
    beta=1.5707963267948966,
    gamma=1.5707963267948966,
    atomsFraction=[1, 1, 1, 1, 1, 1, 1, 1],
    d=1.916079064786341,
    V=159.1751463370933,
    elements=['Si', 'Si', 'Si', 'Si', 'Si', 'Si', 'Si', 'Si'],
    quantities=[1, 1, 1, 1, 1, 1, 1, 1],
    rho=2.3439368026411915,
    kind=r"crystal harmonics")
```

```{code-cell} ipython3
def build_beamline():
    BioXAS_Main = raycing.BeamLine(
        alignE=8000,
        name=r"BioXAS_Main")

    BioXAS_Main.Wiggler = raycing.sources.synchr.Wiggler(
        bl=BioXAS_Main,
        name=r"Wiggler",
        center=[0, 0, 0],
        nrays=1500000,
        eE=2.9,
        eI=0.25,
        eSigmaX=405.84479792157003,
        eSigmaZ=10.067770358922576,
        eEpsilonX=18.099999999999998,
        eEpsilonZ=0.0362,
        betaX=9.1,
        betaZ=2.8000000000000003,
        xPrimeMax=1.0,
        zPrimeMax=0.375,
        eMin=7998.0,
        eMax=8002.0,
        eN=52,
        K=35,  # Deflection parameter
        period=150,
        n=11,
        nx=51,
        nz=51)

    BioXAS_Main.FEMask = rapts.RectangularAperture(
        bl=BioXAS_Main,
        name=r"FEMask",
        center=[0, 12000, 0],
        opening=[-12.0, 12.0, -1.75, 1.75],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    BioXAS_Main.DiamondFilter = roes.Plate(
        bl=BioXAS_Main,
        name=r"DiamondFilter",
        center=[0, 13600, 0],
        pitch=1.5707963267948966,
        material=CVD,
        limPhysX=[-5.0, 5.0],
        limOptX=[-5, 5],
        limPhysY=[-5.0, 5.0],
        limOptY=[-5, 5],
        t=0.05)

    BioXAS_Main.WhiteBeamSlits = rapts.RectangularAperture(
        bl=BioXAS_Main,
        name=r"WhiteBeamSlits",
        center=[0, 14000, 0],
        opening=[-10.0, 10.0, -1.0, 1.0],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    BioXAS_Main.Mirror1 = roes.ToroidMirror(
        bl=BioXAS_Main,
        name=r"Mirror1",
        center=[0, 14600, 0],
        pitch=r"0.15 deg",
        material=RhOnSi,
        limPhysX=[-12.0, 12.0],
        limOptX=[-12, 12],
        limPhysY=[-495.0, 495.0],
        limOptY=[-495, 495],
        order=1,
        R=7120000.0,
        r=69.81)

    BioXAS_Main.CM_Slits = rapts.RectangularAperture(
        bl=BioXAS_Main,
        name=r"CM_Slits",
        center=[0, 15600, 5.378324617301544],
        opening=[-5.0, 5.0, -2.0, 2.0],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    BioXAS_Main.SSRL_DCM = raycing.oes.DCM(
        bl=BioXAS_Main,
        name=r"SSRL_DCM",
        center=[0, 25300, 56.172806077564246],
        bragg=[8000],
        material=Si220,
        material2=Si220,
        limPhysX=[-20.0, 20.0],
        limOptX=[-20, 20],
        limPhysY=[-72.3913, 3.8087],
        limOptY=[-72.3913, 3.8087],
        limPhysX2=[-20.0, 20.0],
        limPhysY2=[-1.1951, 94.0549],
        limOptX2=[-20, 20],
        limOptY2=[-1.1951, 94.0549],
        order=1,
        cryst2perpTransl=6.5023)

    BioXAS_Main.PreM2Screen = rscreens.Screen(
        bl=BioXAS_Main,
        name=r"PreM2Screen",
        center=[0, 26000, 71.73657379831323],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[0.0, 0.0],
        limPhysY=[0.0, 0.0],
        cLimits=[0.0, 0.0])

    BioXAS_Main.Mirror2 = roes.ToroidMirror(
        bl=BioXAS_Main,
        name=r"Mirror2",
        center=[0, 26900, 76.44962311674786],
        pitch=r"-0.15 degree",
        positionRoll=3.141592653589793,
        material=RhOnSi,
        limOptX=[-12, 12],
        limOptY=[-550, 550],
        order=1,
        R=2500000.0,
        r=35.9)

    BioXAS_Main.PhotonShutter = rapts.RectangularAperture(
        bl=BioXAS_Main,
        name=r"PhotonShutter",
        center=[0, 28300, 76.32517494206057],
        opening=[-5.0, 5.0, -2.0, 2.0],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    BioXAS_Main.DBHR1 = raycing.oes.OE(
        bl=BioXAS_Main,
        name=r"DBHR1",
        center=[0, 29900, 76.42861713167648], 
        pitch=r"0.2 deg",
        material=CVDcoating,
        limPhysX=[-10.0, 10.0],
        limOptX=[-10, 10],
        limPhysY=[-75.0, 75.0],
        limOptY=[-75, 75],
        order=1)

    BioXAS_Main.DBHR2 = raycing.oes.OE(
        bl=BioXAS_Main,
        name=r"DBHR2",
        center=[0, 30075, 77.61464367739664],
        pitch=r"-0.2 deg",
        positionRoll=3.141592653589793,
        material=CVDcoating,
        limPhysX=[-10.0, 10.0],
        limOptX=[-10, 10],
        limPhysY=[-75.0, 75.0],
        limOptY=[-75, 75],
        order=1)

    BioXAS_Main.JJslits = rapts.RectangularAperture(
        bl=BioXAS_Main,
        name=r"JJslits",
        center=[0, 30350, 77.63364600960998],
        opening=[-5.0, 5.0, -0.2, 0.2],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    BioXAS_Main.SampleScreen = rscreens.Screen(
        bl=BioXAS_Main,
        name=r"SampleScreen",
        center=[0, 30650, 77.63884281299194],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[0.0, 0.0],### Checking Optimization Health
        limPhysY=[0.0, 0.0],
        cLimits=[0.0, 0.0])

    return BioXAS_Main
```

```{code-cell} ipython3
def run_process(BioXAS_Main):
    Wiggler_global = BioXAS_Main.Wiggler.shine()

    FEMask_local = BioXAS_Main.FEMask.propagate(
        beam=Wiggler_global)

    DiamondFilter_global, DiamondFilter_local1, DiamondFilter_local2 = BioXAS_Main.DiamondFilter.double_refract(
        beam=Wiggler_global,
        returnLocalAbsorbed=0)

    WhiteBeamSlits_local = BioXAS_Main.WhiteBeamSlits.propagate(
        beam=DiamondFilter_global)

    Mirror1_global, Mirror1_local = BioXAS_Main.Mirror1.reflect(
        beam=DiamondFilter_global,
        returnLocalAbsorbed=0)

    CM_Slits_local = BioXAS_Main.CM_Slits.propagate(
        beam=Mirror1_global)

    SSRL_DCM_global, SSRL_DCM_local1, SSRL_DCM_local2 = BioXAS_Main.SSRL_DCM.double_reflect(
        beam=Mirror1_global)

    PreM2Screen_local = BioXAS_Main.PreM2Screen.expose(
        beam=SSRL_DCM_global)

    Mirror2_global, Mirror2_local = BioXAS_Main.Mirror2.reflect(
        beam=SSRL_DCM_global)

    PhotonShutter_local = BioXAS_Main.PhotonShutter.propagate(
        beam=Mirror2_global)

    DBHR1_global, DBHR1_local = BioXAS_Main.DBHR1.reflect(
        beam=Mirror2_global)

    DBHR2_global, DBHR2_local = BioXAS_Main.DBHR2.reflect(
        beam=DBHR1_global)

    JJslits_local = BioXAS_Main.JJslits.propagate(
        beam=DBHR2_global)

    SampleScreen_local = BioXAS_Main.SampleScreen.expose(
        beam=DBHR2_global)

    outDict = {
        'Wiggler_global': Wiggler_global,
        'FEMask_local': FEMask_local,
        'DiamondFilter_global': DiamondFilter_global,
        'DiamondFilter_local1': DiamondFilter_local1,
        'DiamondFilter_local2': DiamondFilter_local2,
        'WhiteBeamSlits_local': WhiteBeamSlits_local,
        'Mirror1_global': Mirror1_global,
        'Mirror1_local': Mirror1_local,
        'CM_Slits_local': CM_Slits_local,
        'SSRL_DCM_global': SSRL_DCM_global,
        'SSRL_DCM_local1': SSRL_DCM_local1,
        'SSRL_DCM_local2': SSRL_DCM_local2,
        'PreM2Screen_local': PreM2Screen_local,
        'Mirror2_global': Mirror2_global,
        'Mirror2_local': Mirror2_local,
        'PhotonShutter_local': PhotonShutter_local,
        'DBHR1_global': DBHR1_global,
        'DBHR1_local': DBHR1_local,
        'DBHR2_global': DBHR2_global,
        'DBHR2_local': DBHR2_local,
        'JJslits_local': JJslits_local,
        'SampleScreen_local': SampleScreen_local}
    return outDict
```

```{code-cell} ipython3
class CustomXRTBackend(SimBackend):
    def __init__(self, noise: bool = False):
        """Initialize XRT backend."""
        super().__init__()
        self._beamline = None
        self._limits = [[-0.6, 0.6], [-0.45, 0.45]]
        self._noise = noise

    def _ensure_beamline(self):
        if self._beamline is None:
            self._beamline = build_beamline()

    async def generate_beam(self):
        self._ensure_beamline()
        dbhr_info = await self._get_dbhr_information()
        mirror_radius = await self._get_mirror_radii()
        cm_slit = await self._get_cmslit_center()
        self._beamline.DBHR1.center[2] = dbhr_info[0]
        self._beamline.DBHR2.center[2] = dbhr_info[1]
        self._beamline.Mirror1.R = mirror_radius[0]
        self._beamline.Mirror2.R = mirror_radius[1]
        self._beamline.CM_Slits.center[2] = cm_slit
        out = run_process(self._beamline)

        lb = out["SampleScreen_local"]

        hist2d, _, _ = build_histRGB(lb, lb, isScreen=True, shape=[400, 300]  )
        image = hist2d

        if self._noise:
            import numpy as np
            image += 1e-3 * np.abs(np.random.standard_normal(size=image.shape))
        return image
    
    async def _get_dbhr_information(self) -> list[float,float]:
        dbhr_information = [76.42861713167648, 77.61464367739664]
        for name, device in self._device_states.items():
            if device["type"] == "oes_xrt":
                state = await self._get_device_state(name)
        return dbhr_information
    
    async def _get_cmslit_center(self) -> float:
        dbhr_information = 5.378324617301544
        for name, device in self._device_states.items():
            if device["type"] == "slit_xrt":
                state = await self._get_device_state(name)
        return dbhr_information
    
    async def _get_mirror_radii(self) -> list[float]:
        """Get KB mirror radii from registered devices.

        Returns:
            [R1, R2] where R1 is first mirror (vertical), R2 is second mirror (horizontal)
        """
        # Default radii from xrt_kb_model.py
        radii = [7120000.0, 2500000.0]

        for name, device in self._device_states.items():
            if device["type"] == "kb_mirror_xrt":
                state = await self._get_device_state(name)
                mirror_index = state["mirror_index"]
                radius = state["radius"]
                if mirror_index < len(radii):
                    radii[mirror_index] = radius

        return radii
```

```{code-cell} ipython3
class BioXASEvaluation(EvaluationFunction):
    """
    Extracts total flux and illuminated footprint area from SampleScreen_local.

    The xrt backend stores a 2-D inten# After the warm-up it is good practice to run Ax's built-in diagnostics before continuing:sity array (photons/s/mm² per bin) at
    each detector exposure.  We integrate over all bins to get total flux and
    count the bins above a threshold to estimate footprint area.
    """

    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client

    def _compute_stats(self, image: np.array) -> tuple[str, str, str]:
        """Compute integrated intensity and beam width/height from a beam image."""
        # Convert to grayscale
        gray = image.squeeze()
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        # Convert data type for numerical stability
        gray = gray.astype(np.float32)

        # Smooth w/ (5, 5) kernel and threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        max_val = np.max(blurred)
        if max_val == 0:
            return 0.0, 0.0, 0.0
 
        thresh_value = 0.2 * max_val
        _, thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_TOZERO)

        # Total integrated intensity
        total_intensity = np.sum(thresh)

        # Beam width/height from intensity-weighted second moment (σ)
        total_weight = np.sum(thresh)
        if total_weight <= 0:
            return total_intensity, 0.0, 0.0

        h, w = thresh.shape
        y_coords = np.arange(h, dtype=np.float32)
        x_coords = np.arange(w, dtype=np.float32)

        x_bar = np.sum(x_coords * np.sum(thresh, axis=0)) / total_weight
        y_bar = np.sum(y_coords * np.sum(thresh, axis=1)) / total_weight

        x_var = np.sum((x_coords - x_bar) ** 2 * np.sum(thresh, axis=0)) / total_weight
        y_var = np.sum((y_coords - y_bar) ** 2 * np.sum(thresh, axis=1)) / total_weight

        width = 2 * np.sqrt(x_var)   # ~2σ width
        height = 2 * np.sqrt(y_var)   # ~2σ height

        return total_intensity, width, height

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        outcomes = []
        run1 = self.tiled_client[uid]

        images = run1["primary/det_image"].read()
        suggestion_ids = [
            s["_id"]
            for s in run1.metadata["start"]["blop_suggestions"]
        ]

        for idx, sid in enumerate(suggestion_ids):
            intensity, width, height = self._compute_stats(images[idx])
            outcomes.append(
                {
                    "_id": sid,
                    "intensity": intensity,
                    "width": width,
                    "height": height,
                }
            )

        return outcomes
```

```{code-cell} ipython3
DBHR1_PITCH_BOUNDS = (0,39900)
DBHR2_PITCH_BOUNDS = (0, 49900)
VERTICAL_BOUNDS = (25000, 5000000)    # Optimal ~38000 is in upper portion
HORIZONTAL_BOUNDS = (25000, 5000000)   # Optimal ~21000 is in lower portion
CM_SLIT_BOUNDS = (0,200)

# DBHR1_PITCH_BOUNDS = (-0.5,0.5)
# DBHR2_PITCH_BOUNDS = (-0.5,0.5)
# VERTICAL_BOUNDS = (-0.5,0.5)    # Optimal ~38000 is in upper portion
# HORIZONTAL_BOUNDS = (-0.5,0.5)   # Optimal ~21000 is in lower portion
# CM_SLIT_BOUNDS = (-0.5,0.5)

backend = CustomXRTBackend()

det = DetectorDevice(backend, StaticPathProvider(UUIDFilenameProvider(), PurePath(DETECTOR_STORAGE)), name="det")
dbhr1 = DBHR(backend, optic_index=1, center=77, name = "dbhr1")
dbhr2 = DBHR(backend, optic_index=2, center=77, name = "dbhr2")
mirror1 = KBMirror(backend, mirror_index=1, initial_radius=2500000, name="mirror1")
mirror2 = KBMirror(backend, mirror_index=2, initial_radius=2500000, name="mirror2")
cmslit = Slit(backend, center= 0.0, name = "cmslit")

dofs = [
    RangeDOF(actuator=dbhr1.center, bounds = DBHR1_PITCH_BOUNDS, parameter_type = "float"),
    RangeDOF(actuator=dbhr2.center, bounds = DBHR2_PITCH_BOUNDS, parameter_type = "float"),
    RangeDOF(actuator=mirror1.radius, bounds=VERTICAL_BOUNDS, parameter_type="float"),
    RangeDOF(actuator=mirror2.radius, bounds=HORIZONTAL_BOUNDS, parameter_type="float"),
    RangeDOF(actuator=cmslit.center, bounds = CM_SLIT_BOUNDS, parameter_type = "float"),
]


objectives = [
    Objective(name="intensity", minimize=False),
    Objective(name="width", minimize=True),
    Objective(name="height", minimize=True)
]

agent = Agent(
    sensors=[det],
    dofs=dofs,
    objectives=objectives,
    evaluation_function=BioXASEvaluation(tiled_client),
    name="bioxas-blop-demo",
    description=("test"),
    experiment_type="demo",
)
```

```{code-cell} ipython3
RE(agent.optimize(25))
```

```{code-cell} ipython3
### Checking Optimization Health

# agent.ax_client.compute_analyses()
```

```{code-cell} ipython3
agent.ax_client.summarize()
```

```{code-cell} ipython3
RE(agent.optimize(35))
```

```{code-cell} ipython3
# agent.ax_client._experiment.parameters
```

```{code-cell} ipython3
_ = agent.plot_objective(x_dof_name="dbhr1-center", y_dof_name="dbhr2-center", objective_name="intensity")
```

```{code-cell} ipython3
optimal_parameters = next(iter(agent.ax_client.get_pareto_frontier()))[0]
optimal_parameters

# optimal_parameters = agent.ax_client.get_best_parameterization()[0]
```

```{code-cell} ipython3
from bluesky.plans import list_scan

uid = RE(list_scan(
    [det],
    dbhr1.center, [optimal_parameters[dbhr1.center.name]],
    dbhr2.center, [optimal_parameters[dbhr2.center.name]],
    mirror1.radius, [optimal_parameters[mirror1.radius.name]],
    mirror2.radius, [optimal_parameters[mirror2.radius.name]],
    cmslit.center, [optimal_parameters[cmslit.center.name]],
))

```

```{code-cell} ipython3
image = tiled_client[uid[0]]["primary/det_image"].read().squeeze()
plt.imshow(image)
plt.colorbar()
plt.show()
```
