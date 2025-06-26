---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: xrt-blop
  language: python
  name: python3
---

# XRT Blop Demo

+++

For ophyd beamline setup see: 
- https://github.com/NSLS-II/blop/blob/main/src/blop/sim/xrt_beamline.py
- https://github.com/NSLS-II/blop/blob/main/src/blop/sim/xrt_kb_model.py

The picture below displays beam from geometric source propagating through a pair of toroidal mirrors focusing the beam on screen. Simulation of a KB setup.

![xrt_blop_layout_w.jpg](../_static/xrt_blop_layout_w.jpg)

```{code-cell} ipython3
from blop.utils import prepare_re_env  # noqa

%run -i $prepare_re_env.__file__ --db-type=temp
bec.disable_plots()
```

```{code-cell} ipython3
import time

from matplotlib import pyplot as plt

from blop import DOF, Agent, Objective
from blop.digestion import beam_stats_digestion
from blop.sim.xrt_beamline import Beamline
```

```{code-cell} ipython3
plt.ion()

h_opt = 0
dh = 5

R1, dR1 = 40000, 10000
R2, dR2 = 20000, 10000
```

```{code-cell} ipython3
beamline = Beamline(name="bl")
time.sleep(1)
dofs = [
    DOF(description="KBV R", device=beamline.kbv_dsv, search_domain=(R1 - dR1, R1 + dR1)),
    DOF(description="KBH R", device=beamline.kbh_dsh, search_domain=(R2 - dR2, R2 + dR2)),
]
```

```{code-cell} ipython3
objectives = [
    Objective(name="bl_det_sum", target="max", transform="log", trust_domain=(20, 1e12)),
    Objective(
        name="bl_det_wid_x",
        target="min",
        transform="log",
        # trust_domain=(0, 1e12),
        latent_groups=[("bl_kbh_dsh", "bl_kbv_dsv")],
    ),
    Objective(
        name="bl_det_wid_y",
        target="min",
        transform="log",
        # trust_domain=(0, 1e12),
        latent_groups=[("bl_kbh_dsh", "bl_kbv_dsv")],
    ),
]
```

```{code-cell} ipython3
agent = Agent(
    dofs=dofs,
    objectives=objectives,
    detectors=[beamline.det],
    digestion=beam_stats_digestion,
    digestion_kwargs={"image_key": "bl_det_image"},
    verbose=True,
    db=db,
    tolerate_acquisition_errors=False,
    enforce_all_objectives_valid=True,
    train_every=3,
)
```

```{code-cell} ipython3
RE(agent.learn("qr", n=16))
RE(agent.learn("qei", n=16, iterations=4))
```

```{code-cell} ipython3
agent.plot_objectives(axes=(0, 1))
```
