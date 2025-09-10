---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: dev
  language: python
  name: python3
---

# Simulated KB Mirror Demo using Ax and Blop

This notebook introduces the use of the [Ax Adaptive Experimentation Platform](https://ax.dev) with integrations for Blop.

Blop integrates the following into Ax:
- Running Bluesky plans using the run engine
- Using devices as parameters
- Using detectors to produce data
- Retrieving the results from databroker

These features make it simple to optimize your beamline using both the [Bluesky ecosystem](https://blueskyproject.io) and Ax.

+++

## Preparing a test environment

Here we prepare the `RunEngine`

```{code-cell} ipython3
from datetime import datetime

import bluesky.plan_stubs as bps  # noqa F401
import bluesky.plans as bp  # noqa F401
import databroker  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import tiled.client.container
from bluesky.callbacks import best_effort
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree  # type: ignore[import-untyped]
from tiled.client import from_uri  # type: ignore[import-untyped]
from tiled.server import SimpleTiledServer

from blop.data_access import DatabrokerDataAccess, TiledDataAccess
from blop.sim import HDF5Handler
from blop.sim.beamline import DatabrokerBeamline, TiledBeamline

DETECTOR_STORAGE = "/tmp/blop/sim"
```

```{code-cell} ipython3
tiled_server = SimpleTiledServer(readable_storage=[DETECTOR_STORAGE])
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)


def setup_re_env(db_type="default", root_dir="/default/path", method="tiled"):
    RE = RunEngine({})
    bec = best_effort.BestEffortCallback()
    RE.subscribe(bec)
    _ = make_dir_tree(datetime.now().year, base_path=root_dir)
    if method.lower() == "tiled":
        RE.subscribe(tiled_writer)
        return {"RE": RE, "db": tiled_client, "bec": bec}
    elif method.lower() == "databroker":
        db = Broker.named(db_type)
        db.reg.register_handler("HDF5", HDF5Handler, overwrite=True)
        try:
            databroker.assets.utils.install_sentinels(db.reg.config, version=1)
        except Exception:
            pass
        RE.subscribe(db.insert)
        return {"RE": RE, "db": db, "bec": bec, }
    else:
        raise ValueError("The method for data storage used is not supported")


def register_handlers(db, handlers):
    for handler_spec, handler_class in handlers.items():
        db.reg.register_handler(handler_spec, handler_class, overwrite=True)


env = setup_re_env(db_type="temp", root_dir="/tmp/blop/sim", method="tiled")
globals().update(env)
bec.disable_plots()
```

## Simulated beamline with KB mirror pair

Here we describe an analytical simulated beamline with a [KB mirror](https://en.wikipedia.org/wiki/Kirkpatrick%E2%80%93Baez_mirror) pair.

```{code-cell} ipython3
if isinstance(db, tiled.client.container.Container):
    beamline = TiledBeamline(name="bl")
elif isinstance(db, databroker.v1.Broker):
    beamline = DatabrokerBeamline(name="bl")

beamline.det._root_dir = DETECTOR_STORAGE
beamline.det.noise.put(False)
```

## Create a Blop-Ax experiment

Now we can define the experiment we plan to run.

This involves setting 4 parameters that simulate motor positions controlling two KB mirrors. The objectives of the experiment are to maximize the beam intensity while minimizing the area of the beam.

```{code-cell} ipython3
from ax.service.ax_client import AxClient, ObjectiveProperties

from blop.integrations.ax.helpers import create_blop_experiment

ax_client = AxClient()
create_blop_experiment(
    ax_client,
    parameters=[
        {
            "movable": beamline.kbv_dsv,
            "type": "range",
            "bounds": [-5.0, 5.0],
        },
        {
            "movable": beamline.kbv_usv,
            "type": "range",
            "bounds": [-5.0, 5.0],
        },
        {
            "movable": beamline.kbh_dsh,
            "type": "range",
            "bounds": [-5.0, 5.0],
        },
        {
            "movable": beamline.kbh_ush,
            "type": "range",
            "bounds": [-5.0, 5.0],
        },
    ],
    objectives={
        "beam_intensity": ObjectiveProperties(minimize=False, threshold=200.0),
        "beam_area": ObjectiveProperties(minimize=True, threshold=1000.0),
    },
)
```

## Create an evaluation function

Now that we have setup the experiment, we need to define how to compute the objective values.

In this example, the `RunEngine` produces readings from the detector that are retrieved from `Databroker` and transformed into a Pandas `DataFrame`. Using the image produced from this, we can compute some statistics from the image to produce the beam intensity and beam area (our objectives).

Ax expects a `tuple[float, float]` for each objective representing the mean value and standard error, respectively. For a single image, the average intensity is just the intensity (same for the area), and we assume no uncertainty.

```{code-cell} ipython3
import numpy as np

from blop.utils import get_beam_stats


def evaluate(results_df: dict) -> dict[str, tuple[float, float]]:
    stats = get_beam_stats(np.asarray(results_df["bl_det_image"][0], dtype=np.float32))
    area = stats["wid_x"] * stats["wid_y"]
    return {
        "beam_intensity": (stats["sum"], None),
        "beam_area": (area, None),
    }
```

## Create an evaluator

We need a Bluesky evaluator that actually launches the experiment using the `RunEngine` and retreives the result using `Databroker`. Here we need to specify which detectors will produce the image as well as which motors we will be moving. Also, we pass the evaulation function here to produce the objective values.

This evaluator will be used to produce the raw data needed by Ax to optimize the parameters to satisfy our objectives.

```{code-cell} ipython3
from blop.integrations.ax.helpers import create_bluesky_evaluator

evaluator = create_bluesky_evaluator(
    RE, db, [beamline.det], [beamline.kbv_dsv, beamline.kbv_usv, beamline.kbh_dsh, beamline.kbh_ush], evaluate
)
```

## Optimize!

Finally, with all of our experimental setup done, we can optimize the parameters to satisfy our objectives.

For this example, Ax will optimize the 4 motor positions to produce the greatest intensity beam with the smallest beam width and height (smallest area). It does this by first running a couple of `Trial`s which are random samples, then the remainder using Bayesian optimization through BoTorch.

A single Ax `Trial` represents the training and evaluation of BoTorch models to produce a suggested next `Arm`. An `Arm` in Ax is a single parameterization to be evaluated while a `Trial` can consist of many `Arm`s. In this demo, we have a single `Arm` per `Trial`.

```{code-cell} ipython3
for _ in range(25):
    parameterization, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluator(parameterization))
```

## Analyze Results

Below we will show how we can use Ax to visualize the results and retrieve each step of the experiment that was run. This is where Ax becomes extremely useful for beamline optimization.

```{code-cell} ipython3
ax_client.experiment.to_df()
```

### Viewing slices of parameter space

```{code-cell} ipython3
from ax.plot.slice import plot_slice
from ax.utils.notebook.plotting import render

model = ax_client.generation_strategy.model
render(plot_slice(model, "bl_kbv_dsv", "beam_intensity"))
```

```{code-cell} ipython3
render(plot_slice(model, "bl_kbv_dsv", "beam_area"))
```

### Viewing each arm's objective values

```{code-cell} ipython3
from ax.plot.scatter import interact_fitted

render(interact_fitted(model, rel=False))
```

### Visualizing the optimal beam

Below we get the optimal parameters, move the motors to their optimal positions, and observe the resulting beam.

```{code-cell} ipython3
optimal_arm = next(iter(ax_client.get_pareto_optimal_parameters()))
optimal_parameters = ax_client.get_trial(optimal_arm).arm.parameters
optimal_parameters
```

```{code-cell} ipython3
from bluesky.plans import list_scan

scan_motor_params = []
for motor in [beamline.kbv_dsv, beamline.kbv_usv, beamline.kbh_dsh, beamline.kbh_ush]:
    scan_motor_params.append(motor)
    scan_motor_params.append([optimal_parameters[motor.name]])
(uid, ) = RE(list_scan([beamline.det], *scan_motor_params))
```

```{code-cell} ipython3
if isinstance(db, tiled.client.container.Container):
    data_access = TiledDataAccess(db)
elif isinstance(db, databroker.Broker):
    data_access = DatabrokerDataAccess(db)
else:
    raise ValueError("Cannot run acquisition without databroker or tiled instance!")
```

```{code-cell} ipython3
plt.imshow(data_access.get_data(uid)["bl_det_image"][0])
plt.colorbar()
plt.show()
```
