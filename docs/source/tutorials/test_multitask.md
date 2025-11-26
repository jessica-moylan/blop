---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: dev
  language: python
  name: python3
---

```{code-cell} ipython3
import time
from datetime import datetime

import bluesky.plan_stubs as bps  # noqa F401
import bluesky.plans as bp  # noqa F401
import databroker  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
from bluesky.callbacks import best_effort
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree  # type: ignore[import-untyped]
from tiled.client import from_uri  # type: ignore[import-untyped]
from tiled.server import SimpleTiledServer

from blop.sim import HDF5Handler

DETECTOR_STORAGE = "/tmp/blop/sim"
tiled_server = SimpleTiledServer(readable_storage=[DETECTOR_STORAGE])
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)


def setup_re_env(db_type="default", root_dir="/default/path", method="tiled"):
    RE = RunEngine({})
    bec = best_effort.BestEffortCallback()
    RE.subscribe(bec)
    _ = make_dir_tree(datetime.now().year, base_path=root_dir)

    if method == "tiled":
        RE.subscribe(tiled_writer)
        return {"RE": RE, "db": tiled_client, "bec": bec}

    elif method == "databroker":
        db = Broker.named(db_type)
        db.reg.register_handler("HDF5", HDF5Handler, overwrite=True)
        try:
            databroker.assets.utils.install_sentinels(db.reg.config, version=1)
        except Exception:
            pass
        RE.subscribe(db.insert)
        return {
            "RE": RE,
            "db": db,
            "bec": bec,
        }
    else:
        raise ValueError("The method for data storage used is not supported")


def register_handlers(db, handlers):
    for handler_spec, handler_class in handlers.items():
        db.reg.register_handler(handler_spec, handler_class, overwrite=True)


env = setup_re_env(db_type="temp", root_dir="/tmp/blop/sim", method="tiled")
globals().update(env)
bec.disable_plots()
```

```{code-cell} ipython3
from typing import Any
import time
import logging
from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
from bluesky.run_engine import RunEngine
from bluesky.callbacks.tiled_writer import TiledWriter
from tiled.server import SimpleTiledServer

class AlwaysSuccessfulStatus(Status):
    def add_callback(self, callback) -> None:
        callback(self)
    def exception(self, timeout = 0.0):
        return None
    
    @property
    def done(self) -> bool:
        return True
    
    @property
    def success(self) -> bool:
        return True
class ReadableSignal(Readable, HasHints, HasParent):
    def __init__(self, name: str) -> None:
        self._name = name
        self._value = 0.0
    @property
    def name(self) -> str:
        return self._name
    @property
    def hints(self) -> Hints:
        return { 
            "fields": [self._name],
            "dimensions": [],
            "gridding": "rectilinear",
        }
    
    @property
    def parent(self) -> Any | None:
        return None
    def read(self):
        return {
            self._name: { "value": self._value, "timestamp": time.time() }
        }
    def describe(self):
        return {
            self._name: { "source": self._name, "dtype": "number", "shape": [] }
        }
class MovableSignal(ReadableSignal, NamedMovable):
    def __init__(self, name: str, initial_value: float = 0.0) -> None:
        super().__init__(name)
        self._value: float = initial_value
    def set(self, value: float) -> Status:
        self._value = value
        return AlwaysSuccessfulStatus()
```

```{code-cell} ipython3
import numpy as np
import torch
import gpytorch
from blop.ax import Agent
from blop.dofs import DOF
from blop.objectives import Objective
from botorch.models.multitask import MultiTaskGP

from ax.modelbridge.registry import Generators
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec
from ax.models.torch.botorch_modular.model import ModelConfig


# Create a proper MultitaskGPModel that accepts task_feature parameter
class CustomMultitaskGPModel(MultiTaskGP):
    def __init__(self, train_X, train_Y, task_feature, likelihood=None, **kwargs):
        # Pass task_feature to the parent MultiTaskGP class
        super().__init__(train_X, train_Y, task_feature, likelihood=likelihood, **kwargs)
        
        # Determine number of tasks from the unique values in the task feature column
        num_tasks = int(train_X[:, task_feature].unique().numel())
        
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )
        self.trained = False


# Define your degrees of freedom (input parameters)
dof1 = MovableSignal("dof1")
dof2 = MovableSignal("dof2")
# Add a task DOF - this tells the model which objective we're evaluating
task_dof = MovableSignal("task", initial_value=0.0)

class ObjectiveDevice(ReadableSignal):
    def __init__(self, name: str, dof1: MovableSignal, dof2: MovableSignal, func) -> None:
        super().__init__(name)
        self.dof1 = dof1
        self.dof2 = dof2
        self.func = func
    
    def read(self):
        x1 = self.dof1._value
        x2 = self.dof2._value
        objective_value = self.func(x1, x2)
        return {
            self._name: { "value": objective_value, "timestamp": time.time()}
        }

# Define your objective functions
def booth_function(x1, x2):
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2

def three_hump_function(x1, x2):
    return 2*x1**2 - 1.05*x1**4 + x1**6/6 + x2*x1 + x2**2


# Create objective devices that compute based on DOF values
readable1 = ObjectiveDevice("booth", dof1, dof2, booth_function)
readable2 = ObjectiveDevice("three_hump", dof1, dof2, three_hump_function)


dofs = [
    DOF(movable=dof1, search_domain=(-5.0, 5.0)),
    DOF(movable=dof2, search_domain=(-5.0, 5.0)),
    # Add task as a choice parameter with values 0 and 1 (one for each objective)
    DOF(movable=task_dof, search_domain=[0.0, 1.0], type="ordinal"),
]

# the tasks
objectives = [
    Objective(name="booth", target="min"),
    Objective(name="three_hump", target="min"),
]
```

# Multi-Task Optimization with MultitaskGPModel

This notebook demonstrates how to use multi-task Gaussian Process models for optimizing multiple objectives simultaneously.

## Key Concepts:

1. **Task Feature**: A special DOF that indicates which task/objective is being evaluated
2. **Task Index**: Maps to specific objectives (0 for booth, 1 for three_hump)
3. **Multi-Task GP**: Learns correlations between tasks to improve sample efficiency

## Setup Requirements:

- Add a `task` DOF as an ordinal parameter with values corresponding to each objective
- Create a custom MultitaskGPModel that properly accepts the `task_feature` parameter
- Configure the generation strategy to specify which DOF column is the task feature
- The task feature should be the last DOF (index 2 in this case with 3 DOFs total)

```{code-cell} ipython3
generation_strategy = GenerationStrategy(
    name="MultiTask with CustomMultitaskGPModel",
    nodes=[
        GenerationNode(
            node_name="Sobol",
            model_specs=[GeneratorSpec(model_enum=Generators.SOBOL)],
            transition_criteria=[MinTrials(threshold=10, transition_to="MultiTaskGP", use_all_trials_in_exp=True)],
        ),
        GenerationNode(
            node_name="MultiTaskGP",
            model_specs=[
                GeneratorSpec(
                    model_enum=Generators.BOTORCH_MODULAR,
                    model_kwargs={
                        "surrogate_spec": SurrogateSpec(
                            model_configs=[
                                ModelConfig(
                                    botorch_model_class=CustomMultitaskGPModel,
                                    # Specify which parameter index is the task feature
                                    # Since task_dof is the 3rd DOF (index 2), set task_feature=2
                                    model_options={"task_feature": 2},
                                ),
                            ],
                        ),
                    },
                )
            ],
        ),
    ],
)
```

```{code-cell} ipython3
agent = Agent(
    readables=[readable1, readable2],
    dofs=dofs,
    objectives=objectives,
    db=db,
)
agent.configure_experiment(name="test_ax_agent", description="Test the AxAgent")
agent.set_generation_strategy(generation_strategy)
RE(agent.learn(iterations=25, n=1))
```
