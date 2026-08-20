.. testsetup::
    
    from unittest.mock import MagicMock
    from typing import Any
    import time

    from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
    from tiled.client.container import Container

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

    db = MagicMock(spec=Container)

Set Stopping Conditions
=======================

This guide shows how to set stopping conditions for an optimization run for the Ax optimizer backend. 
These conditions can be a variety of methods such as meeting a certain toleration, reaching a maximum number of iterations, or exceeding a time limit

Defining Stopping Conditions
----------------------------

Ax has built-in stopping conditions, `found here <https://github.com/facebook/Ax/blob/6e711c86a162f8fc5173b0d5e18f0326c329bac7/ax/global_stopping/strategies/__init__.py>`_.

However, if you need a more customizable stopping condition, you can configure one following this format.

.. code-block:: python

    from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy
    from ax.core.experiment import Experiment

    class CustomStoppingStrategy(BaseGlobalStoppingStrategy):
        def __init__(
            self,
            min_trials: int = 1,
            inactive_when_pending_trials: bool = True,
            # Add any additional parameters you want to customize here
        ) -> None:
            super().__init__(
                min_trials=min_trials, 
                inactive_when_pending_trials=inactive_when_pending_trials
            )
            # Initialize your custom parameters here

        def _should_stop_optimization(self, experiment: Experiment, **kwargs) -> tuple[bool, str]:
            # Implement your custom stopping logic here
            if your_condition_is_met:
                return True, "stopping criteria met"
            return False, "stopping criteria not met"


Adding Stopping Conditions to the Agent
---------------------------------------

The Agent can then be configured with an additional parameter, `stopping_strategy`, to use the custom stopping condition.
