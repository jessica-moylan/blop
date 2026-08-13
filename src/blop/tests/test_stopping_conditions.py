from unittest.mock import MagicMock, patch

import pytest
from bluesky.run_engine import RunEngine

from blop.ax.stopping_conditions import MaxEvaluationsStopping, ValidConfigurationStopping


@pytest.fixture(scope="function")
def RE():
    return RunEngine({})


def test_stopping_condition_max_evaluations():
    """Tests that MaxEvaluationsStopping triggers once max trials are completed."""

    stopping_condition = MaxEvaluationsStopping(max_evaluations=3)
    experiment = MagicMock()
    experiment.completed_trials = {0: MagicMock(), 1: MagicMock(), 2: MagicMock()}

    should_stop, reason = stopping_condition._should_stop_optimization(experiment=experiment)

    assert should_stop is True
    assert "Max evaluations: 3" in reason


def test_stopping_condition_valid_configuration():
    """Tests that ValidConfigurationStopping returns True for a completed valid last trial."""

    stopping_condition = ValidConfigurationStopping(min_trials=1)

    trial = MagicMock()
    trial.status.is_completed = True
    trial.index = 0
    trial.arm.parameters = {"x1": 0.0}

    experiment = MagicMock()
    experiment.completed_trials = {0: trial}
    experiment.trials = {0: trial}

    with patch("blop.ax.stopping_conditions.constraint_satisfaction", return_value=True):
        should_stop, reason = stopping_condition._should_stop_optimization(experiment=experiment)

    assert should_stop is True
    assert "Valid configuration found" in reason
