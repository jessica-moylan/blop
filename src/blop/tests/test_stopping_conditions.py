from unittest.mock import MagicMock, patch

import pytest
from bluesky.run_engine import RunEngine

from blop.ax.stopping_conditions import MaxEvaluationsStopping, ValidConfigurationStopping


@pytest.fixture(scope="function")
def RE():
    return RunEngine({})


def test_stopping_condition_max_evaluations():
    """Tests that MaxEvaluationsStopping triggers once max trials are completed."""

    stopping_condition = MaxEvaluationsStopping(max_evaluations=2, min_trials=1)
    experiment = MagicMock()
    experiment.completed_trials = {0: MagicMock()}

    should_stop, reason = stopping_condition._should_stop_optimization(experiment=experiment)
    assert should_stop is False
    assert "Min trials" in reason

    experiment.completed_trials[1] = MagicMock()

    should_stop, reason = stopping_condition._should_stop_optimization(experiment=experiment)

    assert should_stop is True
    assert "Max evaluations: 2" in reason


def test_stopping_condition_valid_configuration():
    """Tests that ValidConfigurationStopping returns True for a completed valid last trial."""

    stopping_condition = ValidConfigurationStopping(min_trials=2)

    trial = MagicMock()
    trial.status.is_completed = True
    trial.index = 0
    trial.arm.parameters = {"x1": 0.0}

    experiment = MagicMock()

    experiment.completed_trials = {0: trial}
    experiment.trials = {0: trial}
    should_stop, reason = stopping_condition._should_stop_optimization(experiment=experiment)
    assert should_stop is False
    assert "Min trials" in reason

    experiment.completed_trials[1] = MagicMock()
    experiment.trials[1] = MagicMock()

    with patch("blop.ax.stopping_conditions.constraint_satisfaction", return_value=True):
        should_stop, reason = stopping_condition._should_stop_optimization(experiment=experiment)

    assert should_stop is True
    assert "Valid configuration found" in reason


def test_stopping_condition_no_valid_configuration():
    """Tests that ValidConfigurationStopping returns False when no valid configuration is found."""

    stopping_condition = ValidConfigurationStopping(min_trials=1)

    trial = MagicMock()
    trial.status.is_completed = True
    trial.index = 0
    trial.arm.parameters = {"x1": 0.0}

    experiment = MagicMock()
    experiment.completed_trials = {0: trial}
    experiment.trials = {0: trial}

    with patch("blop.ax.stopping_conditions.constraint_satisfaction", return_value=False):
        should_stop, reason = stopping_condition._should_stop_optimization(experiment=experiment)

    assert should_stop is False
    assert "No valid configuration found" in reason
