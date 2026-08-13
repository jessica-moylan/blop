"""Stopping conditions that can be used to halt the optimization process when using the Ax backend."""

from ax.core.experiment import Experiment
from ax.global_stopping.strategies import BaseGlobalStoppingStrategy
from ax.global_stopping.strategies.improvement import constraint_satisfaction


class MaxEvaluationsStopping(BaseGlobalStoppingStrategy):
    """A stopping strategy that halts the optimization after a maximum number of evaluations."""

    def __init__(
        self,
        max_evaluations: int,
        min_trials: int = 1,
        inactive_when_pending_trials: bool = True,
    ) -> None:
        """
        Stop the optimization after a maximum number of evaluations.

        Args:
            max_evaluations: The maximum number of evaluations to perform before stopping.
            min_trials: The minimum number of trials to run before considering stopping.
            inactive_when_pending_trials: Whether to consider the optimization inactive when there are pending trials.
        """
        super().__init__(min_trials=min_trials, inactive_when_pending_trials=inactive_when_pending_trials)

        self.max_evaluations = max_evaluations

    def _should_stop_optimization(self, experiment: Experiment) -> tuple[bool, str]:
        """Check if the optimization should stop based on the number of evaluations."""
        completed_trials = len(experiment.completed_trials)
        if completed_trials < self.min_trials:
            return False, f"Completed trials: {completed_trials}, Min trials: {self.min_trials}"
        should_stop = completed_trials >= self.max_evaluations
        return should_stop, f"Completed trials: {completed_trials}, Max evaluations: {self.max_evaluations}"


class ValidConfigurationStopping(BaseGlobalStoppingStrategy):
    """A stopping strategy that halts the optimization when the first valid configuration is found."""

    def __init__(
        self,
        min_trials: int = 1,
        inactive_when_pending_trials: bool = True,
    ) -> None:
        """
        Stop the optimization when a valid configuration is found.

        Args:
            min_trials: The minimum number of trials to run before considering stopping.
            inactive_when_pending_trials: Whether to consider the optimization inactive when there are pending trials.
        """
        super().__init__(min_trials=min_trials, inactive_when_pending_trials=inactive_when_pending_trials)

    def _should_stop_optimization(self, experiment: Experiment) -> tuple[bool, str]:
        """Check to see if a valid configuration has been found in the completed trials."""
        completed_trials = experiment.completed_trials
        if len(completed_trials) < self.min_trials:
            return False, f"Completed trials: {len(completed_trials)}, Min trials: {self.min_trials}"

        last_index = next(reversed(experiment.trials))
        if experiment.trials[last_index].status.is_completed and constraint_satisfaction(experiment.trials[last_index]):
            return True, f"Valid configuration found in trial {experiment.trials[last_index].index}."

        return False, "No valid configuration found yet."
