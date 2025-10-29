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

```{code-cell} ipython3
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.modelbridge.registry import Generators
from ax.models.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec

from typing import Optional
import gpytorch
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.datasets import SupervisedDataset

from gpytorch.models import ExactGP
from torch import Tensor
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

from typing import Any
from ax.utils.stats.model_fit_stats import MSE
```

```{code-cell} ipython3
from gpytorch.likelihoods import MultitaskGaussianLikelihood
class MultitaskCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 2

    def __init__(
        self, 
        train_X: Tensor, 
        train_Y: Tensor, 
        train_Yvar: Optional[Tensor] = None,
        **kwargs: Any
    ):
        # 1. Use the MultitaskGaussianLikelihood
        likelihood = MultitaskGaussianLikelihood(num_tasks=self._num_outputs)

        super().__init__(
            train_X, 
            train_Y,
            likelihood 
        )

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=self._num_outputs
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_X.shape[-1]), 
            num_tasks=self._num_outputs, 
            rank=1
        )
        print(train_X)
        print(train_Y)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.to(train_X)

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
    ) -> dict[str, Tensor]:
        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "train_Yvar": training_data.Yvar,
        }
 
```

```{code-cell} ipython3
generation_strategy = GenerationStrategy(
    name="Hybrid Strategy",
    nodes=[
        GenerationNode(
                node_name="Sobol",
                model_specs=[
                    GeneratorSpec(model_enum=Generators.SOBOL, model_kwargs={"seed": 0}),
                ],
                transition_criteria=[
                    MinTrials(
                        threshold=10,
                        transition_to="BoTorch_Custom_Only",
                        use_all_trials_in_exp=True,
                    ),
                ],
            ),
        GenerationNode(
            node_name="BoTorch_Custom_Only",
            model_specs=[GeneratorSpec(
                model_enum=Generators.BOTORCH_MODULAR,
                model_kwargs={
                    "surrogate_spec": SurrogateSpec(
                        model_configs=[
                            ModelConfig(
                                botorch_model_class=MultitaskCustomGP,
                                input_transform_classes=None,
                            )
                        ]
                    ),
                }
            )],
        )
    ]
)
```

```{code-cell} ipython3
client = Client()

client.configure_experiment(
    name="multitask_custom_only_client",
    parameters=[
        RangeParameterConfig(name="x1", bounds=(-10.0, 10.0), parameter_type="float"),
        RangeParameterConfig(name="x2", bounds=(-10.0, 10.0), parameter_type="float"),
    ],
)

client.configure_optimization(objective="-booth, -three_hump")
client.set_generation_strategy(generation_strategy)

# for _ in range(20):
#     trials = client.get_next_trials(max_trials=1)
#     print(trials)

#     # Use higher value of `max_trials` to run trials in parallel.
#     for trial_index, parameters in trials.items():
#         print(parameters)
#         print("\n")
#         client.complete_trial(
#             trial_index=trial_index,
#             raw_data={
#             "booth": (parameters["x1"] + 2 * parameters["x2"] - 7) ** 2 + (2 * parameters["x1"] + parameters["x2"] - 5) ** 2,
#             "three_hump" : (2*parameters["x1"]**2 -1.05*parameters["x1"]**4 + parameters["x1"]**6/6 + parameters["x2"]*parameters["x1"] + parameters["x2"]**2)
#             },
sobol_results = {}

for i in range(20):  # 20 Sobol trials
    trials = client.get_next_trials(max_trials=1)  # One trial per iteration
    for trial_index, parameters in trials.items():
        sobol_results[i] = {
            "booth": {"x1": parameters["x1"], "x2": parameters["x2"]},
            "three_hump": {"x1": parameters["x1"], "x2": parameters["x2"]},
        }
        client.complete_trial(
            trial_index=trial_index,
            raw_data={
                "booth": (parameters["x1"] + 2 * parameters["x2"] - 7) ** 2 + (2 * parameters["x1"] + parameters["x2"] - 5) ** 2,
                "three_hump": (2*parameters["x1"]**2 -1.05*parameters["x1"]**4 + parameters["x1"]**6/6 + parameters["x2"]*parameters["x1"] + parameters["x2"]**2),
            },
        )

print(sobol_results)
# booth is at (1, 3).
# three-hump minimum at (0,0)
client.get_best_parameterization()
```
