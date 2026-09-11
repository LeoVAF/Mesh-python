# A-MESH

A-MESH is an adaptive multi-objective optimization algorithm that combines
swarm-based search, Differential Evolution, and elitist selection. It is
designed to approximate a diverse set of mutually non-dominated solutions for
continuous minimization problems.

The implementation maintains a population of candidate solutions and a bounded
external memory of non-dominated solutions. During the search, it:

- generates candidates with configurable Differential Evolution strategies;
- selects personal and global guides for the swarm;
- preserves promising solutions using non-dominated sorting and crowding
  distance;
- adapts Differential Evolution and swarm parameters from successful updates;
- supports stopping by generation count, fitness-evaluation count, or both;
- optionally evaluates the objective function in parallel.

In prose, the algorithm is named **A-MESH**. Its Python package and public
classes use the code identifiers `amesh`, `AMESH`, and `AMESHParameters`.

## Requirements and installation

The supported environment uses Python 3.11. Conda is recommended because the
project depends on scientific packages distributed through both Conda and pip.

Create the complete development environment from the repository root:

```console
conda env create --file environment.yml
conda activate amesh
```

The environment file installs the project in editable mode together with the
runtime, experimentation, testing, and documentation dependencies.

## Basic usage

An objective function receives one decision vector and returns one value for
each objective. A-MESH follows the minimization convention; negate an objective
inside the function if it must be maximized.

```python
import numpy as np

from amesh import AMESH, AMESHParameters


def objectives(x: np.ndarray) -> np.ndarray:
    return np.array([
        np.sum(x**2),
        np.sum((x - 1.0) ** 2),
    ])


decision_dim = 5
params = AMESHParameters(
    objective_dim=2,
    decision_dim=decision_dim,
    decision_lower_bounds=np.full(decision_dim, -5.0),
    decision_upper_bounds=np.full(decision_dim, 5.0),
    population_size=40,
    max_fit_eval=4_000,
    random_state=42,
)

amesh = AMESH(params, objectives)
amesh.run()
positions, objective_values = amesh.get_results()
```

`positions` contains the decision vectors retained in the external memory, and
`objective_values` contains their corresponding objective vectors.

### Main configuration options

| Parameter | Purpose |
| --- | --- |
| `objective_dim` | Number of objectives; at least two are required. |
| `decision_dim` | Number of decision variables. |
| `decision_lower_bounds`, `decision_upper_bounds` | Per-variable search bounds. |
| `population_size` | Number of particles and maximum external-memory size. |
| `global_guide_method` | Global-guide selection strategy. |
| `dm_pool_type` | Source pool used by Differential Evolution. |
| `dm_operation_type` | Differential mutation strategy. |
| `max_gen` | Optional maximum number of generations. |
| `max_fit_eval` | Optional maximum number of objective-function evaluations. |
| `max_personal_guides` | Maximum number of personal guides per particle. |
| `initial_points` | Optional initial population. |
| `random_state` | Optional seed for reproducible random sampling. |

At least one of `max_gen` and `max_fit_eval` must be provided. See the generated
API reference for the available strategy values and complete parameter
contracts.

For expensive objective functions, evaluations can be distributed across
processes:

```python
amesh = AMESH(params, objectives, num_proc=4)
```

## Generating the HTML documentation

The documentation uses Sphinx, MyST-NB, Sphinx AutoAPI, and the Sphinx Book
Theme. All required packages are included in `environment.yml`.

On Linux or macOS, run from the repository root:

```console
conda activate amesh
make -C docs html
```

On Windows, run:

```console
conda activate amesh
cd docs
make.bat html
```

The same build can be launched directly on any supported platform:

```console
python -m sphinx -M html docs docs/_build
```

Open `docs/_build/html/index.html` after the build completes. To force a clean
rebuild on Linux or macOS, run `make -C docs clean html`.

The generated site includes this overview, a usage example, project policies,
the changelog, and an API reference produced automatically from the current
Python sources.

## Validation

Run the automated test suite from the repository root:

```console
conda activate amesh
pytest
```

## License

A-MESH was created by Leonardo Veiga Acioly Filho and is distributed under the
MIT License.

## Credits

The original project structure and HTML-documentation scaffold were generated
with [Cookiecutter](https://cookiecutter.readthedocs.io/en/latest/) and the
[`py-pkgs-cookiecutter`](https://github.com/py-pkgs/py-pkgs-cookiecutter)
template.
