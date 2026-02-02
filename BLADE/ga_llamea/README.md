# GA-LLAMEA

**GA-LLAMEA: LLaMEA with Discounted Thompson Sampling for Adaptive Operator Selection**

A standalone package that integrates seamlessly with BLADE with **zero code changes** required to BLADE.

## Features

- ✅ **Discounted Thompson Sampling (D-TS)** for adaptive operator selection
- ✅ **Three genetic operators**: mutation, crossover, random_new
- ✅ **Zero BLADE changes required** - uses Protocol-based interfaces
- ✅ **No external dependencies** - pure Python implementation
- ✅ **Drop-in replacement** for other BLADE methods

## Quick Start

### Installation

Simply copy/clone the `ga_llamea` folder to your project or add it to your Python path.

### Usage with BLADE

```python
from iohblade.llm import LLM
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB
from iohblade.experiment import Experiment

# Import GA-LLAMEA
from ga_llamea import GA_LLaMEA

# Initialize LLM
llm = LLM(model="gemini-2.0-flash")

# Create GA-LLAMEA method
method = GA_LLaMEA(
    llm=llm,
    budget=100,
    solution_class=Solution,  # Pass BLADE's Solution class
    n_parents=4,
    n_offspring=16,
    elitism=True,
    discount=0.9,
)

# Run on a problem
problem = MA_BBOB(function_id=1, dimension=5, instance=1)
experiment = Experiment(problem=problem, method=method)
results = experiment.run()

### Usage with Custom Endpoints (AIML, etc.)

GA-LLAMEA works with any OpenAI-compatible API (like AIML API) by using the `base_url` parameter in BLADE's `LLM` class:

```python
from iohblade.llm import LLM
from ga_llamea import GA_LLaMEA

# Initialize with AIML endpoint
llm = LLM(
    api_key="your-aiml-key",
    model="gpt-4o-mini",
    base_url="https://api.aimlapi.com/v1"
)

# Use with GA-LLAMEA
method = GA_LLaMEA(
    llm=llm, 
    budget=100, 
    solution_class=Solution
)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm` | required | LLM instance (BLADE's `LLM` class) |
| `budget` | required | Total LLM queries allowed |
| `solution_class` | None | Solution class to use (pass BLADE's `Solution`) |
| `n_parents` | 4 | Population size (μ) |
| `n_offspring` | 16 | Offspring per generation (λ) |
| `elitism` | True | Use (μ+λ) selection if True |
| `discount` | 0.9 | D-TS discount factor γ ∈ (0, 1] |
| `tau_max` | 1.0 | Maximum posterior uncertainty |
| `reward_variance` | 1.0 | Expected reward variance |

## Algorithm Overview

GA-LLAMEA adaptively selects between three operators using a multi-armed bandit:

```
┌────────────────────────────────────────────────────────┐
│              GA-LLAMEA Pipeline                        │
├────────────────────────────────────────────────────────┤
│  1. D-TS Bandit samples θᵢ ~ N(μ̂ᵢ, τᵢ²) for each arm   │
│  2. Select operator = argmax(θᵢ)                       │
│  3. Generate offspring using selected operator         │
│     • Mutation: Improve single parent                  │
│     • Crossover: Combine two parents                   │
│     • Random New: Generate novel algorithm             │
│  4. Evaluate offspring fitness                         │
│  5. Compute reward = max(0, child - parent fitness)    │
│  6. Update D-TS bandit with discounted statistics      │
│  7. Selection: Keep best μ individuals                 │
└────────────────────────────────────────────────────────┘
```

## Why Zero BLADE Changes?

This package uses **Python Protocols** (PEP 544) for type compatibility:

- `LLMProtocol` - Expects a `query(prompt) -> str` method
- `SolutionProtocol` - Expects `code`, `fitness`, `error`, `metadata` attributes  
- `ProblemProtocol` - Expects `evaluate()`, `task_prompt`, `format_prompt`

BLADE's classes satisfy these protocols automatically, so no changes are needed!

## Files

```
ga_llamea/
├── __init__.py       # Package exports
├── ds_ts.py          # Discounted Thompson Sampling bandit
├── ga_llamea.py      # Main GA-LLAMEA implementation
├── interfaces.py     # Protocol definitions
├── requirements.txt  # Dependencies (none required)
├── README.md         # This file
├── examples/
│   └── integrate_with_blade.py
└── tests/
    └── test_ga_llamea.py
```

## Testing

```bash
cd ga_llamea
python -m pytest tests/ -v
```

## License

MIT License - Same as BLADE
