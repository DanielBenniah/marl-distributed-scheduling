# Multi-Agent Reinforcement Learning for Distributed Task Scheduling

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the paper: **"Decentralized Task Scheduling in Distributed Systems: A Deep Reinforcement Learning Approach"**

## 🎯 Overview

This repository contains the complete implementation of a lightweight Multi-Agent Reinforcement Learning (MARL) framework for distributed task scheduling. The framework uses only **NumPy** (no TensorFlow/PyTorch), making it deployable on resource-constrained edge devices.

### Key Features

- ✅ **Lightweight**: NumPy-only implementation (~19K parameters per agent)
- ✅ **Decentralized**: Multi-agent actor-critic architecture
- ✅ **Priority-Aware**: Handles Production, Batch, and Best-Effort workloads
- ✅ **Energy-Efficient**: Explicit energy consumption modeling
- ✅ **Realistic Workloads**: Based on Google Cluster Trace statistics

### Performance

On a 100-node heterogeneous system with 1,000 tasks per episode:

| Metric | MARL (Ours) | Random | Weighted-RR | Priority-MM |
|--------|-------------|--------|-------------|-------------|
| Avg Completion Time (s) | **30.8 ± 1.5** | 36.5 ± 2.1 | 36.2 ± 1.9 | 52.3 ± 3.4 |
| Energy Consumption (kWh) | **745 ± 38** | 878 ± 45 | 1007 ± 52 | 155 ± 12* |
| SLA Satisfaction (%) | **82.3 ± 2.1** | 75.5 ± 3.2 | 76.1 ± 2.8 | 47.3 ± 5.1 |
| Tasks Completed | 982 ± 10 | 985 ± 12 | 992 ± 8 | 280 ± 15* |

*Priority-MM completes only 28% of tasks, resulting in artificially low energy

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/danielbenniah/marl-distributed-scheduling.git
cd marl-distributed-scheduling

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
```

## 🚀 Quick Start

### Run Full Experiment

```bash
python simulation.py
```

This will:
1. Run 30 episodes for each scheduler (Random, Weighted-RR, Priority-MinMin, MARL)
2. Generate 1,000 tasks per episode based on Google Cluster Trace statistics
3. Simulate on 100 heterogeneous compute nodes
4. Save results to `results.json`

### Run Single Scheduler

```python
from simulation import run_experiment

# Run MARL scheduler
metrics, episode_results = run_experiment(
    scheduler_name='MARL',
    num_episodes=30,
    num_tasks=1000
)

print(f"Average completion time: {metrics['avg_completion_time']:.1f}s")
print(f"SLA satisfaction: {metrics['avg_sla']:.1f}%")
```

## 📂 Repository Structure

```
marl-distributed-scheduling/
├── marl_scheduler.py      # Core MARL implementation
├── simulation.py          # Simulation runner
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── results/              # Experimental results
    ├── results.json      # Numerical results
    └── figures/          # Generated plots
```

## 🏗️ Architecture

### Actor-Critic Network (Per Agent)

```
Input (50-dim state)
    ↓
Hidden Layer (128 neurons, ReLU)
    ↓
    ├── Actor Head → Policy (softmax)
    └── Critic Head → Value estimate
```

**Total parameters per agent**: ~19,557  
**Memory footprint**: ~78KB (float32)

### State Representation

Each agent observes:
- Own utilization, memory, queue length, capacity (5 features)
- Neighbor statistics (10 features)
- Task queue statistics (5 features)  
- Global system state (30 features)

**Total**: 50-dimensional state vector

### Reward Function

```python
reward = w_sla * SLA_reward 
       + w_completion * completion_bonus
       - w_energy * energy_cost
       - w_balance * load_variance
```

Weights: w_sla=0.4, w_completion=0.3, w_energy=0.2, w_balance=0.1

## 📊 Workload Generation

Tasks are generated using Google Cluster Trace-derived statistics:

- **Execution time**: Pareto distribution (α=2.5, t_min=10s)
- **CPU requirement**: LogNormal(μ=0.5, σ=0.8) → 1-8 cores
- **Memory requirement**: LogNormal(μ=2.0, σ=1.0) → 2-20 GB
- **Arrival**: Poisson process (λ=0.5 tasks/sec)
- **Priority**: Production (25%), Batch (60%), Best-Effort (15%)

## 🖥️ Infrastructure Model

100 heterogeneous nodes across 3 tiers:

### High-Capacity (20 nodes, 20%)
- CPU: 24-32 cores
- Memory: 96-128 GB
- Idle Power: 150-200 W
- Dynamic Power: 200-300 W

### Medium-Capacity (50 nodes, 50%)
- CPU: 8-16 cores
- Memory: 32-64 GB
- Idle Power: 50-80 W
- Dynamic Power: 60-120 W

### Low-Capacity (30 nodes, 30%)
- CPU: 2-8 cores
- Memory: 8-32 GB
- Idle Power: 15-30 W
- Dynamic Power: 20-50 W

## 🔬 Reproducing Paper Results

```bash
# Run full experiment (30 episodes × 4 schedulers)
python simulation.py

# Results will match Table I in the paper:
# - MARL: 30.8s completion, 745 kWh energy, 82.3% SLA
# - Random: 36.5s completion, 878 kWh energy, 75.5% SLA
# - Weighted-RR: 36.2s completion, 1007 kWh energy, 76.1% SLA
# - Priority-MM: 52.3s completion, 155 kWh energy*, 47.3% SLA
```

## 📈 Visualization

Generate plots:

```python
import matplotlib.pyplot as plt
import json

# Load results
with open('results.json') as f:
    results = json.load(f)

# Plot comparison
schedulers = list(results.keys())
completion_times = [results[s]['avg_completion_time'] for s in schedulers]

plt.bar(schedulers, completion_times)
plt.ylabel('Avg Completion Time (s)')
plt.title('Scheduler Comparison')
plt.savefig('comparison.png')
```

## ⚙️ Configuration

Modify simulation parameters in `simulation.py`:

```python
# Number of nodes
nodes = InfrastructureGenerator.create_nodes(num_nodes=100)

# Number of tasks per episode
tasks = workload_gen.generate_tasks(num_tasks=1000)

# Time step for discrete-event simulation
sim = Simulator(nodes, scheduler, time_step=5.0)
```

## 🧪 Extending the Framework

### Add New Scheduler

```python
class MyScheduler:
    def schedule_task(self, task, nodes, **kwargs):
        # Add custom scheduling logic here
        return selected_node_id
```

### Modify Neural Network

```python
# In marl_scheduler.py
network = ActorCriticNetwork(
    input_dim=50,      # State size
    hidden_dim=128,    # Hidden layer size
    output_dim=100,    # Number of nodes
    learning_rate=0.001
)
```

### Custom Workload

```python
class CustomWorkloadGenerator:
    def generate_tasks(self, num_tasks):
        # Add custom task generation logic here
        return tasks
```

## 📝 Citation

If you use this code, please cite our paper:

```bibtex
@article{TBD},
  title={Decentralized Task Scheduling in Distributed Systems: A Deep Reinforcement Learning Approach},
  author={Daniel Benniah John},
  journal={IEEE [TBD]},
  year={2026}
}
```

## ⚠️ Limitations

This is a **simulation-based** implementation with the following limitations:

1. **No real deployment**: Not tested on actual distributed systems
2. **Synthetic workload**: Uses statistical models, not actual trace replay
3. **Scale**: Evaluated on 100 nodes (modest compared to large datacenters)
4. **Independent tasks**: Does not handle DAG workflows or dependencies
5. **No network delays**: Simulation assumes instant communication

See paper Section VI for detailed discussion.

## 🔮 Future Work

- [ ] Deploy on CloudLab/Chameleon Cloud testbed
- [ ] Implement actual Google Cluster Trace replay
- [ ] Extend to support DAG task workflows
- [ ] Scale to 1000+ nodes
- [ ] Add federated learning for multi-cluster scenarios
- [ ] Integrate with Kubernetes scheduler

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: [danielbenniah@berkeley.edu]

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- Google for releasing the Cluster Trace dataset
- NumPy and SciPy development teams
- Anonymous reviewers for their helpful feedback

---

**Note**: Code will be made fully public upon paper acceptance. Current version is for review purposes.
