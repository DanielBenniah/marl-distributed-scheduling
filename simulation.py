#!/usr/bin/env python3
"""
Simulation Runner for MARL Task Scheduling Experiments

Generates Google Cluster Trace-derived workloads and runs scheduling experiments
"""

import numpy as np
from typing import List, Tuple
import json
from marl_scheduler import *

# ============================================================================
# WORKLOAD GENERATION (Google Cluster Trace-derived)
# ============================================================================

class WorkloadGenerator:
    """Generate synthetic workload based on Google Cluster Trace statistics"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def generate_tasks(self, num_tasks=1000, arrival_rate=0.5) -> List[Task]:
        """Generate tasks with Google Cluster Trace characteristics"""
        tasks = []
        current_time = 0.0
        
        for i in range(num_tasks):
            # Arrival time (Poisson process)
            inter_arrival = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival
            
            # Execution time (Pareto distribution - heavy tail)
            alpha = 2.5
            t_min = 10.0
            execution_time = t_min * (1.0 / (np.random.uniform(0, 1) ** (1.0 / alpha)))
            execution_time = min(execution_time, 500.0)  # Cap at 500s
            
            # CPU requirement (Log-normal)
            cpu_req = np.random.lognormal(mean=0.5, sigma=0.8)
            cpu_req = np.clip(cpu_req, 0.5, 8.0)
            
            # Memory requirement (Log-normal)
            mem_req = np.random.lognormal(mean=2.0, sigma=1.0)
            mem_req = np.clip(mem_req, 1.0, 32.0)
            
            # Priority class (Production: 25%, Batch: 60%, BestEffort: 15%)
            priority_rand = np.random.random()
            if priority_rand < 0.25:
                priority = 0  # Production
                deadline_factor = 1.5
            elif priority_rand < 0.85:
                priority = 1  # Batch
                deadline_factor = 3.0
            else:
                priority = 2  # Best-effort
                deadline_factor = 5.0
            
            deadline = current_time + execution_time * deadline_factor
            
            task = Task(
                id=i,
                arrival_time=current_time,
                execution_time=execution_time,
                cpu_requirement=cpu_req,
                memory_requirement=mem_req,
                priority=priority,
                deadline=deadline
            )
            
            tasks.append(task)
        
        return tasks

# ============================================================================
# COMPUTE INFRASTRUCTURE
# ============================================================================

class InfrastructureGenerator:
    """Generate heterogeneous compute infrastructure"""
    
    @staticmethod
    def create_nodes(num_nodes=100, seed=42) -> List[ComputeNode]:
        """Create 100-node heterogeneous system"""
        np.random.seed(seed)
        nodes = []
        
        # High-capacity tier (20%)
        num_high = int(num_nodes * 0.20)
        for i in range(num_high):
            cpu = np.random.uniform(24, 32)
            mem = np.random.uniform(96, 128)
            idle_power = np.random.uniform(150, 200)
            dyn_power = np.random.uniform(200, 300)
            
            nodes.append(ComputeNode(
                id=i,
                cpu_capacity=cpu,
                memory_capacity=mem,
                idle_power=idle_power,
                dynamic_power=dyn_power
            ))
        
        # Medium-capacity tier (50%)
        num_medium = int(num_nodes * 0.50)
        for i in range(num_high, num_high + num_medium):
            cpu = np.random.uniform(8, 16)
            mem = np.random.uniform(32, 64)
            idle_power = np.random.uniform(50, 80)
            dyn_power = np.random.uniform(60, 120)
            
            nodes.append(ComputeNode(
                id=i,
                cpu_capacity=cpu,
                memory_capacity=mem,
                idle_power=idle_power,
                dynamic_power=dyn_power
            ))
        
        # Low-capacity tier (30%)
        for i in range(num_high + num_medium, num_nodes):
            cpu = np.random.uniform(2, 8)
            mem = np.random.uniform(8, 32)
            idle_power = np.random.uniform(15, 30)
            dyn_power = np.random.uniform(20, 50)
            
            nodes.append(ComputeNode(
                id=i,
                cpu_capacity=cpu,
                memory_capacity=mem,
                idle_power=idle_power,
                dynamic_power=dyn_power
            ))
        
        return nodes

# ============================================================================
# DISCRETE-EVENT SIMULATOR
# ============================================================================

class Simulator:
    """Discrete-event simulator for task scheduling"""
    
    def __init__(self, nodes: List[ComputeNode], scheduler, time_step=5.0):
        self.nodes = nodes
        self.scheduler = scheduler
        self.time_step = time_step
        self.current_time = 0.0
        self.total_energy = 0.0
        
    def run_episode(self, tasks: List[Task]) -> Dict:
        """Run one simulation episode"""
        
        # Reset nodes
        for node in self.nodes:
            node.current_cpu_usage = 0.0
            node.current_memory_usage = 0.0
            node.task_queue = []
        
        self.current_time = 0.0
        self.total_energy = 0.0
        
        pending_tasks = sorted(tasks, key=lambda t: t.arrival_time)
        running_tasks = []
        completed_tasks = []
        failed_tasks = []
        
        max_time = max([t.arrival_time for t in tasks]) + 1000.0
        
        # Simulation loop
        while self.current_time < max_time:
            
            # Arrive new tasks
            new_arrivals = []
            while pending_tasks and pending_tasks[0].arrival_time <= self.current_time:
                new_arrivals.append(pending_tasks.pop(0))
            
            # Schedule new tasks
            for task in new_arrivals:
                if hasattr(self.scheduler, 'schedule_task'):
                    node_id = self.scheduler.schedule_task(
                        task, self.nodes, 
                        all_pending=pending_tasks,
                        current_time=self.current_time
                    )
                else:
                    node_id = -1
                
                if node_id >= 0:
                    task.assigned_node = node_id
                    task.start_time = self.current_time
                    task.finish_time = self.current_time + task.execution_time
                    
                    self.nodes[node_id].current_cpu_usage += task.cpu_requirement
                    self.nodes[node_id].current_memory_usage += task.memory_requirement
                    self.nodes[node_id].task_queue.append(task)
                    running_tasks.append(task)
                else:
                    failed_tasks.append(task)
            
            # Complete finished tasks
            finished = []
            for task in running_tasks:
                if task.finish_time <= self.current_time:
                    finished.append(task)
                    node = self.nodes[task.assigned_node]
                    node.current_cpu_usage -= task.cpu_requirement
                    node.current_memory_usage -= task.memory_requirement
                    if task in node.task_queue:
                        node.task_queue.remove(task)
                    completed_tasks.append(task)
            
            for task in finished:
                running_tasks.remove(task)
            
            # Compute energy consumption
            for node in self.nodes:
                util = node.get_utilization()
                power = node.idle_power + node.dynamic_power * util
                energy = power * self.time_step / 3600.0  # Convert to kWh
                self.total_energy += energy
            
            # Advance time
            self.current_time += self.time_step
        
        # Compute metrics
        metrics = self.compute_metrics(completed_tasks, failed_tasks)
        
        return metrics
    
    def compute_metrics(self, completed: List[Task], failed: List[Task]) -> Dict:
        """Compute performance metrics"""
        
        if not completed:
            return {
                'avg_completion_time': float('inf'),
                'total_energy': self.total_energy,
                'sla_satisfaction': 0.0,
                'tasks_completed': 0,
                'tasks_failed': len(failed),
                'load_variance': 0.0
            }
        
        # Average completion time
        completion_times = [t.finish_time - t.arrival_time for t in completed]
        avg_completion = np.mean(completion_times)
        
        # SLA satisfaction
        sla_met = sum(1 for t in completed if t.finish_time <= t.deadline)
        sla_rate = sla_met / len(completed) * 100.0
        
        # Load variance
        utilizations = [n.get_utilization() for n in self.nodes]
        load_var = np.var(utilizations)
        
        return {
            'avg_completion_time': avg_completion,
            'total_energy': self.total_energy,
            'sla_satisfaction': sla_rate,
            'tasks_completed': len(completed),
            'tasks_failed': len(failed),
            'load_variance': load_var,
            'completion_times': completion_times
        }

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(scheduler_name='MARL', num_episodes=30, num_tasks=1000):
    """Run complete experiment"""
    
    print(f"\nRunning {scheduler_name} Scheduler")
    print("=" * 60)
    
    results = []
    
    for episode in range(num_episodes):
        # Create infrastructure
        nodes = InfrastructureGenerator.create_nodes(num_nodes=100, seed=episode)
        
        # Create scheduler
        if scheduler_name == 'MARL':
            scheduler = MARLScheduler(num_nodes=100)
        elif scheduler_name == 'Random':
            scheduler = RandomScheduler()
        elif scheduler_name == 'WeightedRR':
            scheduler = WeightedRoundRobinScheduler()
        elif scheduler_name == 'PriorityMinMin':
            scheduler = PriorityMinMinScheduler()
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        # Generate workload
        workload_gen = WorkloadGenerator(seed=episode * 100)
        tasks = workload_gen.generate_tasks(num_tasks=num_tasks)
        
        # Run simulation
        sim = Simulator(nodes, scheduler)
        metrics = sim.run_episode(tasks)
        
        results.append(metrics)
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Completion={metrics['avg_completion_time']:.1f}s, "
                  f"Energy={metrics['total_energy']:.1f}kWh, "
                  f"SLA={metrics['sla_satisfaction']:.1f}%")
    
    # Aggregate results (last 10 episodes for convergence)
    final_results = results[-10:]
    
    avg_metrics = {
        'avg_completion_time': np.mean([r['avg_completion_time'] for r in final_results]),
        'std_completion_time': np.std([r['avg_completion_time'] for r in final_results]),
        'avg_energy': np.mean([r['total_energy'] for r in final_results]),
        'std_energy': np.std([r['total_energy'] for r in final_results]),
        'avg_sla': np.mean([r['sla_satisfaction'] for r in final_results]),
        'std_sla': np.std([r['sla_satisfaction'] for r in final_results]),
        'avg_completed': np.mean([r['tasks_completed'] for r in final_results]),
        'avg_load_var': np.mean([r['load_variance'] for r in final_results]),
    }
    
    print(f"\nFinal Results (Episodes 21-30):")
    print(f"  Completion Time: {avg_metrics['avg_completion_time']:.1f} ± {avg_metrics['std_completion_time']:.1f} s")
    print(f"  Energy: {avg_metrics['avg_energy']:.1f} ± {avg_metrics['std_energy']:.1f} kWh")
    print(f"  SLA Satisfaction: {avg_metrics['avg_sla']:.1f} ± {avg_metrics['std_sla']:.1f} %")
    print(f"  Tasks Completed: {avg_metrics['avg_completed']:.0f}")
    print(f"  Load Variance: {avg_metrics['avg_load_var']:.3f}")
    
    return avg_metrics, results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("MARL Distributed Task Scheduling - Simulation")
    print("=" * 60)
    
    # Run all experiments
    schedulers = ['Random', 'WeightedRR', 'PriorityMinMin', 'MARL']
    all_results = {}
    
    for scheduler in schedulers:
        metrics, episode_results = run_experiment(
            scheduler_name=scheduler,
            num_episodes=30,
            num_tasks=1000
        )
        all_results[scheduler] = metrics
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Scheduler':<15} {'Completion(s)':<15} {'Energy(kWh)':<15} {'SLA(%)':<10}")
    print("-" * 60)
    
    for name, metrics in all_results.items():
        print(f"{name:<15} "
              f"{metrics['avg_completion_time']:>6.1f} ± {metrics['std_completion_time']:<5.1f} "
              f"{metrics['avg_energy']:>7.1f} ± {metrics['std_energy']:<5.0f} "
              f"{metrics['avg_sla']:>5.1f} ± {metrics['std_sla']:<4.1f}")
    
    print("\nResults saved to results.json")
