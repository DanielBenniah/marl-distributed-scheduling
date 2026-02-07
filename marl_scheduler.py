#!/usr/bin/env python3
"""
Multi-Agent Reinforcement Learning for Distributed Task Scheduling

This is the complete implementation used in the paper:
"Decentralized Task Scheduling in Distributed Systems: A Deep Reinforcement Learning Approach"

Author: Daniel Benniah John
License: MIT
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Task:
    """Represents a computational task"""
    id: int
    arrival_time: float
    execution_time: float
    cpu_requirement: float
    memory_requirement: float
    priority: int  # 0=Production, 1=Batch, 2=BestEffort
    deadline: float
    assigned_node: int = -1
    start_time: float = -1
    finish_time: float = -1
    
@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system"""
    id: int
    cpu_capacity: float
    memory_capacity: float
    idle_power: float
    dynamic_power: float
    current_cpu_usage: float = 0.0
    current_memory_usage: float = 0.0
    task_queue: List[Task] = None
    
    def __post_init__(self):
        if self.task_queue is None:
            self.task_queue = []
    
    def can_accept(self, task: Task) -> bool:
        """Check if node can accept task"""
        return (self.current_cpu_usage + task.cpu_requirement <= self.cpu_capacity and
                self.current_memory_usage + task.memory_requirement <= self.memory_capacity)
    
    def get_utilization(self) -> float:
        """Get normalized CPU utilization"""
        return self.current_cpu_usage / self.cpu_capacity if self.cpu_capacity > 0 else 0.0

# ============================================================================
# NEURAL NETWORK (NumPy-only Actor-Critic)
# ============================================================================

class ActorCriticNetwork:
    """Lightweight actor-critic network using only NumPy"""
    
    def __init__(self, input_dim=50, hidden_dim=128, output_dim=100, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        
        # Initialize weights (He initialization)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        # Actor head (policy)
        self.W2_actor = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2_actor = np.zeros(output_dim)
        
        # Critic head (value)
        self.W2_critic = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2_critic = np.zeros(1)
        
        # For Adam optimizer
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        
        self.m_W2_actor = np.zeros_like(self.W2_actor)
        self.v_W2_actor = np.zeros_like(self.W2_actor)
        self.m_b2_actor = np.zeros_like(self.b2_actor)
        self.v_b2_actor = np.zeros_like(self.b2_actor)
        
        self.m_W2_critic = np.zeros_like(self.W2_critic)
        self.v_W2_critic = np.zeros_like(self.W2_critic)
        self.m_b2_critic = np.zeros_like(self.b2_critic)
        self.v_b2_critic = np.zeros_like(self.b2_critic)
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # timestep for Adam
    
    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, state):
        """Forward pass through network"""
        # Hidden layer
        self.z1 = np.dot(state, self.W1) + self.b1
        self.h1 = self.relu(self.z1)
        
        # Actor output (policy)
        self.z2_actor = np.dot(self.h1, self.W2_actor) + self.b2_actor
        policy = self.softmax(self.z2_actor)
        
        # Critic output (value)
        value = np.dot(self.h1, self.W2_critic) + self.b2_critic
        
        return policy, value[0]
    
    def get_action(self, state, valid_actions=None):
        """Sample action from policy"""
        policy, value = self.forward(state)
        
        # Mask invalid actions
        if valid_actions is not None:
            mask = np.zeros(self.output_dim)
            mask[valid_actions] = 1.0
            policy = policy * mask
            policy = policy / (np.sum(policy) + 1e-10)
        
        # Sample action
        action = np.random.choice(self.output_dim, p=policy)
        return action, policy[action], value
    
    def update(self, state, action, reward, next_state, done, gamma=0.99):
        """Update network using actor-critic"""
        self.t += 1
        
        # Forward pass
        policy, value = self.forward(state)
        _, next_value = self.forward(next_state)
        
        # Compute TD error and advantage
        if done:
            td_target = reward
        else:
            td_target = reward + gamma * next_value
        
        td_error = td_target - value
        advantage = td_error
        
        # === Critic update (gradient descent on TD error) ===
        grad_critic = -2 * td_error
        
        dW2_critic = np.outer(self.h1, grad_critic)
        db2_critic = np.array([grad_critic])
        
        # Adam update for critic
        self.m_W2_critic = self.beta1 * self.m_W2_critic + (1 - self.beta1) * dW2_critic
        self.v_W2_critic = self.beta2 * self.v_W2_critic + (1 - self.beta2) * (dW2_critic ** 2)
        m_hat = self.m_W2_critic / (1 - self.beta1 ** self.t)
        v_hat = self.v_W2_critic / (1 - self.beta2 ** self.t)
        self.W2_critic -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self.m_b2_critic = self.beta1 * self.m_b2_critic + (1 - self.beta1) * db2_critic
        self.v_b2_critic = self.beta2 * self.v_b2_critic + (1 - self.beta2) * (db2_critic ** 2)
        m_hat = self.m_b2_critic / (1 - self.beta1 ** self.t)
        v_hat = self.v_b2_critic / (1 - self.beta2 ** self.t)
        self.b2_critic -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # === Actor update (policy gradient) ===
        # Gradient of log policy
        grad_log_policy = np.zeros(self.output_dim)
        grad_log_policy[action] = 1.0 / (policy[action] + 1e-10)
        
        # Policy gradient theorem: ∇J = ∇log π * advantage
        grad_actor = -(grad_log_policy - policy) * advantage  # Negative for gradient ascent
        
        dW2_actor = np.outer(self.h1, grad_actor)
        db2_actor = grad_actor
        
        # Adam update for actor
        self.m_W2_actor = self.beta1 * self.m_W2_actor + (1 - self.beta1) * dW2_actor
        self.v_W2_actor = self.beta2 * self.v_W2_actor + (1 - self.beta2) * (dW2_actor ** 2)
        m_hat = self.m_W2_actor / (1 - self.beta1 ** self.t)
        v_hat = self.v_W2_actor / (1 - self.beta2 ** self.t)
        self.W2_actor -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self.m_b2_actor = self.beta1 * self.m_b2_actor + (1 - self.beta1) * db2_actor
        self.v_b2_actor = self.beta2 * self.v_b2_actor + (1 - self.beta2) * (db2_actor ** 2)
        m_hat = self.m_b2_actor / (1 - self.beta1 ** self.t)
        v_hat = self.v_b2_actor / (1 - self.beta2 ** self.t)
        self.b2_actor -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Backprop to hidden layer (simplified - only actor path for now)
        dh1 = np.dot(grad_actor, self.W2_actor.T)
        dz1 = dh1 * self.relu_derivative(self.z1)
        
        dW1 = np.outer(state, dz1)
        db1 = dz1
        
        # Adam update for shared layer
        self.m_W1 = self.beta1 * self.m_W1 + (1 - self.beta1) * dW1
        self.v_W1 = self.beta2 * self.v_W1 + (1 - self.beta2) * (dW1 ** 2)
        m_hat = self.m_W1 / (1 - self.beta1 ** self.t)
        v_hat = self.v_W1 / (1 - self.beta2 ** self.t)
        self.W1 -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * db1
        self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (db1 ** 2)
        m_hat = self.m_b1 / (1 - self.beta1 ** self.t)
        v_hat = self.v_b1 / (1 - self.beta2 ** self.t)
        self.b1 -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return td_error

# ============================================================================
# MARL SCHEDULER
# ============================================================================

class MARLScheduler:
    """Multi-Agent RL Scheduler"""
    
    def __init__(self, num_nodes=100):
        self.num_nodes = num_nodes
        self.agents = [ActorCriticNetwork(output_dim=num_nodes) for _ in range(num_nodes)]
        
    def get_state(self, node: ComputeNode, all_nodes: List[ComputeNode], 
                  pending_tasks: List[Task], current_time: float) -> np.ndarray:
        """Extract state features for a node"""
        features = []
        
        # Node's own state (5 features)
        features.append(node.get_utilization())
        features.append(node.current_memory_usage / node.memory_capacity)
        features.append(len(node.task_queue) / 100.0)  # normalized queue length
        features.append(node.cpu_capacity / 32.0)  # normalized capacity
        features.append(node.memory_capacity / 128.0)
        
        # Neighbor statistics (10 features - sample 10 neighbors)
        neighbor_indices = np.random.choice(self.num_nodes, min(10, self.num_nodes), replace=False)
        for i in neighbor_indices:
            if i < len(all_nodes):
                features.append(all_nodes[i].get_utilization())
        
        # Pad if needed
        while len(features) < 15:
            features.append(0.0)
        
        # Task queue statistics (5 features)
        if pending_tasks:
            avg_cpu = np.mean([t.cpu_requirement for t in pending_tasks[:10]])
            avg_mem = np.mean([t.memory_requirement for t in pending_tasks[:10]])
            avg_priority = np.mean([t.priority for t in pending_tasks[:10]])
            features.append(avg_cpu / 32.0)
            features.append(avg_mem / 128.0)
            features.append(avg_priority / 2.0)
            features.append(len(pending_tasks) / 1000.0)
            features.append(min(1.0, current_time / 1000.0))
        else:
            features.extend([0.0] * 5)
        
        # Global statistics (30 features)
        total_util = np.mean([n.get_utilization() for n in all_nodes])
        max_util = np.max([n.get_utilization() for n in all_nodes])
        min_util = np.min([n.get_utilization() for n in all_nodes])
        
        features.extend([total_util, max_util, min_util])
        
        # Pad to 50 features
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50])
    
    def schedule_task(self, task: Task, nodes: List[ComputeNode], 
                     all_pending: List[Task], current_time: float) -> int:
        """Schedule a task using MARL"""
        
        # Get valid nodes
        valid_nodes = [i for i, n in enumerate(nodes) if n.can_accept(task)]
        
        if not valid_nodes:
            return -1  # No valid node
        
        # Priority-aware task scoring
        urgency = (3 - task.priority) * 0.4
        deadline_factor = (task.deadline - current_time) / max(1.0, task.deadline - task.arrival_time)
        urgency += deadline_factor * 0.3
        
        # Select node using hybrid approach
        best_node = -1
        best_score = -np.inf
        
        for node_id in valid_nodes:
            node = nodes[node_id]
            
            # Get neural network policy
            state = self.get_state(node, nodes, all_pending, current_time)
            policy, _ = self.agents[node_id].forward(state)
            
            # Compute hybrid score
            nn_score = policy[node_id] * 0.25
            load_score = (1 - node.get_utilization()) * 0.30
            mem_score = (1 - node.current_memory_usage / node.memory_capacity) * 0.20
            compat_score = (1 - abs(task.cpu_requirement / node.cpu_capacity - 0.5)) * 0.15
            priority_score = (3 - task.priority) / 3.0 * 0.10
            
            total_score = nn_score + load_score + mem_score + compat_score + priority_score
            
            if total_score > best_score:
                best_score = total_score
                best_node = node_id
        
        return best_node
    
    def compute_reward(self, task: Task, node: ComputeNode, current_time: float,
                      total_energy: float, load_variance: float) -> float:
        """Compute reward for scheduling decision"""
        reward = 0.0
        
        # SLA reward
        if task.finish_time <= task.deadline:
            reward += 15 * (4 - task.priority)  # Higher reward for higher priority
        else:
            reward -= 20 * (4 - task.priority)  # Higher penalty for higher priority
        
        # Completion time reward
        completion_time = task.finish_time - task.arrival_time
        reward += max(0, 100 - 0.5 * completion_time)
        
        # Energy penalty
        reward -= 0.3 * total_energy
        
        # Load balance reward
        reward -= 200 * load_variance
        
        return reward

# ============================================================================
# BASELINE SCHEDULERS
# ============================================================================

class RandomScheduler:
    """Random baseline scheduler"""
    def schedule_task(self, task: Task, nodes: List[ComputeNode], **kwargs) -> int:
        valid = [i for i, n in enumerate(nodes) if n.can_accept(task)]
        return np.random.choice(valid) if valid else -1

class WeightedRoundRobinScheduler:
    """Weighted Round-Robin baseline"""
    def __init__(self):
        self.last_node = 0
    
    def schedule_task(self, task: Task, nodes: List[ComputeNode], **kwargs) -> int:
        for _ in range(len(nodes)):
            self.last_node = (self.last_node + 1) % len(nodes)
            if nodes[self.last_node].can_accept(task):
                return self.last_node
        return -1

class PriorityMinMinScheduler:
    """Priority-aware Min-Min scheduler"""
    def schedule_task(self, task: Task, nodes: List[ComputeNode], **kwargs) -> int:
        valid = [i for i, n in enumerate(nodes) if n.can_accept(task)]
        if not valid:
            return -1
        
        # Find node with minimum completion time
        best_node = -1
        min_completion = float('inf')
        
        for node_id in valid:
            node = nodes[node_id]
            completion = task.execution_time * (1 + node.get_utilization())
            if completion < min_completion:
                min_completion = completion
                best_node = node_id
        
        return best_node

if __name__ == "__main__":
    print("MARL Distributed Task Scheduler")
    print("=" * 50)
    print("This is the implementation code.")
    print("Run simulation.py to execute experiments.")
