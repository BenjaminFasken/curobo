"""
Multi-Robot Path Planning using cuRobo for 12 Robots in 24D Configuration Space
================================================================================

Planning collision-free paths for N robots (12) on a continuous 2D plane.

    Configuration Space: High-dimensional space (2N dimensions = 24D for 12 robots).
    Each robot has (x, y) position, so full state is [x1,y1, x2,y2, ..., x12,y12].
    
    Constraints: No two robots may overlap (minimum separation distance).

Insights & Geometry:

    The obstacle space is highly structured and symmetric.

    The global constraint is the union of identical pairwise constraints. 
    The forbidden region between any two robots is geometrically identical.

Symmetry Exploitation Strategy:
------------------------------
1. Transform to relative coordinates where constraints are separable
2. Use GPU-accelerated pairwise distance computation
3. Apply smooth barrier cost functions for collision avoidance
4. Optimize using cuRobo's trajectory optimization framework

This implementation leverages CUDA for parallel computation of:
- All N(N-1)/2 = 66 pairwise distances for 12 robots
- Gradient computation for optimization
- Trajectory smoothness costs

TODO:
- Make sure robots stay within defined acceleration and velocity limits (currently not enforced)
- Allow robots to arrive at different times (Currently assumes all robots start and end at the same time)
- Add support for holonomic robots (currently assumes circular robots with simple translation)
- Add support for heterogeneous robot sizes (currently assumes identical robots)
- Add support for dynamic obstacles (currently only static obstacles are supported)
- Implement more advanced sampling strategies that leverage the structure of the problem (currently uses simple heuristics)
- Add visualization tools to better understand the behavior of the planner and the structure of the solution space (currently no visualization) 
- Implement more advanced path optimization techniques that can further improve the quality of the solution (currently uses a simple elastic band method)

"""

# Third Party
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from typing import Tuple, Optional, List
from dataclasses import dataclass
import time

# CuRobo
from curobo.types.base import TensorDeviceType


@dataclass
class MultiRobotPlannerConfig:
    """Configuration for multi-robot path planning."""
    
    # Number of robots
    n_robots: int = 12
    
    # Robot radius for collision checking
    robot_radius: float = 0.15
    
    # Minimum safe distance between robot centers (2 * radius + buffer)
    min_separation: float = 0.35
    
    # Workspace bounds [x_min, x_max, y_min, y_max]
    workspace_bounds: Tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0)
    
    # Trajectory optimization parameters
    n_timesteps: int = 64  # Number of timesteps in trajectory
    dt: float = 0.1  # Time step
    
    # Optimization parameters
    n_iterations: int = 500
    learning_rate: float = 0.1
    
    # Cost weights
    collision_weight: float = 100.0
    smoothness_weight: float = 1.0
    goal_weight: float = 10.0
    velocity_weight: float = 0.5
    acceleration_weight: float = 0.5
    
    # Collision cost parameters
    collision_activation_distance: float = 0.5  # Distance at which collision cost activates
    
    # Device
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float32


class MultiRobotPathPlanner:
    """
    GPU-accelerated multi-robot path planner for N robots in 2N-D configuration space.
    
    Uses cuRobo-style trajectory optimization with:
    - Pairwise collision avoidance between all robots
    - Workspace boundary constraints  
    - Smooth trajectory generation with velocity/acceleration limits
    """
    
    def __init__(self, config: MultiRobotPlannerConfig):
        self.config = config
        self.tensor_args = TensorDeviceType(device=config.device, dtype=config.dtype)
        
        # Pre-compute pairwise indices for collision checking
        self._setup_pairwise_indices()
        
        # Setup optimization buffers
        self._setup_buffers()
        
    def _setup_pairwise_indices(self):
        """Pre-compute indices for all robot pairs (N choose 2)."""
        n = self.config.n_robots
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        
        self.n_pairs = len(pairs)  # N*(N-1)/2 = 66 for N=12
        
        # Convert to tensors for efficient indexing
        self.pair_idx_i = torch.tensor(
            [p[0] for p in pairs], 
            device=self.config.device, 
            dtype=torch.long
        )
        self.pair_idx_j = torch.tensor(
            [p[1] for p in pairs], 
            device=self.config.device, 
            dtype=torch.long
        )
        
        print(f"Multi-Robot Planner initialized:")
        print(f"  - {self.config.n_robots} robots in {2*self.config.n_robots}D C-space")
        print(f"  - {self.n_pairs} pairwise collision constraints")
        
    def _setup_buffers(self):
        """Setup reusable GPU buffers for optimization."""
        n_t = self.config.n_timesteps
        n_r = self.config.n_robots
        
        # Trajectory buffer: [timesteps, n_robots, 2] for (x, y) positions
        self.traj_buffer = torch.zeros(
            (n_t, n_r, 2),
            device=self.config.device,
            dtype=self.config.dtype
        )
        
    def compute_pairwise_distances(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between all robots efficiently on GPU.
        
        Args:
            positions: Robot positions [..., n_robots, 2]
            
        Returns:
            distances: Pairwise distances [..., n_pairs]
        """
        # Extract positions for each pair
        pos_i = positions[..., self.pair_idx_i, :]  # [..., n_pairs, 2]
        pos_j = positions[..., self.pair_idx_j, :]  # [..., n_pairs, 2]
        
        # Compute Euclidean distances
        diff = pos_i - pos_j
        distances = torch.norm(diff, dim=-1)  # [..., n_pairs]
        
        return distances
    
    def collision_cost(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth collision cost using barrier function.
        
        Uses a smooth barrier that:
        - Is zero when distance > activation_distance
        - Increases smoothly as distance decreases
        - Becomes very large at collision
        
        Args:
            positions: Robot positions [batch, timesteps, n_robots, 2]
            
        Returns:
            cost: Collision cost [batch, timesteps]
        """
        distances = self.compute_pairwise_distances(positions)  # [..., n_pairs]
        
        # Signed distance (positive = safe, negative = collision)
        signed_dist = distances - self.config.min_separation
        
        # Smooth barrier cost (similar to cuRobo's collision cost)
        # Cost = weight * sum(relu(activation_dist - signed_dist)^2)
        activation = self.config.collision_activation_distance
        
        # Quadratic barrier when too close
        penetration = torch.clamp(activation - signed_dist, min=0)
        cost = self.config.collision_weight * torch.sum(penetration ** 2, dim=-1)
        
        return cost
    
    def workspace_cost(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Cost for staying within workspace bounds.
        
        Args:
            positions: Robot positions [..., n_robots, 2]
            
        Returns:
            cost: Boundary violation cost [...]
        """
        x_min, x_max, y_min, y_max = self.config.workspace_bounds
        
        x = positions[..., 0]
        y = positions[..., 1]
        
        # Boundary violations
        x_low_violation = torch.clamp(x_min - x + self.config.robot_radius, min=0)
        x_high_violation = torch.clamp(x - x_max + self.config.robot_radius, min=0)
        y_low_violation = torch.clamp(y_min - y + self.config.robot_radius, min=0)
        y_high_violation = torch.clamp(y - y_max + self.config.robot_radius, min=0)
        
        cost = 100.0 * (
            torch.sum(x_low_violation ** 2, dim=-1) +
            torch.sum(x_high_violation ** 2, dim=-1) +
            torch.sum(y_low_violation ** 2, dim=-1) +
            torch.sum(y_high_violation ** 2, dim=-1)
        )
        
        return cost
    
    def smoothness_cost(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory smoothness cost (velocity and acceleration).
        
        Args:
            trajectory: Full trajectory [timesteps, n_robots, 2]
            
        Returns:
            cost: Smoothness cost scalar
        """
        dt = self.config.dt
        
        # Velocity: first derivative
        velocity = (trajectory[1:] - trajectory[:-1]) / dt
        
        # Acceleration: second derivative  
        acceleration = (velocity[1:] - velocity[:-1]) / dt
        
        # L2 norm costs
        vel_cost = self.config.velocity_weight * torch.sum(velocity ** 2)
        acc_cost = self.config.acceleration_weight * torch.sum(acceleration ** 2)
        
        return vel_cost + acc_cost
    
    def goal_cost(
        self, 
        trajectory: torch.Tensor, 
        goal_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Cost for reaching goal positions at final timestep.
        
        Args:
            trajectory: [timesteps, n_robots, 2]
            goal_positions: [n_robots, 2]
            
        Returns:
            cost: Goal reaching cost scalar
        """
        final_positions = trajectory[-1]
        diff = final_positions - goal_positions
        cost = self.config.goal_weight * torch.sum(diff ** 2)
        return cost
    
    def start_cost(
        self,
        trajectory: torch.Tensor,
        start_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Cost for maintaining start positions at first timestep.
        
        Args:
            trajectory: [timesteps, n_robots, 2]
            start_positions: [n_robots, 2]
            
        Returns:
            cost: Start constraint cost scalar
        """
        initial_positions = trajectory[0]
        diff = initial_positions - start_positions
        cost = 1000.0 * torch.sum(diff ** 2)  # Strong constraint
        return cost
    
    def total_cost(
        self,
        trajectory: torch.Tensor,
        start_positions: torch.Tensor,
        goal_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total trajectory cost.
        
        Args:
            trajectory: [timesteps, n_robots, 2]
            start_positions: [n_robots, 2]
            goal_positions: [n_robots, 2]
            
        Returns:
            Total cost scalar
        """
        # Collision cost at each timestep
        coll_cost = torch.sum(self.collision_cost(trajectory))
        
        # Workspace boundary cost
        ws_cost = torch.sum(self.workspace_cost(trajectory))
        
        # Smoothness cost
        smooth_cost = self.smoothness_cost(trajectory)
        
        # Goal and start costs
        goal_c = self.goal_cost(trajectory, goal_positions)
        start_c = self.start_cost(trajectory, start_positions)
        
        # Terminal velocity cost - encourage robots to stop at goal
        if trajectory.shape[0] > 1:
            final_vel = (trajectory[-1] - trajectory[-2]) / self.config.dt
            terminal_vel_cost = 50.0 * torch.sum(final_vel ** 2)
        else:
            terminal_vel_cost = 0.0
        
        total = coll_cost + ws_cost + smooth_cost + goal_c + start_c + terminal_vel_cost
        return total
    
    def plan(
        self,
        start_positions: torch.Tensor,
        goal_positions: torch.Tensor,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Plan collision-free trajectories for all robots.
        
        Args:
            start_positions: Initial positions [n_robots, 2]
            goal_positions: Goal positions [n_robots, 2]
            verbose: Print optimization progress
            
        Returns:
            trajectory: Optimized trajectory [timesteps, n_robots, 2]
            info: Dictionary with optimization info
        """
        n_t = self.config.n_timesteps
        n_r = self.config.n_robots
        
        # Initialize trajectory with linear interpolation
        trajectory = torch.zeros(
            (n_t, n_r, 2),
            device=self.config.device,
            dtype=self.config.dtype,
            requires_grad=True
        )
        
        with torch.no_grad():
            for t in range(n_t):
                alpha = t / (n_t - 1)
                trajectory.data[t] = (1 - alpha) * start_positions + alpha * goal_positions
        
        # Optimizer
        optimizer = torch.optim.Adam([trajectory], lr=self.config.learning_rate)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config.n_iterations,
            eta_min=0.01
        )
        
        # Optimization loop
        history = {
            'total_cost': [],
            'collision_cost': [],
            'goal_cost': [],
            'time': []
        }
        
        start_time = time.time()
        
        for iteration in range(self.config.n_iterations):
            optimizer.zero_grad()
            
            # Compute total cost
            cost = self.total_cost(trajectory, start_positions, goal_positions)
            
            # Backward pass
            cost.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([trajectory], max_norm=10.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            # Record history
            with torch.no_grad():
                coll_cost = torch.sum(self.collision_cost(trajectory)).item()
                goal_c = self.goal_cost(trajectory, goal_positions).item()
                
                history['total_cost'].append(cost.item())
                history['collision_cost'].append(coll_cost)
                history['goal_cost'].append(goal_c)
                history['time'].append(time.time() - start_time)
                
                if verbose and (iteration % 50 == 0 or iteration == self.config.n_iterations - 1):
                    print(f"Iter {iteration:4d}: Total={cost.item():.2f}, "
                          f"Collision={coll_cost:.2f}, Goal={goal_c:.2f}")
        
        # Check for collisions in final trajectory
        with torch.no_grad():
            min_distances = self.compute_pairwise_distances(trajectory)
            min_dist = min_distances.min().item()
            has_collision = min_dist < self.config.min_separation
        
        info = {
            'history': history,
            'final_cost': cost.item(),
            'min_pairwise_distance': min_dist,
            'has_collision': has_collision,
            'planning_time': time.time() - start_time
        }
        
        if verbose:
            print(f"\nPlanning complete in {info['planning_time']:.2f}s")
            print(f"Min pairwise distance: {min_dist:.3f} (required: {self.config.min_separation:.3f})")
            print(f"Collision-free: {not has_collision}")
        
        return trajectory.detach(), info
    
    def check_collision(self, positions: torch.Tensor) -> bool:
        """Check if any collision exists at given positions."""
        distances = self.compute_pairwise_distances(positions)
        return (distances < self.config.min_separation).any().item()


def visualize_trajectories(
    planner: MultiRobotPathPlanner,
    trajectory: torch.Tensor,
    start_positions: torch.Tensor,
    goal_positions: torch.Tensor,
    save_animation: bool = False
):
    """
    Visualize the planned trajectories.
    
    Args:
        planner: The planner instance
        trajectory: Optimized trajectory [timesteps, n_robots, 2]
        start_positions: [n_robots, 2]
        goal_positions: [n_robots, 2]
        save_animation: Whether to save as GIF
    """
    trajectory_np = trajectory.cpu().numpy()
    start_np = start_positions.cpu().numpy()
    goal_np = goal_positions.cpu().numpy()
    
    n_timesteps, n_robots, _ = trajectory_np.shape
    config = planner.config
    
    # Generate distinct colors for each robot
    colors = plt.cm.tab20(np.linspace(0, 1, n_robots))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot settings
    x_min, x_max, y_min, y_max = config.workspace_bounds
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Multi-Robot Path Planning: {n_robots} Robots in {2*n_robots}D C-Space')
    
    # Plot workspace boundary
    ax.axhline(y=y_min, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=y_max, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=x_min, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=x_max, color='k', linestyle='--', alpha=0.5)
    
    # Plot start and goal positions
    for i in range(n_robots):
        # Start position (circle)
        ax.plot(start_np[i, 0], start_np[i, 1], 'o', color=colors[i], 
                markersize=12, markeredgecolor='black', markeredgewidth=2)
        # Goal position (star)
        ax.plot(goal_np[i, 0], goal_np[i, 1], '*', color=colors[i], 
                markersize=15, markeredgecolor='black', markeredgewidth=1)
    
    # Plot full trajectories
    for i in range(n_robots):
        ax.plot(trajectory_np[:, i, 0], trajectory_np[:, i, 1], 
                '-', color=colors[i], alpha=0.5, linewidth=2,
                label=f'Robot {i+1}')
    
    ax.legend(loc='upper left', ncol=3, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('multi_robot_trajectories.png', dpi=150)
    print("Saved trajectory plot to 'multi_robot_trajectories.png'")
    
    # Animation
    if save_animation:
        fig2, ax2 = plt.subplots(figsize=(12, 12))
        
        def init():
            ax2.clear()
            ax2.set_xlim(x_min - 0.5, x_max + 0.5)
            ax2.set_ylim(y_min - 0.5, y_max + 0.5)
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            return []
        
        def animate(frame):
            ax2.clear()
            ax2.set_xlim(x_min - 0.5, x_max + 0.5)
            ax2.set_ylim(y_min - 0.5, y_max + 0.5)
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            ax2.set_title(f'Multi-Robot Motion - Frame {frame}/{n_timesteps}')
            
            # Plot trajectories up to current frame
            for i in range(n_robots):
                ax2.plot(trajectory_np[:frame+1, i, 0], trajectory_np[:frame+1, i, 1], 
                        '-', color=colors[i], alpha=0.3, linewidth=1)
            
            # Plot current robot positions as circles
            for i in range(n_robots):
                circle = Circle(
                    (trajectory_np[frame, i, 0], trajectory_np[frame, i, 1]),
                    config.robot_radius,
                    color=colors[i],
                    alpha=0.7
                )
                ax2.add_patch(circle)
                ax2.plot(trajectory_np[frame, i, 0], trajectory_np[frame, i, 1],
                        'k.', markersize=3)
            
            # Plot goals
            for i in range(n_robots):
                ax2.plot(goal_np[i, 0], goal_np[i, 1], '*', color=colors[i], 
                        markersize=10, markeredgecolor='black', markeredgewidth=0.5)
            
            return []
        
        anim = FuncAnimation(fig2, animate, init_func=init,
                            frames=n_timesteps, interval=50, blit=True)
        anim.save('multi_robot_animation.gif', writer='pillow', fps=20)
        print("Saved animation to 'multi_robot_animation.gif'")
    
    plt.show()


def plot_optimization_history(info: dict):
    """Plot the optimization cost history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    history = info['history']
    
    axes[0].plot(history['total_cost'])
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Total Cost')
    axes[0].set_title('Total Cost')
    axes[0].set_yscale('log')
    axes[0].grid(True)
    
    axes[1].plot(history['collision_cost'])
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Collision Cost')
    axes[1].set_title('Collision Cost')
    axes[1].grid(True)
    
    axes[2].plot(history['goal_cost'])
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Goal Cost')
    axes[2].set_title('Goal Cost')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('optimization_history.png', dpi=150)
    print("Saved optimization history to 'optimization_history.png'")
    plt.show()


def generate_random_configuration(
    n_robots: int,
    bounds: Tuple[float, float, float, float],
    min_separation: float,
    device: str,
    dtype: torch.dtype,
    max_attempts: int = 1000
) -> torch.Tensor:
    """
    Generate a random collision-free configuration for N robots.
    
    Uses rejection sampling to ensure all robots are separated.
    """
    x_min, x_max, y_min, y_max = bounds
    positions = torch.zeros((n_robots, 2), device=device, dtype=dtype)
    
    for i in range(n_robots):
        for attempt in range(max_attempts):
            # Random position within bounds (with margin)
            margin = min_separation / 2
            x = torch.rand(1, device=device, dtype=dtype) * (x_max - x_min - 2*margin) + x_min + margin
            y = torch.rand(1, device=device, dtype=dtype) * (y_max - y_min - 2*margin) + y_min + margin
            
            candidate = torch.tensor([[x.item(), y.item()]], device=device, dtype=dtype)
            
            # Check distance to all previously placed robots
            if i == 0:
                positions[i] = candidate
                break
            else:
                dists = torch.norm(positions[:i] - candidate, dim=1)
                if dists.min() >= min_separation:
                    positions[i] = candidate
                    break
        else:
            raise RuntimeError(f"Could not place robot {i} after {max_attempts} attempts")
    
    return positions


def generate_circle_configuration(
    n_robots: int,
    center: Tuple[float, float],
    radius: float,
    device: str,
    dtype: torch.dtype,
    rotation_offset: float = 0.0
) -> torch.Tensor:
    """
    Generate a circular configuration with N robots evenly distributed.
    
    Args:
        n_robots: Number of robots
        center: (x, y) center of circle
        radius: Radius of circle
        device: Torch device
        dtype: Torch dtype
        rotation_offset: Angle offset in radians (default 0)
        
    Returns:
        positions: [n_robots, 2] positions arranged in a circle
    """
    positions = torch.zeros((n_robots, 2), device=device, dtype=dtype)
    
    for i in range(n_robots):
        angle = 2 * np.pi * i / n_robots + rotation_offset
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        positions[i] = torch.tensor([x, y], device=device, dtype=dtype)
    
    return positions


def generate_circle_swap_configuration(
    n_robots: int,
    center: Tuple[float, float],
    radius: float,
    device: str,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate start and goal configurations where robots on a circle
    need to swap with their opposite partner.
    
    Each robot i needs to move to the position of robot (i + n_robots/2).
    This creates a challenging coordination problem.
    
    Args:
        n_robots: Number of robots (should be even for perfect opposite pairs)
        center: (x, y) center of circle
        radius: Radius of circle
        device: Torch device
        dtype: Torch dtype
        
    Returns:
        start_positions: Initial circle configuration
        goal_positions: Target positions (rotated by π)
    """
    # Generate initial circle
    start_positions = generate_circle_configuration(
        n_robots, center, radius, device, dtype, rotation_offset=0.0
    )
    
    # Goal is the same circle rotated by π (opposite positions)
    goal_positions = generate_circle_configuration(
        n_robots, center, radius, device, dtype, rotation_offset=np.pi
    )
    
    return start_positions, goal_positions


def demo_50_robots():
    """
    Demonstrate path planning for 50 robots in 100D configuration space.
    """
    print("=" * 70)
    print("Multi-Robot Path Planning Demo: 50 Robots in 100D Configuration Space")
    print("=" * 70)
    
    # Configuration - tuned for 50 robots with tight goal reaching
    config = MultiRobotPlannerConfig(
        n_robots=50,
        robot_radius=0.1,
        min_separation=0.4,  # 2*radius + small safety buffer (tighter)
        workspace_bounds=(-8.0, 8.0, -8.0, 8.0),  # Larger workspace for more robots
        n_timesteps=80,  # More timesteps for complex paths
        dt=0.1,
        n_iterations=1000,  # More iterations for convergence
        learning_rate=0.1,
        collision_weight=30.0,  # Reduced - less conservative avoidance
        smoothness_weight=0.3,
        goal_weight=500.0,  # Much higher - very tight goal reaching
        velocity_weight=0.1,
        acceleration_weight=0.1,
        collision_activation_distance=0.3,  # Very tight - only when nearly colliding
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\nUsing device: {config.device}")
    print(f"Configuration space dimension: {2 * config.n_robots}D")
    print(f"Number of pairwise constraints: {config.n_robots * (config.n_robots - 1) // 2}")
    
    # Create planner
    planner = MultiRobotPathPlanner(config)
    
    # Generate circle swap configuration
    print("\nGenerating circle swap configuration...")
    print("Robots arranged in a circle, each must swap with opposite robot")
    
    torch.manual_seed(123)  # For reproducibility
    
    # Calculate appropriate circle radius based on number of robots and separation
    # Arc length between adjacent robots should be at least min_separation
    # Arc length = 2 * π * r / n_robots
    # We want: 2 * π * r / n_robots >= min_separation * 1.5 (safety factor)
    required_radius = (config.min_separation * 1.5 * config.n_robots) / (2 * np.pi)
    circle_radius = max(required_radius, 3.0)  # At least 3.0 for visibility
    
    print(f"Circle radius: {circle_radius:.2f} (required: {required_radius:.2f})")
    
    start_positions, goal_positions = generate_circle_swap_configuration(
        config.n_robots,
        center=(0.0, 0.0),
        radius=circle_radius,
        device=config.device,
        dtype=config.dtype
    )
    
    print(f"Start configuration collision-free: {not planner.check_collision(start_positions)}")
    print(f"Goal configuration collision-free: {not planner.check_collision(goal_positions)}")
    
    # Verify that goal is indeed opposite positions
    center_dist = torch.norm(start_positions + goal_positions, dim=1).mean().item()
    print(f"Mean distance from start+goal to center: {center_dist:.4f} (should be near 0)")
    
    # Plan trajectories
    print("\n" + "-" * 50)
    print("Starting trajectory optimization...")
    print("-" * 50)
    
    trajectory, info = planner.plan(start_positions, goal_positions, verbose=True)
    
    # Print summary
    print("\n" + "=" * 50)
    print("PLANNING SUMMARY")
    print("=" * 50)
    print(f"Planning time: {info['planning_time']:.2f} seconds")
    print(f"Final cost: {info['final_cost']:.2f}")
    print(f"Minimum pairwise distance: {info['min_pairwise_distance']:.3f}")
    print(f"Required separation: {config.min_separation:.3f}")
    print(f"Collision-free trajectory: {not info['has_collision']}")
    print(f"Trajectory shape: {list(trajectory.shape)} (timesteps, robots, xy)")
    
    # Compute trajectory statistics
    velocities = (trajectory[1:] - trajectory[:-1]) / config.dt
    max_velocity = velocities.norm(dim=-1).max().item()
    print(f"Maximum velocity: {max_velocity:.3f} m/s")
    
    # Compute goal reaching accuracy
    final_positions = trajectory[-1]
    goal_errors = torch.norm(final_positions - goal_positions, dim=-1)
    print(f"Mean goal error: {goal_errors.mean().item():.4f}")
    print(f"Max goal error: {goal_errors.max().item():.4f}")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_trajectories(planner, trajectory, start_positions, goal_positions, 
                          save_animation=True)
    plot_optimization_history(info)
    
    return trajectory, info


def demo_scalability():
    """
    Test scalability with different numbers of robots.
    """
    print("\n" + "=" * 70)
    print("Scalability Test: Varying Number of Robots")
    print("=" * 70)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robot_counts = [4, 8, 12, 16, 20]
    results = []
    
    for n_robots in robot_counts:
        print(f"\nTesting with {n_robots} robots ({2*n_robots}D C-space)...")
        
        config = MultiRobotPlannerConfig(
            n_robots=n_robots,
            robot_radius=0.1,
            min_separation=0.25,
            workspace_bounds=(-5.0, 5.0, -5.0, 5.0),
            n_timesteps=48,
            n_iterations=200,
            device=device
        )
        
        planner = MultiRobotPathPlanner(config)
        
        try:
            torch.manual_seed(123)
            start = generate_random_configuration(
                n_robots, config.workspace_bounds, config.min_separation,
                config.device, config.dtype
            )
            goal = generate_random_configuration(
                n_robots, config.workspace_bounds, config.min_separation,
                config.device, config.dtype
            )
            
            _, info = planner.plan(start, goal, verbose=False)
            
            results.append({
                'n_robots': n_robots,
                'dim': 2 * n_robots,
                'n_pairs': n_robots * (n_robots - 1) // 2,
                'time': info['planning_time'],
                'success': not info['has_collision'],
                'min_dist': info['min_pairwise_distance']
            })
            
            print(f"  Time: {info['planning_time']:.2f}s, "
                  f"Success: {not info['has_collision']}, "
                  f"Min dist: {info['min_pairwise_distance']:.3f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({
                'n_robots': n_robots,
                'dim': 2 * n_robots,
                'n_pairs': n_robots * (n_robots - 1) // 2,
                'time': None,
                'success': False
            })
    
    # Print summary table
    print("\n" + "-" * 60)
    print(f"{'Robots':>8} {'C-Space':>10} {'Pairs':>8} {'Time (s)':>10} {'Success':>10}")
    print("-" * 60)
    for r in results:
        time_str = f"{r['time']:.2f}" if r['time'] else "N/A"
        print(f"{r['n_robots']:>8} {r['dim']:>10}D {r['n_pairs']:>8} {time_str:>10} {str(r['success']):>10}")
    
    return results


if __name__ == "__main__":
    # Run main demo with 50 robots
    trajectory, info = demo_50_robots()
    
    # Optionally run scalability test
    # results = demo_scalability()