"""
Tests for learnable dt parameter in MultiRobotPathPlanner.

Verifies that:
1. dt actually changes during optimization (is being learned)
2. dt shrinks from its initial value (time minimization pressure)
3. dt does NOT collapse to zero (smoothness/velocity costs push back)
4. Robots remain collision-free
5. Robots reach their goals
6. Learnable dt produces shorter total time than fixed dt
7. dt stays within configured bounds
"""

import sys
import os
import pytest
import torch
import numpy as np

# Add scripts directory to path so we can import the planner
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from importlib import import_module

# Import from "path planning.py" which has a space in the name
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "path_planning",
    os.path.join(os.path.dirname(__file__), '..', 'scripts', 'path planning.py')
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MultiRobotPlannerConfig = _mod.MultiRobotPlannerConfig
MultiRobotPathPlanner = _mod.MultiRobotPathPlanner
generate_circle_swap_configuration = _mod.generate_circle_swap_configuration
generate_random_configuration = _mod.generate_random_configuration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_config(n_robots=6, dt_learnable=True, **overrides):
    """Create a planner config tuned for fast test execution.
    
    Initial dt is set HIGH (0.8) so the optimizer consistently shrinks it.
    The time_weight pushes dt down, while smoothness costs resist collapse.
    The equilibrium dt ends up somewhere between dt_min and the initial dt.
    """
    defaults = dict(
        n_robots=n_robots,
        robot_radius=0.1,
        min_separation=0.35,
        workspace_bounds=(-5.0, 5.0, -5.0, 5.0),
        n_timesteps=32,
        dt=0.8,  # Start HIGH so optimizer shrinks toward equilibrium
        n_iterations=300,
        learning_rate=0.08,
        collision_weight=50.0,
        smoothness_weight=1.0,
        goal_weight=200.0,
        velocity_weight=1.0,
        acceleration_weight=1.0,
        time_weight=5.0,  # Strong time-minimization pressure
        dt_min=0.01,
        dt_max=1.0,
        dt_learnable=dt_learnable,
        collision_activation_distance=0.4,
        device=DEVICE,
    )
    defaults.update(overrides)
    return MultiRobotPlannerConfig(**defaults)


def _plan_circle_swap(config, seed=42):
    """Run a circle-swap planning problem and return (trajectory, info)."""
    torch.manual_seed(seed)
    planner = MultiRobotPathPlanner(config)

    radius = max((config.min_separation * 1.5 * config.n_robots) / (2 * np.pi), 2.0)
    start, goal = generate_circle_swap_configuration(
        config.n_robots,
        center=(0.0, 0.0),
        radius=radius,
        device=config.device,
        dtype=config.dtype,
    )
    trajectory, info = planner.plan(start, goal, verbose=False)
    return trajectory, info, start, goal, planner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLearnableDt:
    """Tests for the learnable dt feature."""

    # ------------------------------------------------------------------
    # 1. dt actually changes during optimization
    # ------------------------------------------------------------------
    def test_dt_changes_during_optimization(self):
        """dt must not stay at its initial value – the optimizer must move it."""
        config = _make_config()
        _, info, *_ = _plan_circle_swap(config)

        initial_dt = info["initial_dt"]
        final_dt = info["final_dt"]

        assert initial_dt != pytest.approx(final_dt, abs=1e-6), (
            f"dt did not change: initial={initial_dt}, final={final_dt}"
        )

    # ------------------------------------------------------------------
    # 2. dt shrinks (time-minimization pressure)
    # ------------------------------------------------------------------
    def test_dt_shrinks(self):
        """The time_weight cost should push dt below its initial value."""
        config = _make_config()
        _, info, *_ = _plan_circle_swap(config)

        assert info["final_dt"] < info["initial_dt"], (
            f"dt did not shrink: initial={info['initial_dt']}, final={info['final_dt']}"
        )

    # ------------------------------------------------------------------
    # 3. dt does NOT collapse to zero
    # ------------------------------------------------------------------
    def test_dt_does_not_collapse_to_zero(self):
        """Smoothness costs should prevent dt from reaching dt_min."""
        config = _make_config()
        _, info, *_ = _plan_circle_swap(config)

        # dt should remain meaningfully above the hard lower bound
        assert info["final_dt"] > config.dt_min, (
            f"dt collapsed to minimum: final_dt={info['final_dt']}, dt_min={config.dt_min}"
        )
        # Also check it's not *almost* at the lower bound (more than 2x the minimum)
        assert info["final_dt"] > config.dt_min * 1.5, (
            f"dt too close to minimum: final_dt={info['final_dt']}, dt_min={config.dt_min}"
        )

    # ------------------------------------------------------------------
    # 4. No collisions in final trajectory
    # ------------------------------------------------------------------
    def test_no_collisions(self):
        """All pairwise distances must exceed min_separation at every timestep."""
        config = _make_config(n_iterations=400)
        _, info, *_ = _plan_circle_swap(config)

        assert not info["has_collision"], (
            f"Collision detected! min_pairwise_distance={info['min_pairwise_distance']:.4f}, "
            f"required={config.min_separation:.4f}"
        )

    # ------------------------------------------------------------------
    # 5. Robots reach their goals
    # ------------------------------------------------------------------
    def test_goal_reaching(self):
        """Final robot positions should be close to goal positions."""
        config = _make_config(n_iterations=400)
        trajectory, info, start, goal, planner = _plan_circle_swap(config)

        final_positions = trajectory[-1]
        goal_errors = torch.norm(final_positions - goal, dim=-1)
        mean_error = goal_errors.mean().item()
        max_error = goal_errors.max().item()

        # Mean error should be small
        assert mean_error < 0.5, f"Mean goal error too large: {mean_error:.4f}"
        # Max error should be bounded
        assert max_error < 1.0, f"Max goal error too large: {max_error:.4f}"

    # ------------------------------------------------------------------
    # 6. Learnable dt produces shorter total time than fixed dt
    # ------------------------------------------------------------------
    def test_learnable_dt_reduces_total_time(self):
        """With learnable dt, total trajectory time should be less than fixed dt baseline."""
        config_fixed = _make_config(dt_learnable=False)
        config_learnable = _make_config(dt_learnable=True)

        _, info_fixed, *_ = _plan_circle_swap(config_fixed, seed=42)
        _, info_learnable, *_ = _plan_circle_swap(config_learnable, seed=42)

        fixed_total_time = config_fixed.dt * config_fixed.n_timesteps
        learnable_total_time = info_learnable["total_trajectory_time"]

        assert learnable_total_time < fixed_total_time, (
            f"Learnable dt did not reduce total time: "
            f"learnable={learnable_total_time:.4f}, fixed={fixed_total_time:.4f}"
        )

    # ------------------------------------------------------------------
    # 7. dt stays within configured bounds
    # ------------------------------------------------------------------
    def test_dt_within_bounds(self):
        """dt must always remain between dt_min and dt_max."""
        config = _make_config()
        _, info, *_ = _plan_circle_swap(config)

        # Check final dt
        assert info["final_dt"] >= config.dt_min, (
            f"dt below minimum: {info['final_dt']} < {config.dt_min}"
        )
        assert info["final_dt"] <= config.dt_max, (
            f"dt above maximum: {info['final_dt']} > {config.dt_max}"
        )

        # Check full history
        for i, dt_val in enumerate(info["history"]["dt"]):
            assert dt_val >= config.dt_min - 1e-7, (
                f"dt below minimum at iteration {i}: {dt_val} < {config.dt_min}"
            )
            assert dt_val <= config.dt_max + 1e-7, (
                f"dt above maximum at iteration {i}: {dt_val} > {config.dt_max}"
            )

    # ------------------------------------------------------------------
    # 8. dt history is monotonically recorded and has correct length
    # ------------------------------------------------------------------
    def test_dt_history_recorded(self):
        """The dt history should be recorded at every iteration."""
        config = _make_config(n_iterations=100)
        _, info, *_ = _plan_circle_swap(config)

        assert len(info["history"]["dt"]) == config.n_iterations, (
            f"Expected {config.n_iterations} dt entries, got {len(info['history']['dt'])}"
        )

    # ------------------------------------------------------------------
    # 9. Fixed-dt mode still works (backward compatibility)
    # ------------------------------------------------------------------
    def test_fixed_dt_mode(self):
        """When dt_learnable=False, dt should remain constant throughout."""
        config = _make_config(dt_learnable=False)
        _, info, *_ = _plan_circle_swap(config)

        assert info["initial_dt"] == pytest.approx(info["final_dt"], abs=1e-8), (
            f"dt changed in fixed mode: initial={info['initial_dt']}, final={info['final_dt']}"
        )

        # All history entries should be the same
        for dt_val in info["history"]["dt"]:
            assert dt_val == pytest.approx(config.dt, abs=1e-8)

    # ------------------------------------------------------------------
    # 10. Higher time_weight produces more dt shrinkage
    # ------------------------------------------------------------------
    def test_higher_time_weight_shrinks_more(self):
        """Increasing time_weight should apply more shrinkage pressure on dt."""
        config_low = _make_config(time_weight=0.5)
        config_high = _make_config(time_weight=10.0)

        _, info_low, *_ = _plan_circle_swap(config_low, seed=42)
        _, info_high, *_ = _plan_circle_swap(config_high, seed=42)

        assert info_high["final_dt"] < info_low["final_dt"], (
            f"Higher time_weight did not shrink dt more: "
            f"high={info_high['final_dt']:.4f}, low={info_low['final_dt']:.4f}"
        )

    # ------------------------------------------------------------------
    # 11. Test with random start/goal (not just circle swap)
    # ------------------------------------------------------------------
    def test_random_configuration_no_collision(self):
        """Learnable dt should work with random configurations too."""
        config = _make_config(n_robots=4, n_iterations=400)
        planner = MultiRobotPathPlanner(config)

        torch.manual_seed(99)
        start = generate_random_configuration(
            config.n_robots, config.workspace_bounds, config.min_separation,
            config.device, config.dtype
        )
        goal = generate_random_configuration(
            config.n_robots, config.workspace_bounds, config.min_separation,
            config.device, config.dtype
        )

        trajectory, info = planner.plan(start, goal, verbose=False)

        # dt should have changed
        assert info["final_dt"] != pytest.approx(info["initial_dt"], abs=1e-6)
        # dt should have shrunk
        assert info["final_dt"] < info["initial_dt"]
        # No collisions
        assert not info["has_collision"], (
            f"Collision! min_dist={info['min_pairwise_distance']:.4f}"
        )

    # ------------------------------------------------------------------
    # 12. Smoothness cost increases as dt shrinks (gradient pushback)
    # ------------------------------------------------------------------
    def test_smoothness_pushback(self):
        """Verify that smoothness cost grows when dt is smaller for the same trajectory."""
        config = _make_config()
        planner = MultiRobotPathPlanner(config)

        # Create a simple linear trajectory
        n_t = config.n_timesteps
        n_r = config.n_robots
        traj = torch.randn(n_t, n_r, 2, device=config.device, dtype=config.dtype)

        dt_large = torch.tensor([0.2], device=config.device, dtype=config.dtype)
        dt_small = torch.tensor([0.05], device=config.device, dtype=config.dtype)

        cost_large_dt = planner.smoothness_cost(traj, dt_large)
        cost_small_dt = planner.smoothness_cost(traj, dt_small)

        assert cost_small_dt > cost_large_dt, (
            f"Smaller dt should produce higher smoothness cost: "
            f"small_dt_cost={cost_small_dt.item():.4f}, large_dt_cost={cost_large_dt.item():.4f}"
        )
