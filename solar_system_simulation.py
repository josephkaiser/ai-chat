from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

G = 6.67430e-11  # m^3 kg^-1 s^-2


@dataclass
class CelestialBody:
    name: str
    mass: float
    position: np.ndarray
    velocity: np.ndarray


def compute_accelerations(masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Compute pairwise Newtonian accelerations for all bodies."""
    count = len(masses)
    accelerations = np.zeros_like(positions)

    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            r = positions[j] - positions[i]
            distance = np.linalg.norm(r)
            if distance == 0.0:
                continue
            accelerations[i] += G * masses[j] * r / distance**3

    return accelerations


def system_derivatives(_t: float, state: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Return derivatives for [positions, velocities]."""
    body_count = len(masses)
    positions = state[: 3 * body_count].reshape((body_count, 3))
    velocities = state[3 * body_count :].reshape((body_count, 3))
    accelerations = compute_accelerations(masses, positions)

    return np.concatenate([velocities.ravel(), accelerations.ravel()])


def initialize_solar_system() -> list[CelestialBody]:
    """Approximate starting state for the Sun and eight planets."""
    sun_mass = 1.98847e30

    sun = CelestialBody(
        name="Sun",
        mass=sun_mass,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
    )

    # name, mass [kg], semi-major axis [m], eccentricity
    planetary_data = [
        ("Mercury", 3.285e23, 5.791e10, 0.2056),
        ("Venus", 4.867e24, 1.082e11, 0.0067),
        ("Earth", 5.972e24, 1.496e11, 0.0167),
        ("Mars", 6.417e23, 2.279e11, 0.0934),
        ("Jupiter", 1.898e27, 7.785e11, 0.0484),
        ("Saturn", 5.683e26, 1.427e12, 0.0565),
        ("Uranus", 8.681e25, 2.871e12, 0.0457),
        ("Neptune", 1.024e26, 4.515e12, 0.0113),
    ]

    bodies = [sun]

    for name, mass, semi_major_axis, eccentricity in planetary_data:
        perihelion = semi_major_axis * (1.0 - eccentricity)
        orbital_speed = np.sqrt(G * sun_mass * (1.0 + eccentricity) / perihelion)

        bodies.append(
            CelestialBody(
                name=name,
                mass=mass,
                position=np.array([perihelion, 0.0, 0.0]),
                velocity=np.array([0.0, orbital_speed, 0.0]),
            )
        )

    return bodies


def simulate_solar_system(duration_days: float = 365.0, samples: int = 365):
    """Run the N-body simulation and return bodies, times, and solution states."""
    bodies = initialize_solar_system()
    masses = np.array([body.mass for body in bodies], dtype=float)

    initial_state = np.concatenate(
        [
            np.array([body.position for body in bodies]).ravel(),
            np.array([body.velocity for body in bodies]).ravel(),
        ]
    )

    t_span = (0.0, duration_days * 24 * 3600)
    t_eval = np.linspace(t_span[0], t_span[1], samples)

    solution = solve_ivp(
        fun=system_derivatives,
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        args=(masses,),
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )

    if not solution.success:
        raise RuntimeError(f"Simulation failed: {solution.message}")

    body_count = len(bodies)
    states = solution.y.T.reshape((len(solution.t), 2, body_count, 3))
    return bodies, solution.t, states


def plot_orbits(bodies: list[CelestialBody], states: np.ndarray) -> None:
    """Plot x/y orbits in astronomical units."""
    positions = states[:, 0, :, :]
    au = 1.495978707e11

    plt.figure(figsize=(12, 8))
    for index, body in enumerate(bodies):
        x = positions[:, index, 0] / au
        y = positions[:, index, 1] / au
        plt.plot(x, y, label=body.name)
        plt.scatter(x[-1], y[-1], s=20)

    plt.title("Solar System N-Body Simulation")
    plt.xlabel("X Position (AU)")
    plt.ylabel("Y Position (AU)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def main() -> None:
    print("Running solar system simulation...")
    bodies, times, states = simulate_solar_system(duration_days=365.0, samples=500)
    print(f"Computed {len(times)} timesteps for {len(bodies)} bodies.")
    plot_orbits(bodies, states)
    plt.show()


if __name__ == "__main__":
    main()
