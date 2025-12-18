import jax
import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt


"""
Finite Volume Solver for 2D Isothermal Euler Equations
Philip Mocz (2025) @PMocz
"""

# Simulation parameters
parser = argparse.ArgumentParser(
    description="Finite Volume Solver for 2D Isothermal Euler Equations"
)
parser.add_argument("--n", type=int, default=128, help="resolution")
parser.add_argument("--t", type=float, default=3.0, help="stopping time")
parser.add_argument("--nt", type=int, default=1000, help="number of time steps")
args = parser.parse_args()

n = args.n
t_stop = args.t
nt = args.nt

dt = t_stop / nt
box_size = 1.0
dx = box_size / n


# set double precision
# jax.config.update("jax_enable_x64", True)


def get_conserved(rho, vx, vy, vol):
    """
    Calculate the conserved variable from the primitive
    """
    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol

    return Mass, Momx, Momy


def get_primitive(Mass, Momx, Momy, vol):
    """
    Calculate the primitive variable from the conserved
    """
    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol

    return rho, vx, vy


def get_gradient(f, dx):
    """
    Calculate the gradients of a field
    """
    f_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)
    f_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dx)

    return f_dx, f_dy


def extrapolate_to_face(f, f_dx, f_dy, dx):
    """
    Extrapolate the field from face centers to faces using gradients
    """
    f_XL = f + 0.5 * f_dx * dx
    f_XR = f - 0.5 * f_dx * dx
    f_XR = jnp.roll(f_XR, -1, axis=0)

    f_YL = f + 0.5 * f_dy * dx
    f_YR = f - 0.5 * f_dy * dx
    f_YR = jnp.roll(f_YR, -1, axis=1)

    return f_XL, f_XR, f_YL, f_YR


def apply_fluxes(F, flux_F_X, flux_F_Y, dx, dt):
    """
    Apply fluxes to conserved variables
    """
    fac = dx * dt
    F += -fac * flux_F_X
    F += fac * jnp.roll(flux_F_X, 1, axis=0)
    F += -fac * flux_F_Y
    F += fac * jnp.roll(flux_F_Y, 1, axis=1)

    return F


def get_flux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R):
    """
    Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule
    """

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)

    P_star = rho_star

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star
    flux_Momy = momx_star * momy_star / rho_star

    # find wavespeeds
    C_L = 1 + jnp.abs(vx_L)
    C_R = 1 + jnp.abs(vx_R)
    C = jnp.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_Mass -= C * 0.5 * (rho_R - rho_L)
    flux_Momx -= C * 0.5 * (rho_R * vx_R - rho_L * vx_L)
    flux_Momy -= C * 0.5 * (rho_R * vy_R - rho_L * vy_L)

    return flux_Mass, flux_Momx, flux_Momy


def update_sim(i, state):
    """
    Take a simulation step
    """
    rho = state["rho"]
    vx = state["vx"]
    vy = state["vy"]

    # calculate gradients
    rho_dx, rho_dy = get_gradient(rho, dx)
    vx_dx, vx_dy = get_gradient(vx, dx)
    vy_dx, vy_dy = get_gradient(vy, dx)
    P_dx = rho_dx
    P_dy = rho_dy

    # extrapolate half-step in time
    rho_prime = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
    vx_prime = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1 / rho) * P_dx)
    vy_prime = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + (1 / rho) * P_dy)

    # extrapolate in space to face centers
    rho_XL, rho_XR, rho_YL, rho_YR = extrapolate_to_face(rho_prime, rho_dx, rho_dy, dx)
    vx_XL, vx_XR, vx_YL, vx_YR = extrapolate_to_face(vx_prime, vx_dx, vx_dy, dx)
    vy_XL, vy_XR, vy_YL, vy_YR = extrapolate_to_face(vy_prime, vy_dx, vy_dy, dx)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass_X, flux_Momx_X, flux_Momy_X = get_flux(
        rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR
    )
    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y = get_flux(
        rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR
    )

    # get conserved variables
    Mass, Momx, Momy = get_conserved(rho, vx, vy, dx**2)

    # update solution
    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)

    # get primitive variables
    rho, vx, vy = get_primitive(Mass, Momx, Momy, dx**2)

    state["rho"] = rho
    state["vx"] = vx
    state["vy"] = vy

    return state


def run_simulation(state):
    """
    Run the finite volume simulation (Main Loop)
    """
    state = jax.lax.fori_loop(0, nt, update_sim, init_val=state)

    return state


def set_up_state():
    """
    Set up the initial state for the simulation
    """

    # Mesh
    x_lin = jnp.linspace(0.5 * dx, box_size - 0.5 * dx, n)
    X, Y = jnp.meshgrid(x_lin, x_lin, indexing="ij")

    # Initial condition
    rho = jnp.ones((n, n))
    vx = 0.2 * jnp.sin(8.0 * jnp.pi * Y)
    vy = -0.2 * jnp.cos(4.0 * jnp.pi * X) * jnp.sin(2.0 * jnp.pi * Y)

    state = {}
    state["rho"] = rho
    state["vx"] = vx
    state["vy"] = vy

    return state


def plot_solution(state):
    """
    Plot the solution
    """
    plt.figure(figsize=(6, 4), dpi=240)

    plt.imshow(state["rho"].T, cmap="inferno")
    plt.clim(0.95, 1.1)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(
        f"output_isothermal_t{t_stop}_n{n}_nt{nt}.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=240,
    )
    plt.show()


def main():
    state = set_up_state()
    state = run_simulation(state)
    plot_solution(state)


if __name__ == "__main__":
    main()
