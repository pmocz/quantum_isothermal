import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Quantum

# Philip Mocz (2025)

# Simulate a compressible isothermal fluid with a quantum Schrodinger-type analog equation.

n = 128  # resolution

box_size = 1.0
kx_lin = 2.0 * jnp.pi / box_size * jnp.arange(-n / 2, n / 2)
ky_lin = 2.0 * jnp.pi / box_size * jnp.arange(-n / 2, n / 2)
kx, ky = jnp.meshgrid(kx_lin, ky_lin, indexing="ij")
kx = jnp.fft.ifftshift(kx)
ky = jnp.fft.ifftshift(ky)
k_sq = kx**2 + ky**2


def div(fx, fy):
    """
    Compute the divergence of a periodic vector field f = (f_x, f_y)
    spectral method
    """
    fx_k = jnp.fft.fftn(fx)
    fy_k = jnp.fft.fftn(fy)

    div_k = 1j * (kx * fx_k + ky * fy_k)
    div_f = jnp.fft.ifftn(div_k).real

    return div_f


def curl(fx, fy):
    """
    Compute the curl of a periodic vector field f = (f_x, f_y)
    spectral method
    """
    fx_k = jnp.fft.fftn(fx)
    fy_k = jnp.fft.fftn(fy)

    curl_k = 1j * (kx * fy_k - ky * fx_k)
    curl_f = jnp.fft.ifftn(curl_k).real

    return curl_f


def grad(f):
    """
    Compute the gradient of a periodic scalar field f
    spectral method
    """
    f_k = jnp.fft.fftn(f)

    grad_x_k = 1j * kx * f_k
    grad_y_k = 1j * ky * f_k

    grad_x = jnp.fft.ifftn(grad_x_k).real
    grad_y = jnp.fft.ifftn(grad_y_k).real

    return grad_x, grad_y


def run_simulation(state):
    # TODO

    return state


def set_up_state():
    """
    Set up the initial state for the simulation
    """
    # Simulation parameters
    dt = 0.003
    m_per_hbar = 1.0  # mass parameter in quantum analogy

    # Mesh
    dx = box_size / n
    x_lin = jnp.linspace(0.5 * dx, box_size - 0.5 * dx, n)
    X, Y = jnp.meshgrid(x_lin, x_lin, indexing="ij")

    # Initial condition
    rho = jnp.ones((n, n))
    vx = 0.2 * jnp.sin(8.0 * jnp.pi * Y)
    vy = -0.2 * jnp.cos(4.0 * jnp.pi * X) * jnp.sin(2.0 * jnp.pi * Y)

    # solve div(v) = -nabla^2 phi for phi
    div_v = div(vx, vy)
    phi_k = -1j * div_v / (k_sq + (k_sq == 0))
    phi = jnp.fft.ifftn(phi_k).real
    theta = -m_per_hbar * phi

    # v = -grad(phi) + A
    grad_phi_x, grad_phi_y = grad(phi)
    Ax = vx + grad_phi_x
    Ay = vy + grad_phi_y

    state = {}
    state["psi"] = jnp.sqrt(rho) * jnp.exp(1j * theta)
    state["Ax"] = Ax
    state["Ay"] = Ay
    state["dx"] = dx
    state["dt"] = dt

    return state


def plot_solution(state):
    """
    Plot the solution
    """
    plt.figure(figsize=(6, 4), dpi=240)

    plt.imshow(jnp.abs(state["psi"]).T, cmap="inferno")
    plt.clim(0.95, 1.1)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig("output_quantum.png", dpi=240)
    plt.show()


def main():
    state = set_up_state()
    state = run_simulation(state)
    plot_solution(state)


if __name__ == "__main__":
    main()
