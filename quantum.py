import jax
import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Quantum

# Philip Mocz (2025)

# Simulate a compressible isothermal fluid with a quantum Schrodinger-type analog equation.

# spectral method
# kick-drift-kick scheme

# i d(psi)/dt = ( -hbar/(2m) nabla^2 - (i/2)*(2 A.nabla + nabla.A) + (m/hbar)*A^2/2 + (m/hbar)*ln(rho)) psi
# d(A)/dt = v times curl(A)

parser = argparse.ArgumentParser(description="Quantum isothermal fluid simulation")
parser.add_argument("--n", type=int, default=128, help="resolution")
parser.add_argument("--nt", type=int, default=100, help="number of time steps")
parser.add_argument("--t", type=float, default=0.1, help="stopping time")
parser.add_argument("--m", type=float, default=100.0, help="m/hbar")
args = parser.parse_args()

n = args.n
nt = args.nt
t_stop = args.t
m_per_hbar = args.m


box_size = 1.0
dt = t_stop / nt

# Mesh
dx = box_size / n
x_lin = jnp.linspace(0.5 * dx, box_size - 0.5 * dx, n)
X, Y = jnp.meshgrid(x_lin, x_lin, indexing="ij")

# spectral grid
kx_lin = 2.0 * jnp.pi / box_size * jnp.arange(-n / 2, n / 2)
ky_lin = 2.0 * jnp.pi / box_size * jnp.arange(-n / 2, n / 2)
kx, ky = jnp.meshgrid(kx_lin, ky_lin, indexing="ij")
kx = jnp.fft.ifftshift(kx)
ky = jnp.fft.ifftshift(ky)
k_sq = kx**2 + ky**2


def div_real(fx, fy):
    """
    Compute the divergence of a periodic vector field f = (f_x, f_y)
    spectral method
    """
    fx_hat = jnp.fft.fftn(fx)
    fy_hat = jnp.fft.fftn(fy)

    div_hat = 1j * (kx * fx_hat + ky * fy_hat)
    div_f = jnp.fft.ifftn(div_hat).real

    return div_f


def curl_real(fx, fy):
    """
    Compute the curl of a periodic vector field f = (f_x, f_y)
    spectral method
    """
    fx_hat = jnp.fft.fftn(fx)
    fy_hat = jnp.fft.fftn(fy)

    curl_hat = 1j * (kx * fy_hat - ky * fx_hat)
    curl_f = jnp.fft.ifftn(curl_hat).real

    return curl_f


def grad_real(f):
    """
    Compute the gradient of a periodic scalar field f
    spectral method
    """
    f_hat = jnp.fft.fftn(f)

    grad_x_hat = 1j * kx * f_hat
    grad_y_hat = 1j * ky * f_hat

    grad_x = jnp.fft.ifftn(grad_x_hat).real
    grad_y = jnp.fft.ifftn(grad_y_hat).real
    return grad_x, grad_y


def grad(f):
    """
    Compute the gradient of a periodic scalar field f
    spectral method
    """
    f_hat = jnp.fft.fftn(f)

    grad_x_hat = 1j * kx * f_hat
    grad_y_hat = 1j * ky * f_hat

    grad_x = jnp.fft.ifftn(grad_x_hat)
    grad_y = jnp.fft.ifftn(grad_y_hat)
    return grad_x, grad_y


def run_simulation(state):
    tiny = 1e-6

    def kick(state, dt):
        psi = state["psi"]
        Ax = state["Ax"]
        Ay = state["Ay"]

        rho = jnp.abs(psi) ** 2
        psi = psi * jnp.exp(
            -1j * dt * m_per_hbar * ((Ax**2 + Ay**2) / (2.0) + jnp.log(rho + tiny))
        )

        # also include quantum vector potential term
        # i d(psi)/dt = (hbar/(2m))*(nabla^2 sqrt(rho)/sqrt(rho)) psi
        # sx, sy = grad_real(jnp.sqrt(rho))
        # s = div_real(sx, sy)
        # quantum_potential = -0.5 / m_per_hbar * (s / (jnp.sqrt(rho) + tiny))
        # psi = psi * jnp.exp(-1j * dt * quantum_potential)

        # add A terms:
        # i d(psi)/dt = (i/2)*(2 A.nabla + nabla.A) psi
        grad_psi_x, grad_psi_y = grad(psi)
        div_A = div_real(Ax, Ay)
        # psi = psi * jnp.exp(
        #    -dt * (Ax * grad_psi_x + Ay * grad_psi_y + 0.5 * div_A)
        # )
        # do forward euler instead:
        psi = psi - dt * (Ax * grad_psi_x + Ay * grad_psi_y + 0.5 * div_A * psi)

        # do rk2 instead
        # def L_adv(psi, Ax, Ay):
        #    gx, gy = grad(psi)
        #    return -(Ax*gx + Ay*gy + 0.5*div_real(Ax, Ay)*psi)

        # k1 = L_adv(psi, Ax, Ay)
        # psi1 = psi + 0.5*dt*k1
        # k2 = L_adv(psi1, Ax, Ay)
        # psi = psi + dt*k2

        state["psi"] = psi
        return state

    def drift(state, dt):
        psi = state["psi"]
        Ax = state["Ax"]
        Ay = state["Ay"]

        psi_hat = jnp.fft.fftn(psi)
        psi_hat = jnp.exp(dt * (-1.0j * k_sq / m_per_hbar / 2.0)) * psi_hat
        psi = jnp.fft.ifftn(psi_hat)

        # update A
        rho = jnp.abs(psi) ** 2
        # v = -grad(phi) + A
        # psi = R * exp(i theta) = sqrt(rho) * exp(-i * m_per_hbar * phi)

        # theta = jnp.angle(psi)  # XXX
        # phi = -theta / m_per_hbar
        # grad_phi_x, grad_phi_y = grad_real(phi)

        gx, gy = grad(psi)
        inv = 1.0 / (psi + 1e-12)
        grad_theta_x = jnp.imag(gx * inv)
        grad_theta_y = jnp.imag(gy * inv)

        grad_phi_x = -(1.0 / m_per_hbar) * grad_theta_x
        grad_phi_y = -(1.0 / m_per_hbar) * grad_theta_y
        vx = -grad_phi_x + Ax
        vy = -grad_phi_y + Ay

        curl_A = curl_real(Ax, Ay)
        Ax = Ax + dt * (vy * curl_A)
        Ay = Ay - dt * (vx * curl_A)

        state["psi"] = psi
        state["Ax"] = Ax
        state["Ay"] = Ay
        return state

    for _ in range(nt):
        state = kick(state, 0.5 * dt)
        state = drift(state, dt)
        state = kick(state, 0.5 * dt)

    return state


def set_up_state():
    """
    Set up the initial state for the simulation
    """

    # Initial condition
    rho = jnp.ones((n, n))
    vx = 0.2 * jnp.sin(8.0 * jnp.pi * Y)
    vy = -0.2 * jnp.cos(4.0 * jnp.pi * X) * jnp.sin(2.0 * jnp.pi * Y)

    # solve div(v) = -nabla^2 phi for phi
    div_v = div_real(vx, vy)
    div_v_hat = jnp.fft.fftn(div_v)
    phi_hat = -div_v_hat / (k_sq + (k_sq == 0))
    phi = jnp.fft.ifftn(phi_hat).real
    theta = -m_per_hbar * phi

    # v = -grad(phi) + A
    grad_phi_x, grad_phi_y = grad_real(phi)
    Ax = vx + grad_phi_x
    Ay = vy + grad_phi_y

    state = {}
    state["psi"] = jnp.sqrt(rho) * jnp.exp(1j * theta)
    state["Ax"] = Ax
    state["Ay"] = Ay

    return state


def plot_solution(state):
    """
    Plot the solution
    """
    plt.figure(figsize=(6, 4), dpi=240)

    plt.imshow((jnp.abs(state["psi"]) ** 2).T, cmap="inferno")
    plt.clim(0.95, 1.1)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(
        f"output_quantum_t{t_stop}_n{n}_nt{nt}_m{m_per_hbar}.png",
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
