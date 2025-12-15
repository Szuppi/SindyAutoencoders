import numpy as np
from scipy.integrate import odeint
from scipy.special import legendre, chebyt
import sys
sys.path.append('../src')
import os
#print(os.getcwd())
from sindy_utils import library_size
from scipy.signal import savgol_filter


def finite_diff_first(y, dt):
    """
    y: (T, D)
    returns dy: (T, D)
    """
    dy = np.empty_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
    dy[0]    = (y[1] - y[0]) / dt
    dy[-1]   = (y[-1] - y[-2]) / dt
    return dy

def finite_diff_second(y, dt):
    """
    y: (T, D)
    returns ddy: (T, D)
    """
    ddy = np.empty_like(y)
    ddy[1:-1] = (y[2:] - 2.0*y[1:-1] + y[:-2]) / (dt**2)
    # boundary copy
    ddy[0]  = ddy[1]
    ddy[-1] = ddy[-2]
    return ddy

def maybe_smooth(y, window=11, poly=3):
    """
    Savitzky-Golay smoothing along time axis.
    y: (T, D)
    """
    if window is None:
        return y
    # window must be odd and <= T
    T = y.shape[0]
    w = int(window)
    if w >= T:
        w = T - 1 if (T - 1) % 2 == 1 else T - 2
    if w < 5:
        return y
    if w % 2 == 0:
        w += 1
    return savgol_filter(y, window_length=w, polyorder=poly, axis=0, mode="interp")


'''def get_lorenz_data(n_ics, noise_strength=0, smooth=True, sg_window=11, sg_poly=3, seed=0):
    t = np.arange(0, 5, .02)
    n_steps = t.size
    input_dim = 128
    dt = t[1] - t[0]

    ic_means = np.array([0, 0, 25])
    ic_widths = 2*np.array([36, 48, 41])

    # training ICs
    rng = np.random.RandomState(seed)
    ics = ic_widths*(rng.rand(n_ics, 3) - .5) + ic_means

    # Generate CLEAN observed data via your DCL-style mixer
    data = generate_lorenz_data(
        ics, t, input_dim,
        linear=False,
        normalization=np.array([1/40, 1/40, 1/40]),
        seed=seed
    )

    # clean observed (T,N) per trajectory lives here (your generator sets x_obs into x and x_nl)
    x_clean = data['x_nl']  # shape: (n_ics, n_steps, input_dim)

    # --- make x noisy (ONLY x gets noise!) ---
    noise = noise_strength * rng.randn(*x_clean.shape).astype(np.float32)
    x_noisy = (x_clean + noise).astype(np.float32)

    # --- optional smoothing on the noisy x, per trajectory ---
    x_used = np.empty_like(x_noisy)
    if smooth:
        for i in range(n_ics):
            x_used[i] = maybe_smooth(x_noisy[i], window=sg_window, poly=sg_poly).astype(np.float32)
    else:
        x_used = x_noisy

    # --- finite difference derivatives from x_used ---
    dx_fd  = np.empty_like(x_used)
    ddx_fd = np.empty_like(x_used)
    for i in range(n_ics):
        dx_fd[i]  = finite_diff_first(x_used[i], dt).astype(np.float32)
        ddx_fd[i] = finite_diff_second(x_used[i], dt).astype(np.float32)

    # --- flatten to match AE-SINDy expected shapes ---
    data['x']   = x_used.reshape((-1, input_dim))
    data['dx']  = dx_fd.reshape((-1, input_dim))
    data['ddx'] = ddx_fd.reshape((-1, input_dim))

    # optional debug exports (kept unflattened too if you like)
    data['x_noisy'] = x_noisy
    data['x_used']  = x_used
    data['dx_fd']   = dx_fd
    data['ddx_fd']  = ddx_fd

    # keep original clean analytic versions for comparison (from generator)
    # (these are NOT consistent with noisy x; for diagnostics only)
    data['x_clean_flat']   = x_clean.reshape((-1, input_dim))
    data['dx_clean_flat']  = data['dx_nl'].reshape((-1, input_dim))
    data['ddx_clean_flat'] = data['ddx_nl'].reshape((-1, input_dim))

    return data'''


def get_lorenz_data(n_ics, noise_strength=0, smooth=True, sg_window=11, sg_poly=3,
                    seed=0, input_dim=128, t_step=0.01):
    t = np.arange(0, 5, t_step)
    dt = t[1] - t[0]

    ic_means = np.array([0, 0, 25])
    ic_widths = 2*np.array([36, 48, 41])
    rng = np.random.RandomState(seed)
    ics = ic_widths*(rng.rand(n_ics, 3) - .5) + ic_means

    data = generate_lorenz_data(
        ics, t, n_points=input_dim,
        linear=False,
        normalization=np.array([1/40, 1/40, 1/40]),
        seed=seed,
        noise_strength=noise_strength,
        smooth=smooth,
        sg_window=sg_window,
        sg_poly=sg_poly,
    )

    # If your TF pipeline expects flattened:
    data['x']   = data['x'].reshape((-1, input_dim))
    data['dx']  = data['dx'].reshape((-1, input_dim))
    data['ddx'] = data['ddx'].reshape((-1, input_dim))

    return data



def lorenz_coefficients(normalization, poly_order=3, sigma=10., beta=8/3, rho=28.):
    """
    Generate the SINDy coefficient matrix for the Lorenz system.

    Arguments:
        normalization - 3-element list of array specifying scaling of each Lorenz variable
        poly_order - Polynomial order of the SINDy model.
        sigma, beta, rho - Parameters of the Lorenz system
    """
    Xi = np.zeros((library_size(3,poly_order),3))
    Xi[1,0] = -sigma
    Xi[2,0] = sigma*normalization[0]/normalization[1]
    Xi[1,1] = rho*normalization[1]/normalization[0]
    Xi[2,1] = -1
    Xi[6,1] = -normalization[1]/(normalization[0]*normalization[2])
    Xi[3,2] = -beta
    Xi[5,2] = normalization[2]/(normalization[0]*normalization[1])
    return Xi


def simulate_lorenz(z0, t, sigma=10., beta=8/3, rho=28.):
    """
    Simulate the Lorenz dynamics.

    Arguments:
        z0 - Initial condition in the form of a 3-value list or array.
        t - Array of time points at which to simulate.
        sigma, beta, rho - Lorenz parameters

    Returns:
        z, dz, ddz - Arrays of the trajectory values and their 1st and 2nd derivatives.
    """
    f = lambda z,t : [sigma*(z[1] - z[0]), z[0]*(rho - z[2]) - z[1], z[0]*z[1] - beta*z[2]]
    df = lambda z,dz,t : [sigma*(dz[1] - dz[0]),
                          dz[0]*(rho - z[2]) + z[0]*(-dz[2]) - dz[1],
                          dz[0]*z[1] + z[0]*dz[1] - beta*dz[2]]

    z = odeint(f, z0, t)

    dt = t[1] - t[0]
    dz = np.zeros(z.shape)
    ddz = np.zeros(z.shape)
    for i in range(t.size):
        dz[i] = f(z[i],dt*i)
        ddz[i] = df(z[i], dz[i], dt*i)
    return z, dz, ddz


def generate_lorenz_data(
    ics,
    t,
    n_points,
    linear=True,                 # kept for signature compatibility; unused here
    normalization=None,
    sigma=10,
    beta=8/3,
    rho=28,
    seed=0,
    # --- new knobs (safe defaults keep old behavior: clean x, FD derivatives) ---
    noise_strength=0.0,          # std of additive Gaussian noise applied to x ONLY
    smooth=True,                # if True, smooth x before finite differences
    sg_window=11,                # Savitzky-Golay window length (odd)
    sg_poly=3,                   # Savitzky-Golay poly order
    cond_thresh=50.0,            # max condition number allowed per MLP layer weight
    n_layers=4,                  # number of square linear layers in the invertible-ish MLP
    leaky_alpha=0.2,             # leaky ReLU slope
    add_bias=True,               # include bias in final linear map to R^n
):
    """
    Generate high-dimensional Lorenz dataset with DCL-style mixing and FD derivatives.

    Pipeline:
        z(t) in R^3  --(invertible-ish MLP h: R^3->R^3)--> h(z)
                    --(linear map A: R^3->R^n [+ b])--> x_clean(t) in R^n
                    --(+ Gaussian noise)--> x_noisy(t)
                    --(optional Savitzky-Golay smoothing)--> x_used(t)
                    --(finite differences)--> dx_used(t), ddx_used(t)

    Outputs:
        data['x_nl']   = x_used (what AE-SINDy should train/test on)
        data['dx_nl']  = finite-diff dx from x_used
        data['ddx_nl'] = finite-diff ddx from x_used

    Also includes:
        data['x_clean'], data['x_noisy'], data['x_used'] for debugging.
    """

    ics = np.asarray(ics, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    n_ics = ics.shape[0]
    n_steps = t.size
    if n_steps < 3:
        raise ValueError("Need at least 3 time points for finite differences.")
    dt = float(t[1] - t[0])

    # -----------------------
    # 1) Simulate latent Lorenz
    # -----------------------
    d = 3
    z = np.zeros((n_ics, n_steps, d), dtype=np.float32)
    dz = np.zeros_like(z)
    ddz = np.zeros_like(z)
    for i in range(n_ics):
        zi, dzi, ddzi = simulate_lorenz(ics[i], t, sigma=sigma, beta=beta, rho=rho)
        z[i] = zi.astype(np.float32)
        dz[i] = dzi.astype(np.float32)
        ddz[i] = ddzi.astype(np.float32)

    if normalization is not None:
        norm = np.asarray(normalization, dtype=np.float32).reshape((1, 1, 3))
        z *= norm
        dz *= norm
        ddz *= norm

    # -----------------------
    # 2) Build DCL-style invertible-ish MLP h: R^3 -> R^3 (fixed random weights)
    # -----------------------
    rng = np.random.RandomState(seed)

    def _col_normalize(W):
        return W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)

    def _sample_well_conditioned_W(dim=3, max_tries=20000):
        # Similar spirit to DCL: sample many random matrices and keep those with acceptable cond
        for _ in range(max_tries):
            W = rng.uniform(-1.0, 1.0, size=(dim, dim)).astype(np.float32)
            W = _col_normalize(W).astype(np.float32)
            if np.linalg.cond(W) < cond_thresh:
                return W
        raise RuntimeError(
            f"Failed to sample well-conditioned W (cond<{cond_thresh}). "
            f"Try increasing cond_thresh or max_tries."
        )

    Ws = [_sample_well_conditioned_W(dim=3) for _ in range(int(n_layers))]

    def _leaky_relu(x):
        return np.where(x >= 0.0, x, leaky_alpha * x)

    def h_mlp(z_batch_2d):
        # z_batch_2d: (T,3) or (N,3)
        xh = z_batch_2d
        for li, W in enumerate(Ws):
            xh = xh @ W.T
            if li < len(Ws) - 1:
                xh = _leaky_relu(xh)
        return xh

    # -----------------------
    # 3) Linear map to observed dimension: A: R^3 -> R^n (+ optional bias)
    # -----------------------
    n = int(n_points)
    if n <= 0:
        raise ValueError("n_points must be positive.")
    A = (rng.randn(n, 3).astype(np.float32) / np.sqrt(3.0)).astype(np.float32)
    b = rng.uniform(-0.5, 0.5, size=(n,)).astype(np.float32) if add_bias else np.zeros((n,), dtype=np.float32)

    x_clean = np.zeros((n_ics, n_steps, n), dtype=np.float32)
    for i in range(n_ics):
        hz = h_mlp(z[i])              # (n_steps,3)
        x_clean[i] = hz @ A.T + b     # (n_steps,n)

    # -----------------------
    # 4) Add noise to x ONLY (optional), then optional smoothing
    # -----------------------
    if noise_strength and noise_strength > 0.0:
        x_noisy = (x_clean + noise_strength * rng.randn(*x_clean.shape).astype(np.float32)).astype(np.float32)
    else:
        x_noisy = x_clean

    if smooth:
        x_used = np.empty_like(x_noisy)
        for i in range(n_ics):
            x_used[i] = maybe_smooth(x_noisy[i], window=sg_window, poly=sg_poly).astype(np.float32)
    else:
        x_used = x_noisy

    # -----------------------
    # 5) Finite-difference derivatives computed FROM x_used
    # -----------------------
    dx_used = np.empty_like(x_used)
    ddx_used = np.empty_like(x_used)
    for i in range(n_ics):
        dx_used[i] = finite_diff_first(x_used[i], dt).astype(np.float32)
        ddx_used[i] = finite_diff_second(x_used[i], dt).astype(np.float32)

    # -----------------------
    # 6) True SINDy coefficients for Lorenz (in latent space)
    # -----------------------
    if normalization is None:
        sindy_coefficients = lorenz_coefficients([1, 1, 1], sigma=sigma, beta=beta, rho=rho)
    else:
        sindy_coefficients = lorenz_coefficients(np.asarray(normalization), sigma=sigma, beta=beta, rho=rho)

    # -----------------------
    # 7) Package dict (AE-SINDy compatibility)
    # -----------------------
    data = {}
    data["t"] = t
    data["z"] = z
    data["dz"] = dz
    data["ddz"] = ddz
    data["sindy_coefficients"] = sindy_coefficients.astype(np.float32)

    # "nonlinear observed" that AE-SINDy will use
    data["x_nl"] = x_used
    data["dx_nl"] = dx_used
    data["ddx_nl"] = ddx_used

    # keep these too (some codepaths might expect them)
    data["x"] = x_used
    data["dx"] = dx_used
    data["ddx"] = ddx_used

    # debug/diagnostics
    data["x_clean"] = x_clean
    data["x_noisy"] = x_noisy
    data["x_used"] = x_used

    # mixing params for reproducibility
    data["mixing_seed"] = int(seed)
    data["mixing_Ws"] = Ws
    data["mixing_A"] = A
    data["mixing_b"] = b

    return data



'''def generate_lorenz_data(ics, t, n_points, linear=True, normalization=None,
                            sigma=10, beta=8/3, rho=28):
    """
    Generate high-dimensional Lorenz data set.

    Arguments:
        ics - Nx3 array of N initial conditions
        t - array of time points over which to simulate
        n_points - size of the high-dimensional dataset created
        linear - Boolean value. If True, high-dimensional dataset is a linear combination
        of the Lorenz dynamics. If False, the dataset also includes cubic modes.
        normalization - Optional 3-value array for rescaling the 3 Lorenz variables.
        sigma, beta, rho - Parameters of the Lorenz dynamics.

    Returns:
        data - Dictionary containing elements of the dataset. This includes the time points (t),
        spatial mapping (y_spatial), high-dimensional modes used to generate the full dataset
        (modes), low-dimensional Lorenz dynamics (z, along with 1st and 2nd derivatives dz and
        ddz), high-dimensional dataset (x, along with 1st and 2nd derivatives dx and ddx), and
        the true Lorenz coefficient matrix for SINDy.
    """

    n_ics = ics.shape[0]
    n_steps = t.size
    dt = t[1]-t[0]

    d = 3
    z = np.zeros((n_ics,n_steps,d))
    dz = np.zeros(z.shape)
    ddz = np.zeros(z.shape)
    for i in range(n_ics):
        z[i], dz[i], ddz[i] = simulate_lorenz(ics[i], t, sigma=sigma, beta=beta, rho=rho)


    if normalization is not None:
        z *= normalization
        dz *= normalization
        ddz *= normalization

    n = n_points
    L = 1
    y_spatial = np.linspace(-L,L,n)

    rng = np.random.RandomState(0)
    M = rng.randn(n, n).astype(np.float32) / np.sqrt(n)   # (128,128)
    b = rng.uniform(-0.3, 0.3, size=n).astype(np.float32) # (128,)
    a = 6.0  # strong saturation


    modes = np.zeros((2*d, n))
    for i in range(2*d):
        modes[i] = legendre(i)(y_spatial)
        # modes[i] = chebyt(i)(y_spatial)
        # modes[i] = np.cos((i+1)*np.pi*y_spatial/2)
    x1 = np.zeros((n_ics,n_steps,n))
    x2 = np.zeros((n_ics,n_steps,n))
    x3 = np.zeros((n_ics,n_steps,n))
    x4 = np.zeros((n_ics,n_steps,n))
    x5 = np.zeros((n_ics,n_steps,n))
    x6 = np.zeros((n_ics,n_steps,n))

    x     = np.zeros((n_ics, n_steps, n))
    dx    = np.zeros_like(x)
    ddx   = np.zeros_like(x)
    x_nl   = np.zeros_like(x)
    dx_nl  = np.zeros_like(x)
    ddx_nl = np.zeros_like(x)

    a = 6.0  # gain; 1.0–3.0

    for i in range(n_ics):
        for j in range(n_steps):
            # --- original construction (same as before) ---
            x1[i,j] = modes[0]*z[i,j,0]
            x2[i,j] = modes[1]*z[i,j,1]
            x3[i,j] = modes[2]*z[i,j,2]
            x4[i,j] = modes[3]*z[i,j,0]**3
            x5[i,j] = modes[4]*z[i,j,1]**3
            x6[i,j] = modes[5]*z[i,j,2]**3

            x_lin = x1[i,j] + x2[i,j] + x3[i,j]
            if not linear:
                x_lin = x_lin + x4[i,j] + x5[i,j] + x6[i,j]
            x[i,j] = x_lin

            dx_lin = modes[0]*dz[i,j,0] + modes[1]*dz[i,j,1] + modes[2]*dz[i,j,2]
            if not linear:
                dx_lin = dx_lin \
                       + modes[3]*3*(z[i,j,0]**2)*dz[i,j,0] \
                       + modes[4]*3*(z[i,j,1]**2)*dz[i,j,1] \
                       + modes[5]*3*(z[i,j,2]**2)*dz[i,j,2]
            dx[i,j] = dx_lin

            ddx_lin = modes[0]*ddz[i,j,0] + modes[1]*ddz[i,j,1] + modes[2]*ddz[i,j,2]
            if not linear:
                ddx_lin = ddx_lin \
                        + modes[3]*(6*z[i,j,0]*dz[i,j,0]**2 + 3*(z[i,j,0]**2)*ddz[i,j,0]) \
                        + modes[4]*(6*z[i,j,1]*dz[i,j,1]**2 + 3*(z[i,j,1]**2)*ddz[i,j,1]) \
                        + modes[5]*(6*z[i,j,2]*dz[i,j,2]**2 + 3*(z[i,j,2]**2)*ddz[i,j,2])
            ddx[i,j] = ddx_lin

            # --- nonlinear sensor version (tanh) ---
            u = M @ x_lin              # (128,)
            du = M @ dx_lin            # (128,)
            ddu = M @ ddx_lin          # (128,)

            x_nl[i,j] = np.tanh(a * u + b)   # (128,)

            sech2 = 1.0 - x_nl[i,j]**2
            dx_nl[i,j]  = a * sech2 * du
            ddx_nl[i,j] = a * sech2 * ddu + (-2.0 * a**2 * x_nl[i,j] * sech2) * (du**2)



    if normalization is None:
        sindy_coefficients = lorenz_coefficients([1,1,1], sigma=sigma, beta=beta, rho=rho)
    else:
        sindy_coefficients = lorenz_coefficients(normalization, sigma=sigma, beta=beta, rho=rho)

    data = {}
    data['t'] = t
    data['y_spatial'] = y_spatial
    data['modes'] = modes
    data['x'] = x
    data['dx'] = dx
    data['x_nl'] = x_nl
    data['dx_nl'] = dx_nl
    data['ddx'] = ddx
    data['ddx_nl'] = ddx_nl
    data['z'] = z
    data['dz'] = dz
    data['ddz'] = ddz
    data['sindy_coefficients'] = sindy_coefficients.astype(np.float32)

    return data
    x = np.zeros((n_ics,n_steps,n))
    dx = np.zeros(x.shape)
    ddx = np.zeros(x.shape)
    for i in range(n_ics):
        for j in range(n_steps):
            x1[i,j] = modes[0]*z[i,j,0]
            x2[i,j] = modes[1]*z[i,j,1]
            x3[i,j] = modes[2]*z[i,j,2]
            x4[i,j] = modes[3]*z[i,j,0]**3
            x5[i,j] = modes[4]*z[i,j,1]**3
            x6[i,j] = modes[5]*z[i,j,2]**3

            x[i,j] = x1[i,j] + x2[i,j] + x3[i,j]
            if not linear:
                x[i,j] += x4[i,j] + x5[i,j] + x6[i,j]

            dx[i,j] = modes[0]*dz[i,j,0] + modes[1]*dz[i,j,1] + modes[2]*dz[i,j,2]
            if not linear:
                dx[i,j] += modes[3]*3*(z[i,j,0]**2)*dz[i,j,0] + modes[4]*3*(z[i,j,1]**2)*dz[i,j,1] + modes[5]*3*(z[i,j,2]**2)*dz[i,j,2]
            
            ddx[i,j] = modes[0]*ddz[i,j,0] + modes[1]*ddz[i,j,1] + modes[2]*ddz[i,j,2]
            if not linear:
                ddx[i,j] += modes[3]*(6*z[i,j,0]*dz[i,j,0]**2 + 3*(z[i,j,0]**2)*ddz[i,j,0]) \
                          + modes[4]*(6*z[i,j,1]*dz[i,j,1]**2 + 3*(z[i,j,1]**2)*ddz[i,j,1]) \
                          + modes[5]*(6*z[i,j,2]*dz[i,j,2]**2 + 3*(z[i,j,2]**2)*ddz[i,j,2])

            # ---------- Adding Nonlinearity ----------
            a = 2.0  # gain; try 1.0–3.0
            x_lin  = x[i, j].copy()
            dx_lin = dx[i, j].copy()
            x[i, j]  = np.tanh(a * x_lin)
            dx[i, j] = a * (1.0 - x[i, j]**2) * dx_lin  # chain rule: d/dt tanh(a u) = a (1 - tanh^2(a u)) u_dot
            # -----------------------------------------'''


