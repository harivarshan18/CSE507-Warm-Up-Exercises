# snakes.py
# Classic Kass–Witkin–Terzopoulos active contour (1988), refactored.
# - Semi-implicit update: (A + γI) v^t = γ v^{t-1} - f(v^{t-1})
# - Image energies: line, edge, termination; Force = -∇E
# - Bilinear force sampling, arc-length resampling, multi-scale schedule
from __future__ import annotations
import numpy as np

# Optional SciPy fast paths
try:
    from scipy.ndimage import gaussian_filter, sobel
    _HAS_SCIPY_ND = True
except Exception:
    _HAS_SCIPY_ND = False

try:
    from scipy.linalg import lu_factor, lu_solve
    _HAS_SCIPY_LA = True
except Exception:
    _HAS_SCIPY_LA = False


# --------------------------- low-level image ops ---------------------------

def _gauss(img, sigma: float):
    if sigma <= 0:
        return img
    if _HAS_SCIPY_ND:
        return gaussian_filter(img, sigma=sigma, mode="reflect")
    # tiny separable fallback
    rad = max(1, int(3*sigma))
    x = np.arange(-rad, rad+1, dtype=np.float64)
    k = np.exp(-(x*x)/(2*sigma*sigma)); k /= k.sum()
    tmp = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, img)
    out = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, tmp)
    return out

def _gradients(img):
    if _HAS_SCIPY_ND:
        gx = sobel(img, axis=1, mode="reflect") / 8.0
        gy = sobel(img, axis=0, mode="reflect") / 8.0
        return gx, gy
    gx = np.zeros_like(img, dtype=np.float64)
    gy = np.zeros_like(img, dtype=np.float64)
    gx[:, 1:-1] = 0.5*(img[:, 2:] - img[:, :-2])
    gx[:, 0] = img[:, 1] - img[:, 0]
    gx[:, -1] = img[:, -1] - img[:, -2]
    gy[1:-1, :] = 0.5*(img[2:, :] - img[:-2, :])
    gy[0, :] = img[1, :] - img[0, :]
    gy[-1, :] = img[-1, :] - img[-2, :]
    return gx, gy

# --------------------------- helper ---------------------------

def radial_window(h, w, cx, cy, r, inner=0.55, outer=1.55, softness=0.15):
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - cx, yy - cy) / (r + 1e-6)
    # two soft sigmoids to form an annulus
    def s(x): return 1.0/(1.0 + np.exp(-x/softness))
    return (s(d - inner) * (1.0 - s(d - outer)))

def _unit_normals(xy, closed=True):
    p = xy
    if closed:
        prev = np.roll(p, 1, axis=0)
        nxt  = np.roll(p,-1, axis=0)
    else:
        prev = np.vstack([p[0], p[:-1]])
        nxt  = np.vstack([p[1:], p[-1]])
    t = nxt - prev
    n = np.stack([-t[:,1], t[:,0]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n

def snap_to_edge(image, xy, sigma=1.0, search_px=3):
    Iσ = _gauss(image.astype(float), sigma)
    Ix, Iy = _gradients(Iσ)
    strength = Ix*Ix + Iy*Iy
    n = _unit_normals(xy, closed=True)
    out = xy.copy()
    h, w = image.shape

    def bilinear(F, xs, ys):
        xs = np.clip(xs, 0, w-1); ys = np.clip(ys, 0, h-1)
        x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
        x1 = np.clip(x0+1, 0, w-1);   y1 = np.clip(y0+1, 0, h-1)
        wa = (x1-xs)*(y1-ys); wb = (xs-x0)*(y1-ys)
        wc = (x1-xs)*(ys-y0); wd = (xs-x0)*(ys-y0)
        return wa*F[y0,x0] + wb*F[y0,x1] + wc*F[y1,x0] + wd*F[y1,x1]

    for i, (x, y) in enumerate(xy):
        nx, ny = n[i]
        # coarse search to the best integer step in [-search_px, +search_px]
        best_s = 0.0
        best_v = bilinear(strength, x, y)
        for s in range(-search_px, search_px+1):
            xs, ys = x + s*nx, y + s*ny
            v = bilinear(strength, xs, ys)
            if v > best_v:
                best_v, best_s = v, float(s)

        # sub-pixel refine around best_s using 3 samples (-1,0,+1)
        s0 = best_s
        vm = bilinear(strength, x + (s0-1)*nx, y + (s0-1)*ny)
        v0 = bilinear(strength, x +  s0   *nx, y +  s0   *ny)
        vp = bilinear(strength, x + (s0+1)*nx, y + (s0+1)*ny)

        denom = (vm - 2*v0 + vp)
        if np.abs(denom) > 1e-12:
            delta = 0.5*(vm - vp)/denom   # argmax of quadratic
            delta = np.clip(delta, -1.0, 1.0)
        else:
            delta = 0.0

        s_star = s0 + delta
        out[i, 0] = np.clip(x + s_star*nx, 0, w-1)
        out[i, 1] = np.clip(y + s_star*ny, 0, h-1)

    return out



# --------------------------- snake internals ---------------------------

def _termination_energy(I_sigma):
    # κ, curvature of level lines: Sec. 3.4 (paper)
    Ix, Iy = _gradients(I_sigma)
    Ixx, Ixy = _gradients(Ix)
    _,   Iyy = _gradients(Iy)
    num = (Ix*Ix)*Iyy - 2*Ix*Iy*Ixy + (Iy*Iy)*Ixx
    den = (Ix*Ix + Iy*Iy)
    eps = 1e-8
    return num / np.power(den + eps, 1.5)

def _build_A(n, alpha, beta, closed=True):
    # Discrete α·D2 − β·D4 with periodic wrap for closed snakes
    a, b = alpha, beta
    diag0 = (2*a + 6*b) * np.ones(n)
    diag1 = (-a - 4*b) * np.ones(n-1)
    diag2 = (b) * np.ones(n-2)

    A = np.zeros((n, n))
    np.fill_diagonal(A, diag0)
    np.fill_diagonal(A[1:],  diag1)
    np.fill_diagonal(A[:,1:],diag1)
    np.fill_diagonal(A[2:],  diag2)
    np.fill_diagonal(A[:,2:],diag2)

    if closed:
        A[0, -1] = A[-1, 0] = (-a - 4*b)
        A[0, -2] = A[-2, 0] = b
        A[1, -1] = A[-1, 1] = b
    else:
        A[0,:] = 0; A[:,0] = 0; A[0,0]   = 1e6  # clamp ends
        A[-1,:]=0; A[:,-1]=0; A[-1,-1] = 1e6
    return A

def _prep_solver(M):
    """Return a callable solve(B) for (M X = B). B can be (n,) or (n, k)."""
    if _HAS_SCIPY_LA:
        lu, piv = lu_factor(M)
        return lambda B: lu_solve((lu, piv), B)
    # fallback: explicit inverse (OK for n ~ few hundreds)
    Minv = np.linalg.inv(M)
    return lambda B: Minv @ B

def _interp_force(Fx, Fy, x, y):
    """Bilinear sample (Fx,Fy) at float coords (x,y)."""
    h, w = Fx.shape
    x = np.clip(x, 0, w-1); y = np.clip(y, 0, h-1)
    x0 = np.floor(x).astype(int); x1 = np.clip(x0+1, 0, w-1)
    y0 = np.floor(y).astype(int); y1 = np.clip(y0+1, 0, h-1)
    wa = (x1-x)*(y1-y); wb = (x-x0)*(y1-y)
    wc = (x1-x)*(y-y0); wd = (x-x0)*(y-y0)
    Fx_s = wa*Fx[y0,x0] + wb*Fx[y0,x1] + wc*Fx[y1,x0] + wd*Fx[y1,x1]
    Fy_s = wa*Fy[y0,x0] + wb*Fy[y0,x1] + wc*Fy[y1,x0] + wd*Fy[y1,x1]
    return Fx_s, Fy_s


# --------------------------- image energy / forces ---------------------------

def force_field(image, w_line=0.0, w_edge=1.0, w_term=0.0, sigma=1.0, clip_percentile: float | None = 95.0, gain: float = 1.0):
    """
    Build E = w_line*Iσ + w_edge*(-||∇Iσ||^2) + w_term*κ and return f = -∇E.
    """
    I = image.astype(np.float64)
    Iσ = _gauss(I, sigma) if sigma > 0 else I

    E_line = Iσ
    Ix, Iy = _gradients(Iσ)
    E_edge = -(Ix*Ix + Iy*Iy)
    E_term = _termination_energy(Iσ)

    E = w_line*E_line + w_edge*E_edge + w_term*E_term
    Ex, Ey = _gradients(E)
    Fx = -Ex; Fy = -Ey
    if clip_percentile is not None:
        mag = np.hypot(Fx, Fy)
        thr = np.percentile(mag, clip_percentile)
        s = np.maximum(1.0, mag/np.maximum(thr, 1e-12))
        Fx /= s; Fy /= s  # clip large spikes

    Fx *= gain; Fy *= gain
    return Fx, Fy

def edge_strength_map(image, sigma=1.5):
    """Convenience for plotting ||∇(Gσ*I)||^2."""
    Iσ = _gauss(image.astype(float), sigma)
    Ix, Iy = _gradients(Iσ)
    return Ix*Ix + Iy*Iy


# --------------------------- public solvers ---------------------------

def resample_by_arclength(xy, n, closed=True):
    """Evenly re-parameterize a polyline (N×2) into n points."""
    p = np.asarray(xy, float)
    if closed and not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[0]])
    seg = np.linalg.norm(p[1:] - p[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.repeat(p[:1], n, axis=0)
    u = np.linspace(0, s[-1], n, endpoint=not closed)
    out = np.empty((n, 2), float); j = 0
    for i, ui in enumerate(u):
        while s[j+1] < ui: j += 1
        t = (ui - s[j]) / max(1e-12, s[j+1]-s[j])
        out[i] = (1-t)*p[j] + t*p[j+1]
    return out

def active_contour(
    image: np.ndarray,
    init_xy: np.ndarray,
    alpha: float = 0.1,
    beta: float = 2.0,
    gamma: float = 1.0,
    w_line: float = 0.0,
    w_edge: float = 1.0,
    w_term: float = 0.0,
    sigma: float = 1.0,
    max_iters: int = 2500,
    conv_tol: float = 1e-3,
    closed: bool = True,
    resample_every: int | None = None,
    force_field_precomp: tuple[np.ndarray, np.ndarray] | None = None,
    verbose: bool = False,
    max_step_px: float | None = None
):
    """
    Single-stage snake with optional resampling and precomputed forces.
    """
    img = image.astype(np.float64)
    pts = np.array(init_xy, dtype=np.float64)
    x, y = pts[:,0].copy(), pts[:,1].copy()
    n = len(x)

    A = _build_A(n, alpha, beta, closed=closed)
    M = A + gamma*np.eye(n)
    solve = _prep_solver(M)

    if force_field_precomp is None:
        Fx, Fy = force_field(img, w_line=w_line, w_edge=w_edge, w_term=w_term, sigma=sigma)
    else:
        Fx, Fy = force_field_precomp

    last = np.stack([x, y], axis=1)
    h, w = img.shape[:2]

    for it in range(max_iters):
        fx, fy = _interp_force(Fx, Fy, x, y)
        x_new = solve(gamma*x - fx)
        y_new = solve(gamma*y - fy)
        if max_step_px is not None:
            dx = x_new - x; dy = y_new - y
            step = np.hypot(dx, dy) + 1e-12
            s = np.minimum(1.0, max_step_px / step)
            x_new = x + s*dx
            y_new = y + s*dy
        x_new = np.clip(x_new, 0, w-1); y_new = np.clip(y_new, 0, h-1)

        v = np.stack([x_new, y_new], axis=1)
        shift = np.linalg.norm(v - last, axis=1).mean()
        x, y = x_new, y_new
        last = v

        if resample_every and ((it+1) % resample_every == 0):
            v = resample_by_arclength(v, n, closed=closed)
            x, y = v[:,0], v[:,1]
            last = v

        if verbose and (it % 50 == 0 or it == max_iters-1):
            print(f"[snake] it={it:4d} mean-shift={shift:.6f}")
        if shift < conv_tol:
            break

    return np.stack([x, y], axis=1)

def snake_stage(
    image: np.ndarray, init_xy: np.ndarray,
    alpha: float, beta: float, gamma: float,
    w_line: float = 0.0, w_edge: float = 1.0, w_term: float = 0.0,
    sigma: float = 1.5, max_iters: int = 400, conv_tol: float = 1e-3,
    closed: bool = True, resample_every: int = 40, verbose: bool = False,
):
    """Run one scale (fixed σ), caching and optionally windowing the force field."""
    # 1) precompute force field with clipping/gain
    fine = (sigma <= 1.0)
    Fx, Fy = force_field(
        image, w_line=w_line, w_edge=w_edge, w_term=w_term,
        sigma=sigma,
        clip_percentile=(90.0 if fine else 95.0),
        gain=(0.75 if fine else 1.0),
    )

    # 2) build an annulus window around current init_xy to ignore far coins
    xy = np.asarray(init_xy, float)
    h, w = image.shape
    cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
    r_est = np.mean(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy))
    W = radial_window(h, w, cx, cy, r_est, inner=0.50, outer=1.70, softness=0.20)
    Fx *= W
    Fy *= W
    FF = (Fx, Fy)

    # 3) run one stage using the precomputed (windowed) forces
    return active_contour(
        image=image, init_xy=init_xy, alpha=alpha, beta=beta, gamma=gamma,
        w_line=w_line, w_edge=w_edge, w_term=w_term, sigma=sigma,
        max_iters=max_iters, conv_tol=conv_tol, closed=closed,
        resample_every=resample_every, force_field_precomp=FF, verbose=verbose, 
        max_step_px=(0.8 if sigma <= 1.0 else None)
    )


def multiscale_active_contour(
    image: np.ndarray, init_xy: np.ndarray,
    schedule: list[dict],         # e.g. [{"sigma":3.0,"max_iters":300}, ...]
    alpha: float, beta: float, gamma: float,
    w_line: float = 0.0, w_edge: float = 1.0, w_term: float = 0.0,
    closed: bool = True, resample_every: int = 30, conv_tol: float = 5e-4,
    verbose: bool = False,
):
    """Run several σ levels from coarse→fine."""
    xy = np.array(init_xy, float)
    history = [xy.copy()]
    for st in schedule:
        xy = snake_stage(
            image, xy, alpha=alpha, beta=beta, gamma=gamma,
            w_line=st.get("w_line", w_line),
            w_edge=st.get("w_edge", w_edge),
            w_term=st.get("w_term", w_term),
            sigma=st["sigma"], max_iters=st["max_iters"],
            conv_tol=conv_tol, closed=closed,
            resample_every=resample_every, verbose=verbose,
        )
    # Sub-pixel snap (already in your code)
    if schedule and schedule[-1].get("sigma", 1.0) <= 1.2:
        xy = snap_to_edge(image, xy, sigma=schedule[-1]["sigma"], search_px=3)

        # --- NEW: polish stage (10–20 iterations) ---
        xy = snake_stage(
                image, xy,
                alpha=0.008,   # minimal shrink
                beta=14.0,     # stronger smoothing
                gamma=0.6,
                w_line=0.0, w_edge=1.0, w_term=0.12,
                sigma=max(0.8, schedule[-1]["sigma"]),
                max_iters=25,
                conv_tol=8e-5,
                closed=closed,
                resample_every=8,
                verbose=False,
            )
    return xy, history


# --------------------------- simple initializers ---------------------------

def circle_init(cx, cy, r, n=200, closed=True):
    t = np.linspace(0, 2*np.pi, n, endpoint=not closed)
    return np.column_stack([cx + r*np.cos(t), cy + r*np.sin(t)])

def polyline_init(points, n=200, closed=True):
    pts = np.asarray(points, dtype=np.float64)
    return resample_by_arclength(pts, n, closed=closed)
