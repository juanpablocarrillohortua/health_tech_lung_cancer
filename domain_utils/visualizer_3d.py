"""
Explorador interactivo de la máscara de un nódulo sobre su volumen de TC.

Recibe un volumen 3D en HU y la máscara binaria de la segmentación (por
ejemplo la ``total`` que devuelve ``segment_nodule`` en medical_criteria) y
monta, con ipywidgets, una interfaz para recorrer los tres planos
ortogonales, controlar el dibujado de la máscara, girar una vista 3D de la
superficie y leer las métricas básicas de la lesión.

Uso desde una notebook::

    from domain_utils.visualizer_3d import explore_mask

    explore_mask(hu[0], total[0, 0], spacing=0.625)

Acepta arrays de numpy o tensores de torch (en CPU o GPU) con dimensiones
de lote o canal al frente, que se eliminan si son de tamaño 1.

Requiere ``ipywidgets``; VS Code y JupyterLab lo renderizan sin extensiones
adicionales. Sin scikit-image no hay marching cubes, así que la vista 3D
dibuja los vóxeles de superficie de la máscara (máscara menos su erosión)
como una nube de puntos, no como una malla. Es más tosca, pero no exige
instalar nada más.

Colores: la máscara total va en azul y el componente sólido en naranja
(slots 1 y 2 de la paleta validada; par con ΔE 24.7 bajo protanopía). El
texto nunca toma el color de la serie.
"""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from scipy import ndimage

INK, TXT, GRID = "#1F2E3D", "#4A4A4A", "#E2E8F0"
MUTED = "#898781"  # crucetas, planos de corte, ejes 3D
PALETTE = ["#2a78d6", "#eb6834"]  # máscara total, componente sólido

SEG_SPACING_MM = 0.625  # el parche de morfometría: 50 mm / 80 px
HU_SOLID = -300.0  # mismo corte que medical_criteria
MAX_SURFACE_POINTS = 20_000  # tope de puntos en la nube 3D

# ventanas de visualización: nombre -> (nivel, ancho) en HU, o None = auto
HU_WINDOWS = {
    "Pulmón (−600 / 1500)": (-600.0, 1500.0),
    "Mediastino (40 / 400)": (40.0, 400.0),
    "Hueso (300 / 1500)": (300.0, 1500.0),
    "Automática (p1–p99)": None,
}
DEFAULT_WINDOW = "Pulmón (−600 / 1500)"
OVERLAY_MODES = ("Contorno", "Relleno")


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
def _to_numpy(x, name: str) -> np.ndarray:
    """Convierte a array 3D (D, H, W) quitando lote/canal de tamaño 1."""
    if hasattr(x, "detach"):  # tensor de torch, en CPU o GPU
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3:
        raise ValueError(
            f"{name} debe ser 3D (D, H, W); llegó con forma {a.shape}"
        )
    return a


def _spacing3(spacing) -> tuple[float, float, float]:
    """mm por vóxel en orden de array (z, y, x); un escalar es isótropo."""
    if np.isscalar(spacing):
        s = (float(spacing),) * 3
    else:
        s = tuple(float(v) for v in spacing)
    if len(s) != 3 or any(v <= 0 for v in s):
        raise ValueError(
            "spacing debe ser un número positivo o una tupla (sz, sy, sx) "
            f"de tres positivos; llegó {spacing!r}"
        )
    return s


def _window(vol: np.ndarray, key: str) -> tuple[float, float]:
    """(vmin, vmax) en HU para la ventana pedida."""
    if key not in HU_WINDOWS:
        raise ValueError(
            f"window debe ser una de {list(HU_WINDOWS)}; llegó {key!r}"
        )
    w = HU_WINDOWS[key]
    if w is None:
        lo, hi = np.percentile(vol, [1, 99])
        return float(lo), float(hi)
    level, width = w
    return level - width / 2, level + width / 2


# ---------------------------------------------------------------------------
# Métricas de la máscara
# ---------------------------------------------------------------------------
def _surface_area_mm2(mask: np.ndarray, spacing, sigma_vox: float = 1.0):
    """Superficie por la fórmula del coárea sobre un indicador suavizado.

    Copia en numpy de ``surface_area_mm2`` de medical_criteria: contar
    caras expuestas sobrestima ~55% en superficies curvas.
    """
    f = ndimage.gaussian_filter(mask.astype(np.float64), sigma_vox)
    g2 = np.zeros_like(f)
    for axis, s in enumerate(spacing):
        g2 += np.gradient(f, s, axis=axis) ** 2
    return float(np.sqrt(g2).sum() * np.prod(spacing))


def _metrics(mask, volume, solid, spacing) -> dict:
    vox_mm3 = float(np.prod(spacing))
    n = int(mask.sum())
    v = n * vox_mm3
    s = _surface_area_mm2(mask, spacing) if n else 0.0
    if n:
        idx = np.argwhere(mask)
        extent = (idx.max(0) - idx.min(0) + 1) * np.asarray(spacing)
        hu_mean = float(volume[mask].mean())
    else:
        extent, hu_mean = np.zeros(3), float("nan")
    return {
        "n_vox": n,
        "volume_mm3": v,
        "diam_equiv_mm": (6 * v / np.pi) ** (1 / 3) if n else 0.0,
        "surface_mm2": s,
        "sphericity": (np.pi ** (1 / 3) * (6 * v) ** (2 / 3) / s)
        if s > 0
        else float("nan"),
        "solid_fraction": int(solid.sum()) / n if n else float("nan"),
        "hu_mean": hu_mean,
        "extent_mm": extent,
    }


def _metrics_html(m: dict, spacing) -> str:
    ez, ey, ex = m["extent_mm"]
    sz, sy, sx = spacing
    rows = [
        ("Vóxeles en la máscara", f"{m['n_vox']:,}"),
        ("Volumen", f"{m['volume_mm3']:,.1f} mm³"),
        ("Diámetro equivalente", f"{m['diam_equiv_mm']:.2f} mm"),
        ("Extensión z × y × x", f"{ez:.1f} × {ey:.1f} × {ex:.1f} mm"),
        ("Superficie", f"{m['surface_mm2']:,.1f} mm²"),
        ("Esfericidad", f"{m['sphericity']:.3f}"),
        ("Fracción sólida (> −300 HU)", f"{m['solid_fraction']:.3f}"),
        ("HU media en la máscara", f"{m['hu_mean']:.1f}"),
        ("Espaciado z × y × x", f"{sz:g} × {sy:g} × {sx:g} mm"),
    ]
    trs = "".join(
        f"<tr><td style='color:{MUTED};padding:2px 10px 2px 0'>{k}</td>"
        f"<td style='text-align:right;font-variant-numeric:tabular-nums'>"
        f"{v}</td></tr>"
        for k, v in rows
    )
    return (
        f"<div style='font-family:system-ui,sans-serif;font-size:12.5px;"
        f"color:{TXT};padding:6px 0 0 10px'>"
        f"<div style='font-weight:600;color:{INK};margin-bottom:4px'>"
        f"Métricas de la máscara</div>"
        f"<table style='border-collapse:collapse'>{trs}</table></div>"
    )


# ---------------------------------------------------------------------------
# Dibujado
# ---------------------------------------------------------------------------
def _surface_points(mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Índices (N, 3) de los vóxeles de superficie, acotados a un tope."""
    if not mask.any():
        return np.zeros((0, 3), dtype=np.int64)
    shell = mask & ~ndimage.binary_erosion(mask)
    pts = np.argwhere(shell)
    if len(pts) > MAX_SURFACE_POINTS:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(pts), MAX_SURFACE_POINTS, replace=False)
        pts = pts[np.sort(keep)]
    return pts


def _draw_slice(
    ax, img, m, s, vmin, vmax, aspect, show, mode, alpha, colors, cross, title
):
    ax.clear()
    ax.imshow(
        img,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        interpolation="nearest",
    )
    if show:
        for arr, c in ((m, colors[0]), (s, colors[1])):
            if not arr.any():
                continue
            if mode == "Contorno" and not arr.all():
                ax.contour(
                    arr.astype(float),
                    levels=[0.5],
                    colors=[c],
                    linewidths=1.5,
                    alpha=alpha,
                )
            else:
                ax.imshow(
                    np.ma.masked_where(~arr, arr),
                    cmap=ListedColormap([c]),
                    alpha=alpha,
                    aspect=aspect,
                    interpolation="nearest",
                )
    row, col = cross
    ax.axhline(row, color=MUTED, lw=0.8, alpha=0.8)
    ax.axvline(col, color=MUTED, lw=0.8, alpha=0.8)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _draw_3d(
    ax, pts_mm, solid_mm, extents, planes, azim, elev, show, alpha, colors
):
    ax.clear()
    zmax, ymax, xmax = extents
    if show:
        for pts, c in ((pts_mm, colors[0]), (solid_mm, colors[1])):
            if len(pts):
                ax.scatter(
                    pts[:, 2],
                    pts[:, 1],
                    pts[:, 0],
                    s=2,
                    c=c,
                    marker="s",
                    linewidths=0,
                    alpha=alpha,
                    depthshade=True,
                )
    # los tres planos de corte, para ligar la vista 3D con los sliders
    z0, y0, x0 = planes
    xx, yy = np.meshgrid([0, xmax], [0, ymax])
    ax.plot_surface(xx, yy, np.full_like(xx, z0), color=MUTED, alpha=0.10)
    xx, zz = np.meshgrid([0, xmax], [0, zmax])
    ax.plot_surface(xx, np.full_like(xx, y0), zz, color=MUTED, alpha=0.10)
    yy, zz = np.meshgrid([0, ymax], [0, zmax])
    ax.plot_surface(np.full_like(yy, x0), yy, zz, color=MUTED, alpha=0.10)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_zlim(0, zmax)
    # zoom < 1 encoge la caja para que las etiquetas de los ejes no se
    # recorten: el layout no mide bien la extensión de un Axes3D
    ax.set_box_aspect((xmax, ymax, zmax), zoom=0.82)
    ax.view_init(elev=elev, azim=azim)
    for name, setter in (
        ("x (mm)", ax.set_xlabel),
        ("y (mm)", ax.set_ylabel),
        ("z (mm)", ax.set_zlabel),
    ):
        setter(name, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor(GRID)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_title("Superficie de la máscara", fontsize=10, color=INK)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
def explore_mask(
    volume,
    mask,
    spacing=SEG_SPACING_MM,
    solid_mask=None,
    window: str = DEFAULT_WINDOW,
    title: str | None = None,
    palette=None,
    figsize=(14.5, 4.4),
) -> widgets.VBox:
    """Interfaz interactiva para recorrer la máscara de un nódulo.

    Parámetros
    ----------
    volume : array o tensor (D, H, W) en HU. Se aceptan ejes de lote y
        canal al frente si miden 1, p. ej. (1, 1, 80, 80, 80).
    mask : máscara binaria con la misma forma que ``volume``.
    spacing : mm por vóxel. Un número para volúmenes isótropos (por
        defecto 0.625, el del parche de morfometría) o una tupla
        (sz, sy, sx) para un bloque crudo, p. ej. (2.0, 0.625, 0.625).
    solid_mask : máscara del componente sólido. Si no se da, se calcula
        como ``mask & (volume > -300 HU)``, el mismo corte que
        medical_criteria.
    window : ventana HU inicial; una de las claves de ``HU_WINDOWS``.
    title : título de la figura; si es None se usa uno genérico.
    palette : lista de al menos dos colores: máscara total y componente
        sólido. Por defecto los slots 1 y 2 de la paleta validada.
    figsize : tamaño de la figura de matplotlib.

    Devuelve
    --------
    ipywidgets.VBox con los controles, la figura (tres planos + vista 3D)
    y el panel de métricas. Al devolverse como último valor de una celda
    se muestra solo; asignarlo a una variable lo silencia. El atributo
    ``.fig`` es la figura de matplotlib con la vista actual, por si se
    quiere guardar con ``savefig``.

    Los sliders de corte arrancan en el centroide de la máscara y
    actualizan al soltar, no mientras se arrastran, para que la vista 3D
    no lastre la interacción.
    """
    vol = _to_numpy(volume, "volume").astype(np.float32)
    msk = _to_numpy(mask, "mask") > 0.5
    if vol.shape != msk.shape:
        raise ValueError(
            f"volume {vol.shape} y mask {msk.shape} no tienen la misma forma"
        )
    sp = _spacing3(spacing)
    _window(vol, window)  # valida el nombre antes de montar nada
    colors = list(palette or PALETTE)
    if len(colors) < 2:
        raise ValueError(
            "palette necesita al menos dos colores: máscara total y "
            "componente sólido"
        )
    if solid_mask is None:
        solid = msk & (vol > HU_SOLID)
    else:
        solid = _to_numpy(solid_mask, "solid_mask") > 0.5
        if solid.shape != msk.shape:
            raise ValueError(
                f"solid_mask {solid.shape} no tiene la forma de mask "
                f"{msk.shape}"
            )

    depth, height, width = vol.shape
    if msk.any():
        cz, cy, cx = (int(round(c)) for c in ndimage.center_of_mass(msk))
    else:
        cz, cy, cx = depth // 2, height // 2, width // 2

    sp_arr = np.asarray(sp)
    metrics = _metrics(msk, vol, solid, sp)
    surf_mm = _surface_points(msk) * sp_arr
    solid_mm = _surface_points(solid) * sp_arr
    extents = tuple(np.array([depth, height, width]) * sp_arr)

    # --- controles -----------------------------------------------------
    kw = dict(continuous_update=False, style={"description_width": "70px"})
    sl_z = widgets.IntSlider(cz, 0, depth - 1, description="Axial z", **kw)
    sl_y = widgets.IntSlider(cy, 0, height - 1, description="Coronal y", **kw)
    sl_x = widgets.IntSlider(cx, 0, width - 1, description="Sagital x", **kw)
    chk = widgets.Checkbox(True, description="Mostrar máscara", indent=False)
    op = widgets.FloatSlider(
        0.9, min=0.1, max=1.0, step=0.05, description="Opacidad", **kw
    )
    mode = widgets.ToggleButtons(options=OVERLAY_MODES, value="Contorno")
    win = widgets.Dropdown(
        options=list(HU_WINDOWS), value=window, description="Ventana"
    )
    azim = widgets.IntSlider(-60, -180, 180, description="Azimut", **kw)
    elev = widgets.IntSlider(25, -90, 90, description="Elevación", **kw)

    # --- figura persistente: se redibuja, no se recrea ------------------
    # Figure a secas, sin pyplot: no pasa por el backend (así funciona
    # también sin Tk ni kernel) y el backend inline no la muestra dos veces.
    fig = Figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3])
    ax_ax = fig.add_subplot(gs[0])
    ax_co = fig.add_subplot(gs[1])
    ax_sa = fig.add_subplot(gs[2])
    ax_3d = fig.add_subplot(gs[3], projection="3d")
    fig.suptitle(
        title or "Exploración del nódulo segmentado",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )

    def _update(z, y, x, show, alpha, mode_, win_, az, el):
        vmin, vmax = _window(vol, win_)
        common = (vmin, vmax)
        _draw_slice(
            ax_ax,
            vol[z],
            msk[z],
            solid[z],
            *common,
            sp[1] / sp[2],
            show,
            mode_,
            alpha,
            colors,
            (y, x),
            f"Axial  z = {z}",
        )
        _draw_slice(
            ax_co,
            vol[:, y, :],
            msk[:, y, :],
            solid[:, y, :],
            *common,
            sp[0] / sp[2],
            show,
            mode_,
            alpha,
            colors,
            (z, x),
            f"Coronal  y = {y}",
        )
        _draw_slice(
            ax_sa,
            vol[:, :, x],
            msk[:, :, x],
            solid[:, :, x],
            *common,
            sp[0] / sp[1],
            show,
            mode_,
            alpha,
            colors,
            (z, y),
            f"Sagital  x = {x}",
        )
        _draw_3d(
            ax_3d,
            surf_mm,
            solid_mm,
            extents,
            (z * sp[0], y * sp[1], x * sp[2]),
            az,
            el,
            show,
            alpha,
            colors,
        )
        display(fig)

    out = widgets.interactive_output(
        _update,
        dict(
            z=sl_z,
            y=sl_y,
            x=sl_x,
            show=chk,
            alpha=op,
            mode_=mode,
            win_=win,
            az=azim,
            el=elev,
        ),
    )
    controls = widgets.VBox(
        [
            widgets.HBox([sl_z, sl_y, sl_x]),
            widgets.HBox([chk, op, mode, win]),
            widgets.HBox([azim, elev]),
        ]
    )
    panel = widgets.HTML(_metrics_html(metrics, sp))
    box = widgets.VBox([controls, widgets.HBox([out, panel])])
    box.fig = fig  # la vista actual, para guardarla con box.fig.savefig(...)
    return box
