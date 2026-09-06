"""
nodule_morphometry.py — Cuantifica tamaño, irregularidad del borde y
consistencia de cada nódulo de LUNA25. Una fila por AnnotationID.

Todas las columnas son métricas numéricas crudas. El script NO clasifica,
NO umbraliza y NO devuelve categorías: los cortes clínicos (20 mm, 30 mm, 5 mm,
sólido/subsólido) se aplican después sobre este DataFrame.

    python nodule_morphometry.py            # procesa el CSV completo
    python nodule_morphometry.py selftest   # fantasmas con verdad conocida

ADVERTENCIA IMPORTANTE
----------------------
LUNA25 no incluye máscara de segmentación del nódulo, así que hay que delimitarlo.
Aquí se hace por umbral adaptativo al parénquima local + crecimiento de región
acotado. Es un proxy razonable, no una segmentación de referencia. Consecuencias:

  - Nódulos adheridos a vaso o pleura se fugan hacia esa estructura. Usa las
    columnas `seg_leak_frac` y `seg_converged` para filtrar o ponderar.
  - Nódulos en vidrio esmerilado puro dependen fuertemente del umbral. El offset
    por defecto (+200 HU sobre el fondo pulmonar) los captura en los fantasmas,
    pero valida visualmente una muestra antes de sacar conclusiones.
  - Nada de esto sustituye una segmentación anotada por radiólogo.

Validación realizada (ver validate_metrics.py / validate_seg.py):
  esfera r=5mm  -> d_eq 10.05 (real 10.00), esfericidad 1.016, superficie 0.7% error
  elipsoide 12/6/6 -> ejes 24.06/11.98/11.98 (real 24/12/12)
  fantasma parcialmente sólido r=8 con núcleo r=4 -> d_total 15.98, d_sólido 7.98,
  fracción sólida 0.124 (real 0.125)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_loader.torch_dataloader import PatchExtractor, build_dataloader, get_device

try:
    from config import config
    CSV_PATH, DATA_DIR = config.LUNA25_LABELS, config.NODULE_BLOCKS_ROOT
except Exception:
    raise RuntimeError("No se pudo importar config.py. Copia config.py.example a config.py y edítalo con tus rutas locales.")

BLOCK_SHAPE = (64, 128, 128)

# --- Parámetros de segmentación -------------------------------------------
SEG_SIZE_MM = 50.0     # campo de visión del patch usado para morfometría
SEG_SIZE_PX = 80       # -> 0.625 mm/vóxel isotrópico
R_MAX_MM = 20.0        # radio máximo del nódulo: acota la fuga por vasos
SEED_R_MM = 1.5        # semilla esférica en el centro
BG_SHELL_MM = (18.0, 24.0)   # corona donde se mide el parénquima sano
THR_OFFSET_HU = 200.0        # umbral = fondo + offset
THR_LIMITS_HU = (-700.0, -450.0)

# --- Cortes de densidad (HU) ----------------------------------------------
HU_AIR = -750.0        # por debajo: aire / enfisema
HU_SOLID = -300.0      # por encima: componente sólido
HU_CALC = 200.0        # por encima: calcificación


# ===========================================================================
# Morfología binaria en GPU
# ===========================================================================
def dilate(m: torch.Tensor, iters: int = 1) -> torch.Tensor:
    """Dilatación con elemento estructurante 3x3x3 (conectividad 26)."""
    for _ in range(iters):
        m = F.max_pool3d(m, kernel_size=3, stride=1, padding=1)
    return m


def erode(m: torch.Tensor, iters: int = 1) -> torch.Tensor:
    """Erosión. max_pool rellena con -inf, así que el exterior cuenta como
    lleno: equivale a border_value=1 en scipy."""
    for _ in range(iters):
        m = -F.max_pool3d(-m, kernel_size=3, stride=1, padding=1)
    return m


def grow_region(seed, allowed, max_iter=96, check_every=8):
    """Crecimiento de región desde la semilla, restringido a `allowed`.

    Equivale a quedarse con la componente conexa de `allowed` que contiene la
    semilla (verificado contra scipy.ndimage.label). Devuelve también si todos
    los elementos del lote convergieron.
    """
    m = seed * allowed
    converged = False
    for i in range(max_iter):
        nxt = dilate(m) * allowed
        if (i + 1) % check_every == 0 or i == max_iter - 1:
            if torch.equal(nxt, m):
                converged = True
                break
        m = nxt
    else:
        converged = torch.equal(dilate(m) * allowed, m)
    return m, converged


def gaussian_blur3d(x, sigma_vox=1.0):
    """Suavizado gaussiano separable (tres conv1d encadenadas)."""
    r = max(1, int(round(3 * sigma_vox)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma_vox) ** 2)
    k = k / k.sum()
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = -1
        pad = [0] * 6
        pad[2 * (2 - dim)] = pad[2 * (2 - dim) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k.view(shape))
    return x


def surface_area_mm2(mask, vs, sigma_vox=1.0):
    """Superficie por la fórmula del coárea: S = integral de |grad f| sobre un
    indicador suavizado. Es isotrópica; contar caras expuestas sobrestimaría
    un 55% en superficies curvas (medido sobre esferas sintéticas).
    """
    f = gaussian_blur3d(mask, sigma_vox)
    g2 = 0.0
    for dim in range(3):
        pad = [0] * 6
        pad[2 * (2 - dim)] = pad[2 * (2 - dim) + 1] = 1
        fp = F.pad(f, pad, mode="replicate")
        if dim == 0:
            g = fp[:, :, 2:] - fp[:, :, :-2]
        elif dim == 1:
            g = fp[:, :, :, 2:] - fp[:, :, :, :-2]
        else:
            g = fp[:, :, :, :, 2:] - fp[:, :, :, :, :-2]
        g2 = g2 + (g / (2 * vs)) ** 2
    return g2.sqrt().flatten(1).sum(1) * vs**3


# ===========================================================================
# Utilidades estadísticas con máscara
# ===========================================================================
def masked_quantiles(x, mask, qs):
    """Cuantiles exactos por muestra sobre los vóxeles enmascarados.
    x, mask: (B, N). Devuelve (B, len(qs))."""
    big = torch.finfo(x.dtype).max
    xs, _ = torch.sort(torch.where(mask > 0, x, torch.full_like(x, big)), dim=1)
    n = mask.sum(1)
    out = []
    for q in qs:
        k = ((n - 1).clamp_min(0) * q).round().long()
        v = xs.gather(1, k[:, None]).squeeze(1)
        out.append(torch.where(n > 0, v, torch.full_like(v, float("nan"))))
    return torch.stack(out, 1)


def masked_mean(x, mask):
    n = mask.sum(1)
    return torch.where(n > 0, (x * mask).sum(1) / n.clamp_min(1), torch.nan)


def masked_std(x, mask):
    mu = masked_mean(x, mask)
    return masked_mean((x - mu[:, None]) ** 2, mask).sqrt()


# ===========================================================================
# Segmentación
# ===========================================================================
def lung_background_hu(hu, rr, shell=BG_SHELL_MM):
    """Densidad del parénquima sano alrededor de la lesión.

    Se toma la mediana de la corona [18, 24] mm restringida a vóxeles que aún
    son aire pulmonar (< -400 HU), para no contaminarse con pared torácica,
    vasos grandes o el propio nódulo si es grande.
    """
    m = ((rr > shell[0]) & (rr <= shell[1]) & (hu < -400.0)).float()
    bg = masked_quantiles(hu.flatten(1), m.flatten(1), [0.5])[:, 0]
    return torch.nan_to_num(bg, nan=-850.0)


def segment_nodule(hu, rr, vs):
    """Devuelve (máscara total, máscara sólida, fondo, umbral, QC).

    El umbral se ancla al parénquima local en vez de ser fijo, porque la
    densidad del pulmón varía entre pacientes y entre reconstrucciones.
    """
    B = hu.shape[0]
    bg = lung_background_hu(hu, rr)
    thr = (bg + THR_OFFSET_HU).clamp(*THR_LIMITS_HU)

    hu5 = hu.view(B, 1, SEG_SIZE_PX, SEG_SIZE_PX, SEG_SIZE_PX)
    rr5 = rr.view(1, 1, SEG_SIZE_PX, SEG_SIZE_PX, SEG_SIZE_PX)
    allowed = ((hu5 > thr.view(B, 1, 1, 1, 1)) & (rr5 <= R_MAX_MM)).float()

    seed = (rr5 <= SEED_R_MM).float().expand(B, -1, -1, -1, -1) * allowed
    # nódulo muy tenue o mal centrado: forzar la semilla central
    fallback = (rr5 <= SEED_R_MM).float().expand(B, -1, -1, -1, -1)
    empty = seed.flatten(1).sum(1) == 0
    seed = torch.where(empty.view(B, 1, 1, 1, 1), fallback, seed)

    total, converged = grow_region(seed, allowed,
                                   max_iter=int(3 * R_MAX_MM / vs))
    solid = total * (hu5 > HU_SOLID).float()

    # ¿la máscara toca el límite de r_max? indicio de fuga a vaso o pleura
    at_limit = (rr5 > R_MAX_MM - 2 * vs).float()
    leak = ((total * at_limit).flatten(1).sum(1)
            / total.flatten(1).sum(1).clamp_min(1))
    return total, solid, bg, thr, converged, leak, empty


# ===========================================================================
# 1. TAMAÑO
# ===========================================================================
def size_metrics(mask, coords_mm, pp, vs):
    """Volumen y diámetros.

    diam_equiv: diámetro de la esfera de igual volumen.
    diam_long / short: ejes del elipsoide uniforme equivalente, obtenidos del
        tensor de inercia. Para densidad uniforme la varianza a lo largo de un
        semieje a vale a^2/5, de ahí a = sqrt(5*lambda). Verificado sobre un
        elipsoide 12/6/6 mm -> 24.06/11.98/11.98.
    diam_mean_axes: promedio de eje mayor y menor, lo más cercano al diámetro
        medio que usan Fleischner y Lung-RADS.
    """
    w = mask.flatten(1).double()
    n = w.sum(1)
    V = n * vs**3

    c = (w @ coords_mm.double()) / n.clamp_min(1)[:, None]         # centroide
    M2 = (w @ pp).view(-1, 3, 3) / n.clamp_min(1)[:, None, None]
    cov = M2 - c[:, :, None] * c[:, None, :]
    lam = torch.linalg.eigvalsh(cov).clamp_min(0).flip(-1)          # desc
    diam = 2 * (5 * lam).sqrt()

    out = {
        "volume_mm3": V,
        "diam_equiv_mm": (6 * V / torch.pi) ** (1 / 3),
        "diam_long_mm": diam[:, 0],
        "diam_mid_mm": diam[:, 1],
        "diam_short_mm": diam[:, 2],
        "diam_mean_axes_mm": (diam[:, 0] + diam[:, 2]) / 2,
    }
    return out, c.float(), n.float()


# ===========================================================================
# 2. IRREGULARIDAD DEL BORDE
# ===========================================================================
def margin_metrics(mask, coords_mm, centroid, vs, size_dict):
    """Cuatro familias complementarias, todas crudas.

    sphericity: 1 para una esfera, baja al alejarse de ella. Combina volumen y
        superficie, así que penaliza tanto el alargamiento como la espiculación.
        Medido: esfera 1.016, elipsoide 0.935, espiculada 0.766.
    elongation: eje menor / eje mayor. Sirve para separar "irregular" de
        simplemente "alargado", que es el principal factor de confusión de
        radial_roughness.
    radial_roughness: dispersión relativa del radio de los vóxeles de borde
        respecto al centroide. Medido: esfera 0.034, elipsoide 0.229,
        espiculada 0.240. OJO: se confunde con la elongación, interprétala
        siempre junto a `elongation`.
    opening_loss_Xmm: fracción de volumen que no sobrevive a una apertura
        morfológica de radio X. Las espículas son finas y desaparecen; el
        cuerpo del nódulo no. Medido a 1.25 mm: esfera r=5 0.110, esfera r=10
        0.014, espiculada 0.325, muy espiculada 0.461. Depende del tamaño: se
        dan tres radios y `diam_equiv_mm` para que normalices aguas abajo.
    closing_solidity: V / V_cerrado, aproximación a la solidez del casco
        convexo. Medido: esfera 1.000, lobulada 0.976, espiculada 0.949.
    """
    B = mask.shape[0]
    vol = mask.flatten(1).sum(1)
    S = surface_area_mm2(mask, vs)
    V = size_dict["volume_mm3"].float()

    out = {
        "surface_mm2": S,
        "sphericity": torch.pi ** (1 / 3) * (6 * V) ** (2 / 3) / S.clamp_min(1e-6),
        "elongation": size_dict["diam_short_mm"].float()
                      / size_dict["diam_long_mm"].float().clamp_min(1e-6),
    }

    # radio de los vóxeles de borde respecto al centroide
    border = (mask - erode(mask)).clamp_min(0).flatten(1)
    d = coords_mm[None] - centroid[:, None]
    r = d.norm(dim=-1)
    r_mu, r_sd = masked_mean(r, border), masked_std(r, border)
    out["radial_roughness"] = r_sd / r_mu.clamp_min(1e-6)
    rq = masked_quantiles(r, border, [0.1, 0.9])
    out["radial_p10_mm"], out["radial_p90_mm"] = rq[:, 0], rq[:, 1]

    for mm in (1.0, 2.0, 3.0):
        it = max(1, int(round(mm / vs)))
        kept = dilate(erode(mask, it), it).flatten(1).sum(1)
        out[f"opening_loss_{mm:g}mm"] = 1.0 - kept / vol.clamp_min(1)

    it = max(1, int(round(2.0 / vs)))
    closed = erode(dilate(mask, it), it).flatten(1).sum(1)
    out["closing_solidity_2mm"] = vol / closed.clamp_min(1)
    return out


# ===========================================================================
# 3. CONSISTENCIA
# ===========================================================================
def density_metrics(hu, mask_total, mask_solid, bg, vs):
    """Distribución de densidad dentro de la lesión.

    solid_fraction distingue el eje sólido / parcialmente sólido / vidrio
    esmerilado: 1.0 es totalmente sólido, 0.0 vidrio esmerilado puro, y los
    valores intermedios son subsólidos. Validado sobre un fantasma con núcleo
    sólido de r=4 dentro de vidrio esmerilado de r=8: 0.124 frente a 0.125 real.

    Las fracciones por ventana son disjuntas y suman 1 dentro de la máscara.
    frac_calcified separa la calcificación, que es el patrón benigno clásico.
    """
    m = mask_total.flatten(1)
    n = m.sum(1)
    x = hu.flatten(1)

    q = masked_quantiles(x, m, [0.1, 0.5, 0.9])
    out = {
        "hu_mean": masked_mean(x, m),
        "hu_std": masked_std(x, m),
        "hu_p10": q[:, 0], "hu_p50": q[:, 1], "hu_p90": q[:, 2],
        "hu_range_p10_p90": q[:, 2] - q[:, 0],
        "hu_max": torch.where(n > 0, (x * m + (-2000) * (1 - m)).max(1).values, torch.nan),
        "frac_air": masked_mean((x < HU_AIR).float(), m),
        "frac_ggo": masked_mean(((x >= HU_AIR) & (x < HU_SOLID)).float(), m),
        "frac_soft": masked_mean(((x >= HU_SOLID) & (x < HU_CALC)).float(), m),
        "frac_calcified": masked_mean((x >= HU_CALC).float(), m),
        "solid_fraction": mask_solid.flatten(1).sum(1) / n.clamp_min(1),
        "volume_solid_mm3": mask_solid.flatten(1).sum(1) * vs**3,
        "contrast_hu": masked_mean(x, m) - bg,
    }
    out["diam_equiv_solid_mm"] = (6 * out["volume_solid_mm3"] / torch.pi) ** (1 / 3)
    return out


# ===========================================================================
# Bucle principal
# ===========================================================================
@torch.no_grad()
def nodule_morphometry(csv_path=CSV_PATH, data_dir=DATA_DIR, coord_order="zyx",
                       batch_size=16, num_workers=8, limit=None,
                       out_csv="nodule_morphometry.csv"):
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    device = get_device()
    vs = SEG_SIZE_MM / SEG_SIZE_PX
    print(f"device={device}  n={len(df)}  patch={SEG_SIZE_PX}^3 a {vs:.3f} mm/vox")

    loader = build_dataloader(
        data_dir, df, batch_size=batch_size, num_workers=num_workers,
        device=device, coord_order=coord_order, expected_shape=BLOCK_SHAPE,
    )
    extractor = PatchExtractor(size_px=SEG_SIZE_PX, size_mm=SEG_SIZE_MM,
                               mode="3D", padding_mode="border").to(device)

    # rejillas fijas, se construyen una sola vez
    ax = (torch.arange(SEG_SIZE_PX, device=device, dtype=torch.float32)
          - (SEG_SIZE_PX - 1) / 2) * vs
    grid = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1)   # (D,H,W,3)
    coords_mm = grid.reshape(-1, 3)                                     # (N,3) mm
    rr = grid.norm(dim=-1)                                              # (D,H,W)
    pp = (coords_mm[:, :, None] * coords_mm[:, None, :]).reshape(-1, 9).double()

    rows, t0, seen = [], time.time(), 0
    pbar = tqdm(loader, total=len(loader), desc="morfometría", unit="batch",
                dynamic_ncols=True)
    for batch in pbar:
        batch = batch.to(device)
        patch = extractor(batch.image, batch.spacing, batch.transform,
                          batch.center_vox)                   # (B,1,80,80,80) HU
        hu = patch[:, 0]

        total, solid, bg, thr, converged, leak, empty = segment_nodule(hu, rr, vs)

        size, centroid, n_vox = size_metrics(total, coords_mm, pp, vs)
        margin = margin_metrics(total, coords_mm, centroid, vs, size)
        density = density_metrics(hu, total, solid, bg, vs)

        rec = {
            "AnnotationID": batch.annotation_id,
            "PatientID": batch.patient_id,
            "label": batch.label.cpu().numpy(),
            # control de calidad de la segmentación
            "bg_lung_hu": bg.cpu().numpy(),
            "seg_threshold_hu": thr.cpu().numpy(),
            "seg_n_vox": n_vox.cpu().numpy(),
            "seg_leak_frac": leak.cpu().numpy(),
            "seg_seed_forced": empty.cpu().numpy().astype(np.uint8),
            "seg_converged": np.full(len(batch), int(converged), np.uint8),
        }
        for group in (size, margin, density):
            for k, v in group.items():
                rec[k] = v.float().cpu().numpy()

        rows.append(pd.DataFrame(rec))
        seen += len(batch)
        pbar.set_postfix(nodulos=seen, refresh=False)
        if seen % (batch_size * 20) == 0:
            el = time.time() - t0
            tqdm.write(f"  {seen}/{len(df)}  {el:.0f}s  ({seen/el:.1f} nódulos/s)")
    pbar.close()

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(f"\n{len(out)} filas, {out.shape[1]} columnas -> {out_csv}")
    print(f"segmentaciones con fuga (>5% en el límite): "
          f"{(out.seg_leak_frac > 0.05).mean():.1%}")
    print(out[["diam_equiv_mm", "diam_mean_axes_mm", "sphericity",
               "radial_roughness", "opening_loss_2mm", "solid_fraction",
               "frac_calcified"]].describe().round(3))
    return out


# ===========================================================================
# Autotest con fantasmas
# ===========================================================================
@torch.no_grad()
def selftest():
    """Reconstruye en torch los fantasmas validados en numpy y comprueba que la
    transcripción a GPU da los mismos números."""
    device = get_device()
    vs = SEG_SIZE_MM / SEG_SIZE_PX
    ax = (torch.arange(SEG_SIZE_PX, device=device, dtype=torch.float32)
          - (SEG_SIZE_PX - 1) / 2) * vs
    grid = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1)
    coords_mm, rr = grid.reshape(-1, 3), grid.norm(dim=-1)
    pp = (coords_mm[:, :, None] * coords_mm[:, None, :]).reshape(-1, 9).double()

    g = torch.Generator(device="cpu").manual_seed(0)
    cases = {}
    for name, build, truth in [
        ("sólido r=6", [(6.0, 30.0)], (12.0, 1.000)),
        ("sólido r=12", [(12.0, 30.0)], (24.0, 1.000)),
        ("vidrio esmerilado r=7", [(7.0, -600.0)], (14.0, 0.000)),
        ("parcialmente sólido r=8", [(8.0, -600.0), (4.0, 30.0)], (16.0, 0.125)),
        ("calcificado r=4", [(4.0, 450.0)], (8.0, 1.000)),
    ]:
        img = (torch.full_like(rr, -850.0)
               + torch.randn(rr.shape, generator=g).to(device) * 25.0)
        for r, hu_val in build:
            img = torch.where(rr <= r, torch.full_like(img, hu_val), img)
        cases[name] = (img, truth)

    hu = torch.stack([v[0] for v in cases.values()])
    total, solid, bg, thr, conv, leak, empty = segment_nodule(hu, rr, vs)
    size, centroid, _ = size_metrics(total, coords_mm, pp, vs)
    margin = margin_metrics(total, coords_mm, centroid, vs, size)
    dens = density_metrics(hu, total, solid, bg, vs)

    print(f"{'fantasma':26s} {'d_eq':>7s} {'real':>6s} {'f_sól':>7s} {'real':>6s} "
          f"{'esfer.':>7s} {'rugos.':>7s}")
    ok = True
    for i, (name, (_, (d_t, f_t))) in enumerate(cases.items()):
        d = size["diam_equiv_mm"][i].item()
        f = dens["solid_fraction"][i].item()
        ok &= abs(d - d_t) < 0.5 and abs(f - f_t) < 0.05
        print(f"{name:26s} {d:7.2f} {d_t:6.1f} {f:7.3f} {f_t:6.3f} "
              f"{margin['sphericity'][i]:7.3f} {margin['radial_roughness'][i]:7.3f}")
    print("\nOK" if ok else "\nFALLO: revisa la transcripción a torch")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        nodule_morphometry()