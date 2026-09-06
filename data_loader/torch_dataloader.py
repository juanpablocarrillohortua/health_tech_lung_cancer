"""
luna25_loader.py — Capa de carga de datos para LUNA25 (nodule blocks).

Basado en https://github.com/DIAGNijmegen/luna25-baseline-public/blob/main/dataloader.py

Diferencias de diseño respecto al baseline
------------------------------------------
1. SEPARACIÓN DE RESPONSABILIDADES. El `Dataset` hace SOLO I/O: lee el bloque
   .npy tal cual está en disco (64x128x128, HU crudos) y sus metadatos. No
   recorta, no reorienta, no normaliza, no aumenta. Todo eso vive en
   `PatchExtractor`, que es un `nn.Module` y corre en el device que le pongas.

2. REMUESTREO EN GPU. El baseline llama a `scipy.ndimage.affine_transform` por
   muestra dentro del worker de CPU. Aquí la transformación afín se expresa
   como un grid y se aplica con `F.grid_sample` sobre el batch completo.
   La matemática es idéntica (verificada contra scipy hasta ~1e-11, ver
   `selftest_vs_scipy`).

3. DEVICE-AGNOSTIC. `get_device()` resuelve cuda/mps/cpu y
    `Luna25Batch.to(device)` mueve el batch entero. Ningún `.cuda()`
    fijo en el código.

Convenios de coordenadas
------------------------
Los ejes del volumen se manejan en orden de array numpy, es decir (0, 1, 2) =
(D, H, W). `origin`, `spacing` y `transform` se pasan tal cual vienen del .npy
de metadatos y se combinan con la MISMA fórmula del baseline:

    voxel = inv(transform) @ (world - origin) / spacing
    world = origin + transform @ (voxel * spacing)

CoordX/Y/Z del CSV van en orden inverso al de los ejes del array, así que hay
que LEERLAS INVERTIDAS: `coord_order="zyx"`, que es el valor por defecto.
Verificado con `verify_coordinate_convention` sobre 300 anotaciones: con "zyx"
el 100% de los centros cae dentro del bloque, a 1.0 vóxel de mediana del
centro; con "xyz" solo el 11.3% cae dentro y la mediana se va a 278 vóxeles.
Leerlas mal no da error: grid_sample muestrea fuera del volumen y
padding_mode="border" devuelve un patch de bandas horizontales, con contraste
nódulo-entorno de ~11 HU en vez de ~344. Si algún día cambia el empaquetado de
los bloques, vuelve a correr `verify_coordinate_convention` antes de asumir
nada.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence  # noqa: F401

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "get_device",
    "Luna25Batch",
    "Luna25BlockDataset",
    "collate_luna25",
    "build_dataloader",
    "PatchExtractor",
    "clip_and_scale",
    "world_to_voxel",
    "voxel_to_world",
    "random_rotation_matrices",
    "random_sphere_offsets",
    "selftest_vs_scipy",
    "verify_coordinate_convention",
]


# ===========================================================================
# Device
# ===========================================================================
def get_device(prefer: str = "auto") -> torch.device:
    """Resuelve el device disponible sin hardcodear cuda."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ===========================================================================
# Batch
# ===========================================================================
@dataclass
class Luna25Batch:
    """Un batch de bloques crudos + sus metadatos geométricos.

    image       (B, 1, D, H, W) float32, unidades Hounsfield SIN normalizar
    origin      (B, 3)   origen del bloque en mm
    spacing     (B, 3)   mm por vóxel, en orden de eje del array
    transform   (B, 3, 3) matriz de dirección del bloque
    coord_world (B, 3)   coordenada del nódulo tomada del CSV
    center_vox  (B, 3)   esa misma coordenada en índices de vóxel del bloque
    label       (B,)     etiqueta de malignidad (-1 si no está en el CSV)
    """

    image: torch.Tensor | None
    origin: torch.Tensor
    spacing: torch.Tensor
    transform: torch.Tensor
    coord_world: torch.Tensor
    center_vox: torch.Tensor
    label: torch.Tensor
    annotation_id: list[str]
    patient_id: list[str]
    row_index: torch.Tensor

    def to(self, device, non_blocking: bool = True) -> "Luna25Batch":
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = (
                v.to(device, non_blocking=non_blocking)
                if torch.is_tensor(v)
                else v
            )
        return Luna25Batch(**kw)

    def pin_memory(self) -> "Luna25Batch":
        kw = {
            f.name: (
                getattr(self, f.name).pin_memory()
                if torch.is_tensor(getattr(self, f.name))
                else getattr(self, f.name)
            )
            for f in fields(self)
        }
        return Luna25Batch(**kw)

    def __len__(self) -> int:
        return len(self.annotation_id)


# ===========================================================================
# Dataset: I/O puro
# ===========================================================================
class Luna25BlockDataset(Dataset):
    """Lee los nodule blocks de LUNA25. Solo I/O, sin transformaciones.

    Parameters
    ----------
    data_dir : ruta a nodule_blocks (contiene image/ y metadata/)
    dataset  : DataFrame del CSV de LUNA25 (una fila por anotación)
    load_image : si False solo devuelve metadatos y coordenadas. Útil para
       recorrer 6.000 anotaciones en segundos cuando el análisis es geométrico.
    coord_order : "zyx" (por defecto) invierte CoordX/Y/Z antes de aplicar la
        fórmula world->voxel, que es lo correcto para los bloques de LUNA25:
        origin/spacing/transform vienen en orden de array (z, y, x) y las
        coordenadas del CSV en orden (x, y, z). "xyz" las usa tal cual y deja a
        88.7% de los centros fuera del bloque. Comprobable con
        `verify_coordinate_convention`.
    mmap : np.load con mmap_mode="r". Solo conviene si vas a leer un subconjunt
        del bloque (p.ej. unos pocos slices); para leerlo entero es más lento.
    """

    def __init__(
        self,
        data_dir: str | Path,
        dataset: pd.DataFrame,
        *,
        load_image: bool = True,
        image_subdir: str = "image",
        metadata_subdir: str = "metadata",
        id_col: str = "AnnotationID",
        label_col: str | None = "label",
        patient_col: str | None = "PatientID",
        coord_cols: Sequence[str] = ("CoordX", "CoordY", "CoordZ"),
        coord_order: str = "zyx",
        expected_shape: tuple[int, ...] | None = (64, 128, 128),
        strict_shape: bool = False,
        mmap: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dataset = dataset.reset_index(drop=True)
        self.load_image = load_image
        self.image_dir = self.data_dir / image_subdir
        self.metadata_dir = self.data_dir / metadata_subdir
        self.id_col = id_col
        self.label_col = label_col if (label_col in dataset.columns) else None
        self.patient_col = (
            patient_col if (patient_col in dataset.columns) else None
        )
        self.coord_cols = list(coord_cols)
        if coord_order not in ("xyz", "zyx"):
            raise ValueError("coord_order debe ser 'xyz' o 'zyx'")
        self.coord_order = coord_order
        self.has_coords = all(c in dataset.columns for c in self.coord_cols)
        self.expected_shape = expected_shape
        self.strict_shape = strict_shape
        self.mmap = mmap
        self.dtype = dtype

    # -- rutas ------------------------------------------------------------
    def image_path(self, annotation_id: str) -> Path:
        return self.image_dir / f"{annotation_id}.npy"

    def metadata_path(self, annotation_id: str) -> Path:
        return self.metadata_dir / f"{annotation_id}.npy"

    # -- acceso -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset.iloc[idx]
        annotation_id = str(row[self.id_col])

        meta = np.load(
            self.metadata_path(annotation_id), allow_pickle=True
        ).item()
        origin = np.asarray(meta["origin"], dtype=np.float64).reshape(3)
        spacing = np.asarray(meta["spacing"], dtype=np.float64).reshape(3)
        transform = np.asarray(meta["transform"], dtype=np.float64).reshape(
            3, 3
        )

        image = None
        if self.load_image:
            arr = np.load(
                self.image_path(annotation_id),
                mmap_mode="r" if self.mmap else None,
                allow_pickle=False,
            )
            if self.expected_shape is not None and tuple(arr.shape) != tuple(
                self.expected_shape
            ):
                msg = (
                    f"{annotation_id}: shape {tuple(arr.shape)} != "
                    f"{tuple(self.expected_shape)}"
                )
                if self.strict_shape:
                    raise ValueError(msg)
            # copia explícita a contiguo: from_numpy sobre un memmap dejaría el
            # archivo mapeado dentro del tensor
            image = torch.from_numpy(
                np.ascontiguousarray(arr, dtype=np.float32)
            )
            image = image.unsqueeze(0).to(self.dtype)  # (1, D, H, W)
            shape = np.asarray(arr.shape, dtype=np.float64)
        else:
            shape = (
                np.asarray(self.expected_shape, dtype=np.float64)
                if self.expected_shape is not None
                else np.full(3, np.nan)
            )

        if self.has_coords:
            coord = np.asarray(
                [row[c] for c in self.coord_cols], dtype=np.float64
            )
            world = coord[::-1].copy() if self.coord_order == "zyx" else coord
            center = world_to_voxel(world, origin, spacing, transform)
        else:
            world = np.full(3, np.nan)
            center = (shape - 1) / 2.0  # el nódulo está centrado en el bloque

        label = -1
        if self.label_col is not None and not pd.isna(row[self.label_col]):
            label = int(row[self.label_col])

        return {
            "image": image,
            "origin": torch.from_numpy(origin).to(self.dtype),
            "spacing": torch.from_numpy(spacing).to(self.dtype),
            "transform": torch.from_numpy(transform).to(self.dtype),
            "coord_world": torch.from_numpy(world).to(self.dtype),
            "center_vox": torch.from_numpy(np.ascontiguousarray(center)).to(
                self.dtype
            ),
            "label": torch.tensor(label, dtype=torch.long),
            "annotation_id": annotation_id,
            "patient_id": (
                str(row[self.patient_col])
                if self.patient_col is not None
                else ""
            ),
            "row_index": torch.tensor(idx, dtype=torch.long),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n={len(self)}, dir='{self.data_dir}', "
            f"load_image={self.load_image}, coord_order='{self.coord_order}')"
        )


def collate_luna25(samples: list[dict]) -> Luna25Batch:
    stack = lambda k: torch.stack([s[k] for s in samples])  # noqa: E731
    return Luna25Batch(
        image=(
            torch.stack([s["image"] for s in samples])
            if samples[0]["image"] is not None
            else None
        ),
        origin=stack("origin"),
        spacing=stack("spacing"),
        transform=stack("transform"),
        coord_world=stack("coord_world"),
        center_vox=stack("center_vox"),
        label=stack("label"),
        annotation_id=[s["annotation_id"] for s in samples],
        patient_id=[s["patient_id"] for s in samples],
        row_index=stack("row_index"),
    )


def worker_init_fn(worker_id: int) -> None:
    """Semilla distinta de numpy por worker y por época."""
    info = torch.utils.data.get_worker_info()
    np.random.seed(int(info.seed) % (2**32))


def build_dataloader(
    data_dir: str | Path,
    dataset: pd.DataFrame,
    *,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = False,
    sampler=None,
    drop_last: bool = False,
    device: torch.device | None = None,
    prefetch_factor: int | None = 4,
    **dataset_kwargs,
) -> DataLoader:
    """DataLoader listo para consumir. `pin_memory` se activa solo con CUDA."""
    ds = Luna25BlockDataset(data_dir, dataset, **dataset_kwargs)
    device = device or get_device()
    kwargs = dict(
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_luna25,
        pin_memory=(device.type == "cuda"),
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
    )
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=prefetch_factor)
    return DataLoader(ds, **kwargs)


# ===========================================================================
# Geometría
# ===========================================================================
def world_to_voxel(world, origin, spacing, transform):
    """voxel = inv(transform) @ (world - origin) / spacing. numpy o torch."""
    if torch.is_tensor(world):
        d = (world - origin).unsqueeze(-1)
        v = torch.linalg.solve(transform, d).squeeze(-1)
        return v / spacing
    world = np.asarray(world, dtype=np.float64)
    return np.linalg.solve(
        np.asarray(transform, dtype=np.float64),
        world - np.asarray(origin, dtype=np.float64),
    ) / np.asarray(spacing)


def voxel_to_world(voxel, origin, spacing, transform):
    """world = origin + transform @ (voxel * spacing). numpy o torch."""
    if torch.is_tensor(voxel):
        return origin + (transform @ (voxel * spacing).unsqueeze(-1)).squeeze(
            -1
        )
    voxel = np.asarray(voxel, dtype=np.float64)
    return np.asarray(origin, dtype=np.float64) + np.asarray(
        transform, dtype=np.float64
    ) @ (voxel * np.asarray(spacing))


def sampling_matrix(spacing, transform, out_spacing, rotation=None):
    """Matriz M tal que  in_vox = centro + M @ (idx_salida - (shape-1)/2).

    Es exactamente el `backwards_matrix` de volumeTransform del baseline,
    reescrito en forma cerrada. Acepta lotes: spacing (B,3), transform (B,3,3).
    """
    batched = spacing.ndim > 1
    s = spacing if batched else spacing.unsqueeze(0)
    W = transform if batched else transform.unsqueeze(0)
    B = s.shape[0]
    if rotation is None:
        R = torch.eye(3, dtype=s.dtype, device=s.device).expand(B, 3, 3)  # noqa: N806
    else:
        R = (  # noqa: N806
            rotation
            if rotation.ndim == 3
            else rotation.unsqueeze(0).expand(B, 3, 3)
        )

    # inv(W).T  ->  usar solve por estabilidad
    invW_T = torch.linalg.inv(W).transpose(-1, -2)  # noqa: N806
    M = R @ invW_T  # noqa: N806
    M = M / M.norm(dim=-1, keepdim=True)  # normalizar filas  # noqa: N806
    out_vs = out_spacing.to(s.dtype).to(s.device).view(1, 1, 3)  # noqa: N806
    M = (1.0 / s).unsqueeze(-1) * M.transpose(-1, -2) * out_vs  # noqa: N806
    return M if batched else M.squeeze(0)  # noqa: N806


# ===========================================================================
# Extracción de patches en GPU
# ===========================================================================
class PatchExtractor(nn.Module):
    """Remuestrea un patch isotrópico centrado en un punto, sobre el batch.

    Equivalente a `extract_patch` del baseline pero vectorizado y sobre el
    device del módulo. Un patch de `size_mm` mm se remuestrea a `size_px`
    vóxeles por eje, es decir a size_mm/size_px mm/vóxel.

    padding_mode="zeros" replica el baseline (cval=0.0 de scipy). Ojo: 0 HU es
    agua, no aire. Si rotas mucho y el patch se sale del bloque, "border" suele
    ser más razonable para TC.
    """

    def __init__(
        self,
        size_px: int = 64,
        size_mm: float = 50.0,
        mode: str = "3D",
        padding_mode: str = "zeros",
        interpolation: str = "bilinear",  # trilineal para entrada 5D
        repeat_channels_2d: int = 3,
    ) -> None:
        super().__init__()
        if mode not in ("2D", "3D"):
            raise ValueError("mode debe ser '2D' o '3D'")
        self.size_px = size_px
        self.size_mm = float(size_mm)
        self.mode = mode
        self.padding_mode = padding_mode
        self.interpolation = interpolation
        self.repeat_channels_2d = repeat_channels_2d

        out_shape = (1, size_px, size_px) if mode == "2D" else (size_px,) * 3
        self.out_shape = out_shape
        vs = self.size_mm / self.size_px
        self.register_buffer(
            "out_spacing", torch.tensor([vs, vs, vs]), persistent=False
        )

        idx = torch.stack(
            torch.meshgrid(
                *[torch.arange(n, dtype=torch.float32) for n in out_shape],
                indexing="ij",
            ),
            dim=-1,
        )
        idx = idx - (torch.tensor(out_shape, dtype=torch.float32) - 1) / 2.0
        self.register_buffer("base_idx", idx, persistent=False)  # (Do,Ho,Wo,3)

    def forward(
        self,
        image: torch.Tensor,  # (B, 1, D, H, W)
        spacing: torch.Tensor,  # (B, 3)
        transform: torch.Tensor,  # (B, 3, 3)
        center_vox: torch.Tensor,  # (B, 3)
        rotation: torch.Tensor | None = None,  # (B, 3, 3)
        translation_vox: torch.Tensor | None = None,  # (B, 3)
    ) -> torch.Tensor:
        if image.ndim != 5:
            raise ValueError(
                f"se esperaba (B,1,D,H,W), llegó {tuple(image.shape)}"
            )
        B = image.shape[0]  # noqa: N806
        dt = image.dtype

        M = sampling_matrix(  # noqa: N806
            spacing.to(dt),
            transform.to(dt),
            self.out_spacing,
            rotation.to(dt) if rotation is not None else None,
        )

        center = center_vox.to(dt)
        if translation_vox is not None:
            center = center + translation_vox.to(dt)

        offs = torch.einsum("bij,dhwj->bdhwi", M, self.base_idx.to(dt))
        in_vox = center.view(B, 1, 1, 1, 3) + offs  # orden (z,y,x)

        size = torch.tensor(image.shape[-3:], dtype=dt, device=image.device)
        grid = 2.0 * in_vox / (size - 1) - 1.0
        grid = grid.flip(-1)  # grid_sample espera (x, y, z)

        patch = F.grid_sample(
            image,
            grid,
            mode=self.interpolation,
            padding_mode=self.padding_mode,
            align_corners=True,
        )

        if self.mode == "2D":
            patch = patch[:, :, 0]  # (B,1,H,W)
            if self.repeat_channels_2d > 1:
                patch = patch.repeat(1, self.repeat_channels_2d, 1, 1)
        return patch

    def extra_repr(self) -> str:
        return (
            f"size_px={self.size_px}, size_mm={self.size_mm}, mode={self.mode}, "  # noqa: E501
            f"out_spacing={self.size_mm / self.size_px:.4f}mm, "
            f"padding_mode={self.padding_mode}"
        )


def clip_and_scale(
    x: torch.Tensor, min_hu: float = -1000.0, max_hu: float = 400.0
) -> torch.Tensor:
    """
    Ventana pulmonar del baseline, a [0, 1]. Es una decisión de análisis,
    por eso está fuera del Dataset.

    Util con el ventaneo clinico, se debe cambiar segun lo que se halle,
    auqnue es el valor de la linea base.
    """
    return ((x - min_hu) / (max_hu - min_hu)).clamp_(0.0, 1.0)


# ===========================================================================
# Aumento (opcional, se aplica en la etapa de extracción, no en el loader)
# ===========================================================================
def random_rotation_matrices(
    n: int,
    degrees=((-20, 20), (-20, 20), (-20, 20)),
    device=None,
    dtype=torch.float32,
    generator=None,
) -> torch.Tensor:
    """Rotaciones aleatorias Rx@Ry@Rz. `degrees` en orden (z, y, x) como el
    baseline: ((zmin,zmax),(ymin,ymax),(xmin,xmax))."""
    (zmin, zmax), (ymin, ymax), (xmin, xmax) = degrees

    def ang(lo, hi):
        u = torch.rand(n, device=device, dtype=dtype, generator=generator)
        return (u * (hi - lo) + lo) * torch.pi / 180.0

    ax, ay, az = ang(xmin, xmax), ang(ymin, ymax), ang(zmin, zmax)
    o, i = (
        torch.zeros(n, device=device, dtype=dtype),
        torch.ones(n, device=device, dtype=dtype),
    )

    def mat(rows):
        return torch.stack([torch.stack(r, -1) for r in rows], -2)

    Rx = mat([(i, o, o), (o, ax.cos(), -ax.sin()), (o, ax.sin(), ax.cos())])  # noqa: N806
    Ry = mat([(ay.cos(), o, ay.sin()), (o, i, o), (-ay.sin(), o, ay.cos())])  # noqa: N806
    Rz = mat([(az.cos(), -az.sin(), o), (az.sin(), az.cos(), o), (o, o, i)])  # noqa: N806
    return Rx @ Ry @ Rz


def random_sphere_offsets(
    spacing: torch.Tensor, radius_mm: float = 2.5, generator=None
) -> torch.Tensor:
    """Desplazamientos aleatorios dentro de una esfera de `radius_mm`,
    devueltos en vóxeles. Réplica del aumento de traslación del baseline."""
    n = spacing.shape[0]
    v = torch.randn(
        n, 3, device=spacing.device, dtype=spacing.dtype, generator=generator
    )
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    r = (
        torch.rand(
            n,
            1,
            device=spacing.device,
            dtype=spacing.dtype,
            generator=generator,
        )
        * radius_mm
    )
    return v * r / spacing


# ===========================================================================
# Verificación de la convención de coordenadas
# ===========================================================================
def verify_coordinate_convention(
    data_dir,
    dataset,
    n: int = 300,
    expected_shape=(64, 128, 128),
    verbose: bool = True,
) -> dict:
    """¿CoordX/Y/Z hay que leerlas tal cual ("xyz") o invertidas ("zyx")?

    Los bloques vienen recortados alrededor del nódulo, así que la convención
    correcta es la que deja el centro anotado en el centro del bloque.
    Se prueban las dos y se comparan contra `(shape - 1) / 2`.

    Solo lee los .npy de metadatos, no las imágenes, así que recorrer unos
    cientos de anotaciones tarda segundos.

    Devuelve {convención: {"inside_frac", "median_dist_vox", "mean_center"}} y,
    en `["best"]`, la convención ganadora. El criterio discrimina de forma
    tajante: para LUNA25 da 100% dentro y 1.0 vóxel para "zyx" frente a 11.3% y
    278 vóxeles para "xyz".
    """
    meta_dir = Path(data_dir) / "metadata"
    df = dataset.head(n)
    shape = np.asarray(expected_shape, dtype=np.float64)
    center = (shape - 1) / 2.0

    out: dict = {}
    for order in ("xyz", "zyx"):
        vox = []
        for _, row in df.iterrows():
            m = np.load(
                meta_dir / f"{row['AnnotationID']}.npy", allow_pickle=True
            ).item()
            coord = np.asarray(
                [row["CoordX"], row["CoordY"], row["CoordZ"]], dtype=np.float64
            )
            world = coord[::-1] if order == "zyx" else coord
            vox.append(
                world_to_voxel(
                    world,
                    np.asarray(m["origin"]).reshape(3),
                    np.asarray(m["spacing"]).reshape(3),
                    np.asarray(m["transform"]).reshape(3, 3),
                )
            )
        v = np.asarray(vox)
        out[order] = {
            "inside_frac": float(((v >= 0) & (v < shape)).all(1).mean()),
            "median_dist_vox": float(
                np.median(np.linalg.norm(v - center, axis=1))
            ),
            "mean_center": v.mean(0),
        }

    out["best"] = max(("xyz", "zyx"), key=lambda k: out[k]["inside_frac"])
    if verbose:
        print(f"centro esperado del bloque: {center}")
        for order in ("xyz", "zyx"):
            r = out[order]
            print(
                f"  coord_order={order!r}: dentro={r['inside_frac']:.1%}  "
                f"dist. mediana={r['median_dist_vox']:.1f} vox  "
                f"centro medio={np.round(r['mean_center'], 1)}"
            )
        print(f"-> usa coord_order={out['best']!r}")
    return out


# ===========================================================================
# Autotest contra la implementación original
# ===========================================================================
def selftest_vs_scipy(tol: float = 1e-3, verbose: bool = True) -> bool:
    """Compara PatchExtractor contra dataloader.extract_patch del baseline.

    Requiere tener el repo baseline importable. Corre esto una vez en tu
    entorno: si pasa, la ruta GPU es intercambiable con la de scipy.
    """
    import scipy.ndimage as ndi  # noqa: F401
    from dataloader import extract_patch  # baseline

    rng = np.random.default_rng(0)
    vol = rng.normal(0, 300, (64, 128, 128)).astype(np.float32)
    vol[28:36, 60:70, 60:70] += 900.0

    cases = [
        (np.array([1.0, 1.0, 1.0]), np.eye(3)),
        (np.array([1.5, 0.7, 0.7]), np.eye(3)),
        (np.array([1.2, 0.8, 0.8]), np.diag([-1.0, 1.0, -1.0])),
    ]
    dev = get_device()
    ex = PatchExtractor(size_px=64, size_mm=50.0, mode="3D").to(dev)
    ok = True
    for spacing, W in cases:  # noqa: N806
        center = np.array(vol.shape) // 2
        ref = extract_patch(
            CTData=vol,
            coord=tuple(center),
            srcVoxelOrigin=np.zeros(3),
            srcWorldMatrix=W,
            srcVoxelSpacing=spacing,
            output_shape=(64, 64, 64),
            voxel_spacing=(50.0 / 64,) * 3,
            rotations=None,
            translations=None,
            coord_space_world=False,
            mode="3D",
        )[0]

        got = (
            ex(
                torch.from_numpy(vol)[None, None].to(dev),
                torch.tensor(spacing, dtype=torch.float32)[None].to(dev),
                torch.tensor(W, dtype=torch.float32)[None].to(dev),
                torch.tensor(center, dtype=torch.float32)[None].to(dev),
            )[0, 0]
            .cpu()
            .numpy()
        )

        d = np.abs(ref - got).max()
        ok &= d < tol
        if verbose:
            print(
                f"spacing={spacing} det(W)={np.linalg.det(W):+.0f}  max|dif|={d:.2e}"  # noqa: E501
            )
    if verbose:
        print("OK" if ok else "FALLO")
    return ok


if __name__ == "__main__":
    print("device:", get_device())
    print(PatchExtractor(mode="3D"))
    print(PatchExtractor(mode="2D"))
