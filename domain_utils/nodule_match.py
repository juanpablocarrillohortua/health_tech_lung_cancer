"""
Emparejamiento geométrico entre los nódulos de LUNA25 y las anomalías
torácicas de NLST (tabla ctab).

LUNA25 tiene una fila por nódulo; ctab tiene una fila por anomalía. Cruzar
solo por (pid, study_yr) y quedarse con una anomalía por paciente-año hace
que todos los nódulos de ese año reciban los mismos descriptores, que casi
nunca son los del nódulo en cuestión. Aquí cada nódulo recibe su propia
anomalía mediante una asignación uno a uno.

Señales usadas, en orden de confianza:

1. Lateralidad (restricción dura). ``sct_epi_loc`` codifica el lóbulo:
   1-3 pulmón derecho, 4-6 izquierdo. En LUNA25 el signo de ``CoordX``
   da el lado (LPS: +X es el lado izquierdo del paciente). Medido contra
   los pares sin ambigüedad, este criterio acierta el 99.7%.
2. Tipo de anomalía (restricción dura). Solo ``sct_ab_desc == 51``
   (nódulo o masa no calcificada >= 4 mm) es un nódulo; el resto de
   códigos son otros hallazgos y ni siquiera tienen diámetro definido.
3. Altura anatómica (desempate). ``sct_slice_num`` crece hacia abajo
   (Spearman -0.42 contra la altura del lóbulo) mientras que ``CoordZ``
   crece hacia arriba, así que ambos ordenan el eje vertical en sentidos
   opuestos. ``CoordZ`` NO es comparable entre pacientes -- cada serie
   trae su propio origen -- pero sí dentro de un mismo paciente-año, que
   es donde se usa.

Limitación: no existe una verdad de terreno por nódulo. Los puntos 1 y 2
están validados; el punto 3 es una heurística razonada que no se puede
verificar de forma directa, porque los únicos pares con respuesta cierta
son los de un solo nódulo, donde el desempate nunca llega a actuar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

#: Lado del pulmón implicado por cada valor de ``sct_epi_loc``.
SIDE_BY_LOBE = {1: "R", 2: "R", 3: "R", 4: "L", 5: "L", 6: "L"}

#: Único código de ``sct_ab_desc`` que corresponde a un nódulo/masa >= 4 mm.
NODULE_AB_DESC = 51

#: Centinela de ``sct_slice_num`` para "sin dato".
MISSING_SLICE = 999

#: Costo de un emparejamiento prohibido por lateralidad. Finito para que
#: linear_sum_assignment siempre encuentre solución, pero mayor que
#: cualquier costo legítimo (que vive en [0, 1]).
FORBIDDEN = 1e6

#: Recargo para una anomalía cuyo lóbulo no determina el lado (códigos 7,
#: 8 o ausente). No se prohíbe -- "lado desconocido" no es "lado
#: equivocado" -- pero al superar el costo máximo legítimo (1.0) cualquier
#: candidata del lado correcto se prefiere frente a ella.
UNKNOWN_SIDE_PENALTY = 1.5

#: Columnas de ctab que se trasladan al nódulo emparejado.
CTAB_COLS = [
    "sct_ab_desc",
    "sct_ab_num",
    "sct_epi_loc",
    "sct_long_dia",
    "sct_perp_dia",
    "sct_margins",
    "sct_pre_att",
    "sct_slice_num",
    "sct_found_after_comp",
]


def _side_from_x(coord_x: pd.Series) -> pd.Series:
    """Lado del pulmón según el signo de CoordX (LPS: +X = izquierda)."""
    return np.where(coord_x > 0, "L", "R")


def _rank01(values: np.ndarray, ascending: bool) -> np.ndarray:
    """Rangos normalizados a [0, 1]; 0.5 constante si hay un solo valor.

    Se usa para poner ``CoordZ`` y ``sct_slice_num`` en una escala común
    dentro del grupo, sin suponer nada sobre sus unidades.
    """
    order = pd.Series(values).rank(method="first", ascending=ascending)
    n = len(order)
    if n == 1:
        return np.array([0.5])
    return ((order - 1) / (n - 1)).to_numpy()


def _cost_matrix(nodules: pd.DataFrame, cands: pd.DataFrame) -> np.ndarray:
    """Costo de asignar cada nódulo (filas) a cada anomalía (columnas).

    El costo base es la distancia entre las posiciones verticales
    normalizadas. Las anomalías sin ``sct_slice_num`` reciben un costo
    neutro (0.5) para que no atraigan ni repelan. La lateralidad se
    impone como costo prohibitivo, no como filtro previo, para que la
    asignación siempre exista.
    """
    # Superior primero en ambos: CoordZ descendente, slice_num ascendente.
    z_rank = _rank01(nodules["CoordZ"].to_numpy(), ascending=False)

    slices = cands["sct_slice_num"].to_numpy(dtype=float)
    known = np.isfinite(slices) & (slices != MISSING_SLICE)
    s_rank = np.full(len(cands), 0.5)
    if known.any():
        s_rank[known] = _rank01(slices[known], ascending=True)

    cost = np.abs(z_rank[:, None] - s_rank[None, :])
    # Las anomalías sin altura conocida quedan neutras frente a todos.
    cost[:, ~known] = 0.5

    nod_side = _side_from_x(nodules["CoordX"])
    cand_side = cands["sct_epi_loc"].map(SIDE_BY_LOBE).to_numpy()
    # Lóbulo sin lado definido (códigos 7, 8 o ausente): se permite con
    # recargo, no se prohíbe.
    unknown = pd.isna(cand_side)
    mismatch = (nod_side[:, None] != cand_side[None, :]) & ~unknown[None, :]
    cost = np.where(unknown[None, :], cost + UNKNOWN_SIDE_PENALTY, cost)
    return np.where(mismatch, FORBIDDEN, cost)


def _match_group(
    nodules: pd.DataFrame, cands: pd.DataFrame
) -> list[tuple[int, int | None, float]]:
    """Asignación uno a uno dentro de un (pid, study_yr).

    Returns
    -------
    list
        Tuplas ``(indice_nodulo, indice_anomalia | None, costo)``. El
        índice de anomalía es None cuando no hubo candidata admisible.
    """
    if cands.empty:
        return [(i, None, np.nan) for i in nodules.index]

    cost = _cost_matrix(nodules, cands)
    rows, cols = linear_sum_assignment(cost)

    out: list[tuple[int, int | None, float]] = []
    assigned = dict(zip(rows, cols))
    for i, idx in enumerate(nodules.index):
        j = assigned.get(i)
        # Sin pareja posible, o la única disponible viola la lateralidad.
        if j is None or cost[i, j] >= FORBIDDEN:
            out.append((idx, None, np.nan))
        else:
            out.append((idx, cands.index[j], float(cost[i, j])))
    return out


def emparejar_nodulos(
    labels: pd.DataFrame,
    ctab: pd.DataFrame,
    devolver_reporte: bool = False,
):
    """Empareja cada nódulo de LUNA25 con su propia anomalía de NLST.

    labels : DataFrame de LUNA25; necesita 'pid', 'study_yr', 'CoordX' y
             'CoordZ'. Se devuelve una copia, no se modifica in situ.
    ctab   : tabla de anomalías de NLST (nlst_780_ctab), con 'pid',
             'study_yr', 'sct_epi_loc', 'sct_slice_num' y 'sct_ab_desc'.
    devolver_reporte : si es True devuelve (df, reporte) en vez de df.

    El resultado conserva una fila por nódulo -- nunca hay fan-out -- y
    añade las columnas de ctab más tres de trazabilidad:

    match_estado        : "ok" | "sin_candidato" (no había anomalía de
                          tipo nódulo ese año) | "sin_lado" (las había,
                          pero ninguna del lado correcto quedó libre)
    match_n_candidatos  : anomalías de tipo nódulo en el paciente-año
    match_costo         : distancia vertical normalizada del par elegido
    match_lado          : "mismo" si el lóbulo confirma la lateralidad,
                          "desconocido" si el código de lóbulo no define
                          lado (7, 8 o ausente); estas últimas son las
                          únicas parejas no confirmadas por lateralidad

    Las filas sin pareja quedan en NaN de forma explícita: no se rellenan
    con la anomalía de otro nódulo.
    """
    faltan = {"pid", "study_yr", "CoordX", "CoordZ"} - set(labels.columns)
    if faltan:
        raise ValueError(f"labels no tiene las columnas: {sorted(faltan)}")
    faltan = {"pid", "study_yr", "sct_epi_loc", "sct_slice_num"} - set(
        ctab.columns
    )
    if faltan:
        raise ValueError(f"ctab no tiene las columnas: {sorted(faltan)}")

    out = labels.copy()
    cand_all = ctab[ctab["sct_ab_desc"] == NODULE_AB_DESC].copy()
    cand_all["sct_long_dia"] = pd.to_numeric(
        cand_all["sct_long_dia"], errors="coerce"
    )
    grupos_cand = dict(list(cand_all.groupby(["pid", "study_yr"])))

    pares: list[tuple[int, int | None, float]] = []
    n_cands: dict[int, int] = {}
    for clave, nodules in out.groupby(["pid", "study_yr"]):
        cands = grupos_cand.get(clave, cand_all.iloc[:0])
        for idx, j, costo in _match_group(nodules, cands):
            pares.append((idx, j, costo))
            n_cands[idx] = len(cands)

    idx_nod = [p[0] for p in pares]
    idx_ano = [p[1] for p in pares]
    costos = [p[2] for p in pares]

    emparejado = pd.Series(idx_ano, index=idx_nod)
    valores = cand_all.reindex(emparejado.dropna().to_numpy())
    valores.index = emparejado.dropna().index
    for col in CTAB_COLS:
        if col in cand_all.columns:
            out[col] = valores[col].reindex(out.index)

    out["match_n_candidatos"] = pd.Series(n_cands).reindex(out.index)
    out["match_costo"] = pd.Series(costos, index=idx_nod).reindex(out.index)
    emparejo = emparejado.reindex(out.index).notna()
    out["match_estado"] = np.where(
        emparejo,
        "ok",
        np.where(out["match_n_candidatos"] > 0, "sin_lado", "sin_candidato"),
    )
    lado_ctab = out["sct_epi_loc"].map(SIDE_BY_LOBE)
    out["match_lado"] = np.where(
        ~emparejo,
        None,
        np.where(
            lado_ctab.isna(),
            "desconocido",
            np.where(lado_ctab == _side_from_x(out["CoordX"]), "mismo", "?"),
        ),
    )

    if not devolver_reporte:
        return out

    conteo = out["match_estado"].value_counts()
    reporte = pd.DataFrame(
        {
            "filas": conteo,
            "porcentaje": (conteo / len(out) * 100).round(2),
        }
    )
    return out, reporte
