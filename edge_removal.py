"""
edge_removal
============

Experimentos de robustez de la red frente a la eliminacion aleatoria de aristas
(cada arista se elimina con probabilidad ``p``). Portado desde
``Codigos/03_analisis/graph_analysis.py``.

Contenido:
    - edge_removal_dependency:  una corrida por cada p -> C(p), L(p), E(p).
    - edge_removal_variability: N corridas por cada p -> permite media +/- desvio.
    - plot_clustering_vs_p / plot_path_length_vs_p / plot_efficiency_vs_p.
    - fit_path_length_powerlaw / fit_clustering_decay: ajustes de curvas.
    - save_results / load_results: persistencia con pickle.

Las metricas (clustering, path length, global efficiency) se calculan con las
funciones de ``NetworkMetrics`` para no duplicar logica.
"""

import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

# NetworkMetrics vive en el repo tesis-lic (github). Si este archivo se ejecuta
# desde otra carpeta (p. ej. la copia en el Drive), agregamos el repo al path.
try:
    import NetworkMetrics as NM
except ModuleNotFoundError:
    _REPO = r"C:\Users\javit\Desktop\tesis-lic (github)"
    if os.path.isdir(_REPO) and _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    import NetworkMetrics as NM


DEFAULT_PS = np.linspace(0.2, 0.5, 30)
_METRIC_KEYS = ["ccoeffs", "paths", "efficiencies",
                "paths_electrodes", "efficiencies_electrodes"]


# =============================================================================
#  Experimentos
# =============================================================================

def edge_removal_dependency(grafos, ps=DEFAULT_PS, k=1000, seed=23082000):
    """
    Una corrida por cada probabilidad p.

    Para cada grafo y cada p: copia el grafo, elimina cada arista con prob p, y
    mide clustering (C), path length (L) y global efficiency (E). Si el grafo
    tiene electrodos (Vin/Vout) tambien mide L y E entre electrodos.

    Devuelve un dict con las claves de ``_METRIC_KEYS``; cada una es
    {nombre_grafo: lista_por_p}. Las claves de electrodos valen None si el grafo
    no tiene electrodos.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    resultados = {key: {} for key in _METRIC_KEYS}

    for nombre, G_original in grafos.items():
        print(f"\nProcesando {nombre}:")
        Vin, Vout = NM.get_electrode_nodes(G_original)
        tiene_electrodos = bool(Vin and Vout)

        res = {key: [] for key in _METRIC_KEYS}

        for p in tqdm(ps):
            G_temp = G_original.copy()
            G_temp.remove_edges_from([e for e in G_temp.edges() if random.random() < p])

            res["ccoeffs"].append(NM.get_avg_clustering(G_temp))
            pl, ge = NM.path_length_efficiency_sampled(G_temp, k=k, metric="both")
            res["paths"].append(pl)
            res["efficiencies"].append(ge)

            if tiene_electrodos:
                Vin_t, Vout_t = NM.get_electrode_nodes(G_temp)
                pl_e, ge_e = NM.path_length_efficiency_src2tgt(G_temp, Vin_t, Vout_t, metric="both")
                res["paths_electrodes"].append(pl_e)
                res["efficiencies_electrodes"].append(ge_e)

        for key in ["ccoeffs", "paths", "efficiencies"]:
            resultados[key][nombre] = res[key]
        for key in ["paths_electrodes", "efficiencies_electrodes"]:
            resultados[key][nombre] = res[key] if tiene_electrodos else None

    return resultados


def edge_removal_variability(grafos, ps=DEFAULT_PS, n_runs=20, k=1000, seed=23082000):
    """
    Igual que ``edge_removal_dependency`` pero repitiendo ``n_runs`` veces cada p
    (con semillas distintas) para poder estimar la variabilidad.

    Cada metrica queda como un array de shape (len(ps), n_runs), listo para
    calcular media y desvio con ``axis=1``.
    """
    rng = np.random.default_rng(seed)  # generador madre

    resultados = {key: {} for key in _METRIC_KEYS}

    for nombre, G_original in grafos.items():
        print(f"\nProcesando {nombre}:")
        Vin, Vout = NM.get_electrode_nodes(G_original)
        tiene_electrodos = bool(Vin and Vout)

        res = {key: [[] for _ in ps] for key in _METRIC_KEYS}

        for run in range(n_runs):
            run_seed = int(rng.integers(0, 2 ** 31))
            rng_run = np.random.default_rng(run_seed)

            for i, p in enumerate(tqdm(ps, desc=f"run {run + 1}/{n_runs}")):
                G_temp = G_original.copy()
                edges = list(G_temp.edges())
                remove = [e for e in edges if rng_run.random() < p]
                G_temp.remove_edges_from(remove)

                res["ccoeffs"][i].append(NM.get_avg_clustering(G_temp))
                pl, ge = NM.path_length_efficiency_sampled(G_temp, k=k, metric="both")
                res["paths"][i].append(pl)
                res["efficiencies"][i].append(ge)

                if tiene_electrodos:
                    Vin_t, Vout_t = NM.get_electrode_nodes(G_temp)
                    pl_e, ge_e = NM.path_length_efficiency_src2tgt(G_temp, Vin_t, Vout_t, metric="both")
                    res["paths_electrodes"][i].append(pl_e)
                    res["efficiencies_electrodes"][i].append(ge_e)

        for key in ["ccoeffs", "paths", "efficiencies"]:
            resultados[key][nombre] = np.array(res[key])  # (len(ps), n_runs)
        for key in ["paths_electrodes", "efficiencies_electrodes"]:
            resultados[key][nombre] = np.array(res[key]) if tiene_electrodos else None

    return resultados


# =============================================================================
#  Persistencia
# =============================================================================

def save_results(resultados, path):
    """Guarda el dict de resultados en un archivo pickle."""
    with open(path, "wb") as f:
        pickle.dump(resultados, f)
    print(f"Datos guardados en '{path}'")


def load_results(path):
    """Carga un dict de resultados desde un archivo pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
#  Graficos vs p
#
#  Aceptan tanto resultados de una corrida (listas 1D) como de variabilidad
#  (arrays 2D): en el segundo caso grafican la media con banda de +/- 1 desvio.
# =============================================================================

def _mean_std(values):
    """Devuelve (media, desvio) por p. Si es 1D, desvio es None."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        return arr.mean(axis=1), arr.std(axis=1)
    return arr, None


def _style(estilos, nombre):
    e = (estilos or {}).get(nombre, {})
    return e.get("color", "black"), e.get("marker", "o"), e.get("alpha", 1.0)


def plot_clustering_vs_p(resultados, ps=DEFAULT_PS, estilos=None, labels=None, ax=None):
    """Coeficiente de agrupamiento C en funcion de p."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    for nombre, vals in resultados["ccoeffs"].items():
        mean, std = _mean_std(vals)
        color, marker, alpha = _style(estilos, nombre)
        lbl = (labels or {}).get(nombre, nombre)
        x = ps[:len(mean)]
        ax.plot(x, mean, color=color, marker=marker, alpha=alpha, label=lbl)
        if std is not None:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Probabilidad de eliminacion de arista", fontsize=16)
    ax.set_ylabel("Coeficiente de agrupamiento ($C$)", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def _plot_metric_with_electrodes(resultados, base_key, elec_key, ylabel,
                                 ps, estilos, labels, ax, mark_pmax):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    for nombre, vals in resultados[base_key].items():
        mean, std = _mean_std(vals)
        color, marker, alpha = _style(estilos, nombre)
        lbl = (labels or {}).get(nombre, nombre)
        x = ps[:len(mean)]
        ax.plot(x, mean, color=color, marker=marker, alpha=alpha, label=lbl)
        if std is not None:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        elec = resultados.get(elec_key, {}).get(nombre)
        if elec is not None:
            emean, estd = _mean_std(elec)
            xe = ps[:len(emean)]
            ax.plot(xe, emean, marker="v", linestyle="dashed", color=color,
                    alpha=0.6, label=lbl + " (Vin<->Vout)")
            if estd is not None:
                ax.fill_between(xe, emean - estd, emean + estd, color=color, alpha=0.1)

            if mark_pmax:
                arr = np.asarray(emean, dtype=float)
                finite = np.isfinite(arr)
                if finite.any():
                    xf = xe[finite]
                    vf = arr[finite]
                    p_max = xf[np.argmax(vf)]
                    ax.vlines(p_max, 0, np.max(vf), linestyles="dashed", colors=color, zorder=-1)

    ax.set_xlabel("Probabilidad de eliminacion de arista", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_path_length_vs_p(resultados, ps=DEFAULT_PS, estilos=None, labels=None,
                          ax=None, mark_pmax=True):
    """Longitud de camino L en funcion de p (con overlay de electrodos si existe)."""
    return _plot_metric_with_electrodes(
        resultados, "paths", "paths_electrodes", "Longitud de camino ($L$)",
        ps, estilos, labels, ax, mark_pmax)


def plot_efficiency_vs_p(resultados, ps=DEFAULT_PS, estilos=None, labels=None,
                         ax=None, mark_pmax=False):
    """Eficiencia global E en funcion de p (con overlay de electrodos si existe)."""
    return _plot_metric_with_electrodes(
        resultados, "efficiencies", "efficiencies_electrodes", "Eficiencia global ($E$)",
        ps, estilos, labels, ax, mark_pmax)


# =============================================================================
#  Ajustes de curvas
# =============================================================================

def fit_path_length_powerlaw(ps, L, asintota_idx=-4, x0_manual=None):
    """
    Ajusta L(p) = A * (x0 - p)^(-beta) (divergencia tipo ley de potencias cerca
    de la percolacion). ``asintota_idx`` recorta los ultimos puntos (ruidosos).

    Devuelve (popt, pcov, modelo) con popt = (A, x0, beta).
    """
    def modelo(x, A, x0, beta):
        return A * (-x + x0) ** (-beta)

    L = np.asarray(L, dtype=float)[:asintota_idx]
    p = np.asarray(ps, dtype=float)[:asintota_idx]

    if x0_manual is None:
        x0_manual = p.max() + 0.01

    # Estimacion inicial por ajuste lineal en log-log
    log_p = np.log(-p + x0_manual)
    log_L = np.log(L)
    beta_est, logA_est = np.polyfit(log_p, log_L, 1)
    beta_est = -beta_est
    A_est = np.exp(logA_est)

    popt, pcov = curve_fit(modelo, p, L, p0=[A_est, x0_manual, beta_est], maxfev=10_000)
    return popt, pcov, modelo


def fit_clustering_decay(ps, ccoeffs):
    """
    Ajusta C(p) = C0 * (1 - p)^alpha.
    Devuelve (popt, pcov, modelo) con popt = (C0, alpha).
    """
    def modelo(p, C0, alpha):
        return C0 * (1 - p) ** alpha

    ps_a = np.asarray(ps, dtype=float)[:len(ccoeffs)]
    cc = np.asarray(ccoeffs, dtype=float)
    popt, pcov = curve_fit(modelo, ps_a, cc, p0=[cc[0], 3.0])
    return popt, pcov, modelo
