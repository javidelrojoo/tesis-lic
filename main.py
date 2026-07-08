"""
main
====

Punto de entrada del pipeline de extraccion: toma la imagen binaria de una
muestra (y opcionalmente las mascaras de electrodos), extrae la red con
``SkeletonAnalysis`` y guarda el grafo resultante en .graphml.

Uso:
    python main.py
Ajustar la muestra y las rutas en el bloque CONFIG.
"""

from SkeletonAnalysis import SkeletonAnalysis

# =============================================================================
#  CONFIG
# =============================================================================
MUESTRA = "M4"
DIR_PATH = f"./muestras/{MUESTRA}/"
IMG_PATH = f"{DIR_PATH}{MUESTRA}_binary.tif"
ELECTRODES_PATH = (
    f"{DIR_PATH}{MUESTRA}_electrode_mask1.tif",
    f"{DIR_PATH}{MUESTRA}_electrode_mask2.tif",
)
SAVE_GRAPH_PATH = f"{DIR_PATH}{MUESTRA}_effective.graphml"

CLEAN_NETWORK = True
USE_ELECTRODES = True
PLOT = True
VERBOSE = True


def main():
    """Corre el analisis completo sobre la muestra configurada y devuelve el objeto SkeletonAnalysis."""
    SA = SkeletonAnalysis(IMG_PATH)
    SA.complete_analysis(
        clean_network=CLEAN_NETWORK,
        electrodes_path=ELECTRODES_PATH if USE_ELECTRODES else None,
        save_graph_path=SAVE_GRAPH_PATH,
        plot=PLOT,
        verbose=VERBOSE,
    )
    return SA


if __name__ == "__main__":
    main()
