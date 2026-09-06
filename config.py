"""
Took from
https://github.com/DIAGNijmegen/luna25-baseline-public/blob/main/experiment_config.py
"""

from pathlib import Path


class Configuration(object):
    def __init__(self) -> None:

        # Working directory
        self.WORKDIR = Path(__file__).resolve().parent

        self.RESOURCES = self.WORKDIR / "resources"

        # Starting weights for the I3D model
        self.MODEL_RGB_I3D = (
            self.RESOURCES / "model_rgb.pth"
        )

        # Data parameters
        # Path to the nodule blocks folder provided for the LUNA25 training data.
        self.DATADIR = self.WORKDIR / "data"

        # Raíz que espera Luna25BlockDataset: contiene image/ y metadata/.
        self.NODULE_BLOCKS_ROOT = self.DATADIR / "images/extracted/luna25_nodule_blocks"

        # Labels for the LUNA25 training data, provided in CSV format.
        self.LUNA25_LABELS = self.DATADIR / "labels/LUNA25_Public_Training_Development_Data.csv"

        # Nlst extra data

        self.NLST_CSVS = self.DATADIR / "nlst_data"

        # Block Nodule Images
        self.NODULE_BLOCKS_DIR = self.DATADIR / "images/extracted/luna25_nodule_blocks/image"
        self.NODULE_METADATA_DIR = self.DATADIR / "images/extracted/luna25_nodule_blocks/metadata"

        # output directories
        self.OUTPUT_DIR = self.WORKDIR / "outputs"
        self.PLOTS_DIR = self.OUTPUT_DIR / "plots"
        self.TABLES_DIR = self.OUTPUT_DIR / "tables"
        self.PICKLE_DIR = self.OUTPUT_DIR / "pickles"

        # Path to the folder containing the CSVs for training and validation.
        self.CSV_DIR = self.WORKDIR / "../data/labels"
        # We provide an NLST dataset CSV, but participants are responsible for splitting the data into training and validation sets.
        self.CSV_DIR_TRAIN = self.CSV_DIR / "train.csv" # Path to the training CSV
        self.CSV_DIR_VALID = self.CSV_DIR / "valid.csv" # Path to the validation CSV

        # Results will be saved in the /results/ directory, inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        if not self.EXPERIMENT_DIR.exists():
            self.EXPERIMENT_DIR.mkdir(parents=True)

        self.EXPERIMENT_NAME = "LUNA25-baseline"
        self.MODE = "2D" # 2D or 3D

        # Training parameters
        self.SEED = 2025
        self.NUM_WORKERS = 8
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.BATCH_SIZE = 32
        self.ROTATION = ((-20, 20), (-20, 20), (-20, 20))
        self.TRANSLATION = True
        self.EPOCHS = 10 # itrations and test, then it shoud be augmented
        self.PATIENCE = 20
        self.PATCH_SIZE = [64, 128, 128]
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 5e-4

        # Interesting characteristics to analyze
        self.INTERESTING_COLS = {
        "patch_mean": "Densidad media de todo el parche de 50 mm.",
        "patch_std": "Desviación estándar de todo el parche.",
        "mean_r5mm": "Densidad media dentro de la esfera de radio 5 mm.",
        "mean_r10mm": "Densidad media dentro de la esfera de radio 10 mm.",
        "mean_r15mm": "Densidad media dentro de la esfera de radio 15 mm.",
        "std_r5mm": "Desviación estándar dentro de la esfera de radio 5 mm.",
        "std_r10mm": "Desviación estándar dentro de la esfera de radio 10 mm.",
        "std_r15mm": "Desviación estándar dentro de la esfera de radio 15 mm.",
        "solid_frac_r5mm": "Fracción sólida dentro de la esfera de radio 5 mm.",
        "solid_frac_r10mm": "Fracción sólida dentro de la esfera de radio 10 mm.",
        "solid_frac_r15mm": "Fracción sólida dentro de la esfera de radio 15 mm.",
        "mean_shell": "Densidad media dentro del casco.",
        "bg_lung_hu": "Densidad promedio en Unidades Hounsfield (HU) del parénquima pulmonar de fondo",
        "seg_leak_frac": "Fracción de la máscara que toca el límite de 20 mm.",
        "seg_ok": "Marca de fila utilizable, definida como seg_n_vox > 0",
        "volume_mm3": "Volumen de la máscara total.",
        "diam_mean_axes_mm": "Promedio de eje mayor y menor. Continua (mm). Lo más cercano al diámetro medio que usan Fleischner y Lung-RADS.",
        "sphericity": "Esfericidad: π^(1/3)·(6V)^(2/3) / S. Continua. 1 es una esfera perfecta; baja al alargarse o espicularse.",
        "closing_solidity_2mm": "V / V_cerrado: aproximación a la solidez del casco convexo. Proporción 0 – 1. Baja con la lobulación y la espiculación.",
        "radial_roughness": "Dispersión relativa del radio de los vóxeles de borde respecto al centroide.",
        "hu_mean": "Densidad media dentro de la máscara. Continua (HU). NaN si la máscara está vacía.",
        "hu_std": "Desviación estándar dentro de la máscara. Continua (HU).",
        "hu_p10": "Percentil 10 de densidad dentro de la máscara. Continua (HU).",
        "hu_p50": "Mediana de densidad dentro de la máscara. Continua (HU).",
        "hu_p90": "Percentil 90 de densidad dentro de la máscara. Continua (HU).",
        "hu_range_p10_p90": "hu_p90 − hu_p10: recorrido intercuantílico de densidad. Continua (HU). Proxy de heterogeneidad.",
        "hu_max": "Densidad máxima dentro de la máscara. Continua (HU).",
        "frac_air": "Fracción de vóxeles con densidad de aire (< −750 HU). Vale 0 por construcción en todas las filas.",
        "frac_ggo": "Fracción en ventana de vidrio esmerilado ([−750, −300) HU). Proporción 0 – 1.",
        "frac_soft": "Fracción en ventana de tejido blando ([−300, 200) HU). Proporción 0 – 1.",
        "frac_calcified": "Fracción calcificada (≥ 200 HU). Proporción 0 – 1. La calcificación es el patrón benigno clásico.",
        "solid_fraction": "Fracción del volumen que es sólida (> −300 HU). Proporción 0 – 1. 1 = sólido, 0 = vidrio esmerilado puro, intermedio = subsólido.",
        "volume_solid_mm3": "Volumen del componente sólido. Continua (mm³).",
        "contrast_hu": "hu_mean − bg_lung_hu: realce de la lesión sobre el parénquima. Continua (HU). Siempre positiva por construcción.",
        "diam_equiv_solid_mm": "Diámetro de la esfera de igual volumen que el componente sólido. Continua (mm).",
    }


config = Configuration()