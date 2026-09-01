"""Une las partes del zip de LUNA25 y las descomprime en data/images.

Las rutas se resuelven desde la ubicación de este archivo, no desde el
directorio de trabajo, así que el script funciona igual ejecutándolo desde
la raíz del repo (`python data_loader/unzip_data.py`) o desde cualquier
otra carpeta.
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "images"

parts = sorted(IMAGES.glob("luna25_nodule_blocks.zip.*"))
combined_zip = IMAGES / "luna25_nodule_blocks_combined.zip"
output_folder = IMAGES / "extracted"

# 1. Verificar si la carpeta de destino ya existe y no está vacía
if output_folder.exists() and any(output_folder.iterdir()):
    print(
        f"La carpeta '{output_folder}' ya existe y contiene archivos. "
        "Proceso omitido."
    )
else:
    # 2. Verificar que existan las partes antes de intentar unir
    if not parts:
        raise FileNotFoundError(
            f"No se encontraron partes .zip.00n en {IMAGES}"
        )

    # 3. Concatenar las partes si el zip combinado no se ha generado antes
    if not combined_zip.exists():
        print("Uniendo partes...")
        with open(combined_zip, "wb") as outfile:
            for part in parts:
                with open(part, "rb") as infile:
                    # Bloques de 8 MB
                    while chunk := infile.read(8 * 1024 * 1024):
                        outfile.write(chunk)
    else:
        print(
            f"El archivo combinado '{combined_zip}' ya existe. Saltando unión."
        )

    # 4. Descomprimir el archivo final
    print("Descomprimiendo...")
    with zipfile.ZipFile(combined_zip, "r") as zip_ref:
        zip_ref.extractall(output_folder)

    print("Archivos extraídos exitosamente en:", output_folder)
