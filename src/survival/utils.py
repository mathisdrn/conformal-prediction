import zipfile
from io import BytesIO
from pathlib import Path

import requests


def download_and_extract_zip(url, extract_to, expected_filename):
    """Télécharge et extrait un fichier ZIP si le fichier n'existe pas déjà"""
    # Vérifier si le fichier existe déjà
    expected_path = extract_to / expected_filename

    if expected_path.exists():
        print(f"✓ Fichier déjà présent: {expected_filename}")
        return

    print(f"Téléchargement depuis {url}...")
    response = requests.get(url)
    response.raise_for_status()

    print(f"Extraction dans {extract_to}...")
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_to)

    print("✓ Extraction terminée")


def download_data():
    """Télécharge et extrait les fichiers de données nécessaires."""

    ressources = [
        {
            "url": "https://github.com/MINCHELLA-Paul/Master-MIASHS/raw/6abd32cc11d73850a0d8c54a3ab9a31200b6d97b/Atelier_SigBERT/df_study_selected.zip",
            "expected_file": "df_study_L18_w6.csv",
        },
        {
            "url": "https://github.com/MINCHELLA-Paul/Master-MIASHS/raw/6abd32cc11d73850a0d8c54a3ab9a31200b6d97b/Atelier_SigBERT/df_study_selected_L36_w6.zip",
            "expected_file": "df_study_L36_w6.csv",
        },
    ]

    data_dir = Path("../../data/")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Télécharger et extraire si nécessaire
    download_and_extract_zip(
        ressources[0]["url"], data_dir, ressources[0]["expected_file"]
    )
    download_and_extract_zip(
        ressources[1]["url"], data_dir, ressources[1]["expected_file"]
    )

    print(f"\nFichiers dans {data_dir}:")
    for file in sorted(data_dir.glob("*")):
        print(f"  - {file.name}")
