from pathlib import Path
import urllib.request
import logging
from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fichiers UD_Arabic-PADT officiels
UD_PADT_BASE_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/"
FILES = {
    "train": "ar_padt-ud-train.conllu",
    "dev": "ar_padt-ud-dev.conllu",
    "test": "ar_padt-ud-test.conllu",
}

def download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"Fichier déjà présent : {dest_path}")
        return
    logger.info(f"Téléchargement de {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    logger.info(f"Téléchargement terminé : {dest_path.name} ({dest_path.stat().st_size} octets)")

def main():
    cfg = load_config()
    raw_dir = cfg.paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    for split, filename in FILES.items():
        url = UD_PADT_BASE_URL + filename
        target_path = raw_dir / filename
        download_file(url, target_path)

    logger.info("Tous les fichiers UD_Arabic-PADT sont prêts dans data/raw/")

if __name__ == "__main__":
    main()
