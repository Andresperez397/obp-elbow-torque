"""Download the OBP pitching POI files used by this project.

Data: The OpenBiomechanics Project (Driveline Baseball), CC BY-NC-SA 4.0.
https://github.com/drivelineresearch/openbiomechanics
The raw files are not redistributed in this repository; run this script instead.
"""
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/baseball_pitching/data"
FILES = {
    "poi_metrics.csv": f"{BASE}/poi/poi_metrics.csv",
    "metadata.csv": f"{BASE}/metadata.csv",
    "data_dictionary.csv": f"{BASE}/data_dictionary.csv",
}

def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        urlretrieve(url, out / name)
        print(f"saved {out / name}")

if __name__ == "__main__":
    main()
