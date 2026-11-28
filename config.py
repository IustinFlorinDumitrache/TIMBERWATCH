"""Toate setarile pipeline-ului intr-un singur loc."""
from pathlib import Path

# --- Zona-pilot (grade, WGS84). Tineti-o sub ~10 x 10 km la inceput. ---
ZONA = {"west": 24.60, "south": 45.55, "east": 24.75, "north": 45.65}

# --- Perioadele comparate (aceeasi luna/anotimp, ani diferiti) ---
PERIOADA_INAINTE = ["2025-06-01", "2025-08-31"]
PERIOADA_DUPA = ["2026-06-01", "2026-08-31"]
ETICHETA_INAINTE = "2025-summer"
ETICHETA_DUPA = "2026-summer"

# --- Sentinel-2 ---
COLECTIE = "SENTINEL2_L2A"
MAX_NORI = 40  # % acoperire maxima cu nori per scena
# Clase SCL eliminate: 0 no data, 1 saturat, 3 umbra nori, 8/9 nori, 10 cirrus, 11 zapada
SCL_INVALID = [0, 1, 3, 8, 9, 10, 11]
OPENEO_URL = "openeo.dataspace.copernicus.eu"

# --- Praguri de detectie ---
NDVI_PADURE_INAINTE = 0.60  # pixelul era padure
NDVI_MAX_DUPA = 0.45        # acum e sol gol / vegetatie rara
SCADERE_MIN = 0.30          # scaderea minima de NDVI
ARIE_MIN_HA = 0.5           # ignoram petele mai mici

# --- Foldere ---
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
ALERTS_FILE = DATA / "alerts.geojson"

# =====================================================================
# ML: reguli largi (candidati) + Random Forest (filtru)
# =====================================================================
# Zona pentru ML: interiorul masivului Fagaras, fara vaile Oltului
ML_ZONA = {"west": 24.40, "south": 45.45, "east": 24.95, "north": 45.67}
ML_REZOLUTIE = 20          # metri (rezolutia nativa a benzilor SWIR B11/B12)
ML_EPSG = 32635            # UTM 35N (est de 24°E); vest de 24°E: 32634
ML_BENZI = ["B02", "B03", "B04", "B08", "B11", "B12"]
COLECTIE_WORLDCOVER = "ESA_WORLDCOVER_10M_2021_V2"  # verificati numele cu list_collection_ids()
COLECTIE_DEM = "COPERNICUS_30"

# Praguri RELAXATE pentru candidati (prindem mai mult; modelul filtreaza)
ML_NDVI_INAINTE_MIN = 0.50
ML_SCADERE_MIN = 0.15
ML_ARIE_MIN_HA = 0.12
ML_PADURE_MIN = 0.3      # fractia minima de padure (WorldCover) in candidat

ML_DIR = DATA / "ml"
ML_CANDIDATI = ML_DIR / "candidates.gpkg"
ML_ETICHETE = ML_DIR / "labels.csv"
ML_MODEL = ML_DIR / "model.joblib"
ML_PRAG_PROBABILITATE = 0.5  # alertele cu probabilitate peste prag ajung pe harta