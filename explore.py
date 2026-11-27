# =============================================================================
# TimberWatch - explorare interactiva a datelor Sentinel-2
#
# Cum se foloseste:
#   - Rulare pe celule: cursorul intr-o celula + Shift+Enter (kernel = venv)
#   - Rulare completa:  python explore.py  (se ruleaza doar pasii activati
#                        din sectiunea SETARI de mai jos)
#
# Pasii:
#   1. Conectare la Copernicus Data Space (openEO)
#   2. Zona de lucru (mica, pentru analiza detaliata)
#   3. Test: o imagine RGB (verificam ca descarcarea merge)
#   4. Functia NDVI median fara nori
#   5. NDVI inainte / dupa pentru zona mica (rezolutie 10 m)
#   6. Grafic comparativ inainte / dupa / scadere
#   7. Scanare larga (60 m) ca sa GASIM taierile automat
#   8. Top 5 taieri din scanare + coordonatele lor + ZONA sugerata
# =============================================================================

# %% 0. SETARI - aici alegi ce pasi ruleaza cand rulezi tot fisierul
RULEAZA_TEST_RGB = False      # pasul 3 (doar prima data, ca verificare)
RULEAZA_ZONA_MICA = False     # pasul 5 (dupa ce ai ales zona reala)
ARATA_GRAFIC_ZONA = False     # pasul 6 (deschide o fereastra; scriptul asteapta)
RULEAZA_SCANARE = True        # pasul 7 (5-15 minute)
ARATA_GRAFIC_SCANARE = True   # pasul 8 (deschide o fereastra la final)

# Perioadele comparate: acelasi anotimp (vara), ani diferiti
PERIOADA_INAINTE = ["2025-06-01", "2025-08-31"]
PERIOADA_DUPA = ["2026-06-01", "2026-08-31"]


# %% 1. Importuri, log si conectare
import time
from datetime import datetime

import numpy as np
import openeo
import rasterio
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy import ndimage

START = time.time()


def log(mesaj):
    """Afiseaza un mesaj cu ora curenta si timpul scurs de la pornire."""
    trecut = time.time() - START
    print(f"[{datetime.now():%H:%M:%S}] (+{trecut:5.0f}s) {mesaj}", flush=True)


log("Conectare la Copernicus Data Space...")
# Prima data deschide un link de autentificare; apoi foloseste token-ul salvat.
conn = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
log(f"Conectat ca {conn.describe_account().get('name', '?')}")


# %% 2. Zona de lucru (mica, ~5 x 5 km)
# Dupa pasul 8, inlocuieste cu ZONA sugerata acolo (taierea gasita).
ZONA = {"west": 24.60, "south": 45.55, "east": 24.67, "north": 45.60}
log(f"Zona mica: {ZONA}")


# %% 3. Test: o imagine RGB (doar ca verificare ca descarcarea merge)
if RULEAZA_TEST_RGB:
    log("Test RGB: descarc mediana pe 1-15 iulie 2026...")
    test = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=ZONA,
        temporal_extent=["2026-07-01", "2026-07-15"],  # interval, nu o singura zi
        bands=["B04", "B03", "B02"],                   # rosu, verde, albastru
        max_cloud_cover=50,
    )
    test.reduce_dimension(dimension="t", reducer="median").download("test_rgb.tif")
    log("Test RGB descarcat -> test_rgb.tif")


# %% 4. Functia NDVI median, fara nori
# Clase SCL eliminate: 0 fara date, 1 saturat, 3 umbra de nor,
# 8/9 nori, 10 cirrus, 11 zapada
SCL_INVALID = [0, 1, 3, 8, 9, 10, 11]


def ndvi_median(conn, zona, perioada):
    """NDVI median pe o perioada, cu pixelii de nori/umbre/zapada eliminati.
    Nu descarca nimic: construieste doar 'reteta' care ruleaza pe server."""
    cube = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=zona,
        temporal_extent=perioada,
        bands=["B04", "B08", "SCL"],  # rosu, infrarosu apropiat, clasificare scena
        max_cloud_cover=40,
    )
    scl = cube.band("SCL")
    nori = scl == SCL_INVALID[0]
    for v in SCL_INVALID[1:]:
        nori = nori | (scl == v)
    curat = cube.filter_bands(["B04", "B08"]).mask(nori)
    ndvi = curat.ndvi(nir="B08", red="B04")          # (NIR - Rosu) / (NIR + Rosu)
    return ndvi.reduce_dimension(dimension="t", reducer="median")


# %% 5. NDVI inainte / dupa pentru zona mica (10 m)
if RULEAZA_ZONA_MICA:
    for nume, perioada in [("before", PERIOADA_INAINTE), ("after", PERIOADA_DUPA)]:
        log(f"Zona mica: descarc NDVI {nume} {perioada}...")
        ndvi_median(conn, ZONA, perioada).download(f"ndvi_{nume}.tif")
        log(f"Descarcat -> ndvi_{nume}.tif")


# %% 6. Grafic comparativ pentru zona mica
if ARATA_GRAFIC_ZONA:
    log("Desenez comparatia pentru zona mica...")
    with rasterio.open("ndvi_before.tif") as a, rasterio.open("ndvi_after.tif") as b:
        inainte, dupa = a.read(1), b.read(1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(inainte, cmap="RdYlGn", vmin=0, vmax=1); ax[0].set_title("Inainte")
    ax[1].imshow(dupa, cmap="RdYlGn", vmin=0, vmax=1); ax[1].set_title("Dupa")
    ax[2].imshow(inainte - dupa, cmap="Reds", vmin=0, vmax=0.6); ax[2].set_title("Scadere NDVI")
    plt.tight_layout()
    plt.savefig("ndvi_comparatie.png", dpi=150)   # buna pentru pitch
    log("Salvat ndvi_comparatie.png - inchide fereastra ca sa continue scriptul")
    plt.show()


# %% 7. Scanare larga (60 m) - gasim taierile automat
# ~55 x 45 km in Muntii Fagaras. Schimba daca vrei alt masiv.
ZONA_MARE = {"west": 24.3, "south": 45.4, "east": 25.0, "north": 45.68}
# 32635 = UTM zona 35 (est de 24°E). Pentru zone la vest de 24°E: 32634.
PROIECTIE_UTM = 32635


def ndvi_grosier(perioada):
    """NDVI median pe zona mare, redus la 60 m ca fisierul sa fie mic."""
    return ndvi_median(conn, ZONA_MARE, perioada).resample_spatial(
        resolution=60, projection=PROIECTIE_UTM
    )


if RULEAZA_SCANARE:
    for nume, perioada in [("before", PERIOADA_INAINTE), ("after", PERIOADA_DUPA)]:
        log(f"Scanare {nume} {perioada}: trimit jobul pe server...")
        # execute_batch afiseaza starea jobului la cateva secunde
        # (queued / running / finished). Jobul se vede si in editor.openeo.org.
        ndvi_grosier(perioada).execute_batch(
            outputfile=f"scan_{nume}.tif",
            title=f"TimberWatch scan {nume}",
            print=print,
        )
        log(f"Scanare {nume} descarcata -> scan_{nume}.tif")
    log("Scanarea s-a terminat")


# %% 8. Top 5 taieri din scanare + coordonate + ZONA sugerata
log("Caut cele mai mari zone cu scadere de NDVI...")
with rasterio.open("scan_before.tif") as a, rasterio.open("scan_after.tif") as b:
    inainte_s, dupa_s = a.read(1), b.read(1)
    transform, crs = a.transform, a.crs

# Aceleasi praguri ca in pipeline: era padure, acum e sol gol, scadere mare
taiere = (inainte_s > 0.6) & (dupa_s < 0.45) & ((inainte_s - dupa_s) > 0.3)
etichete, n = ndimage.label(taiere)
log(f"Zone candidate gasite: {n}")

if n > 0:
    marimi = ndimage.sum(taiere, etichete, range(1, n + 1))  # numar de pixeli
    top = np.argsort(marimi)[::-1][:5]
    la_gps = Transformer.from_crs(crs, 4326, always_xy=True)

    coordonate = []
    for rang, i in enumerate(top, start=1):
        rand, col = ndimage.center_of_mass(taiere, etichete, i + 1)
        lon, lat = la_gps.transform(*rasterio.transform.xy(transform, rand, col))
        ha = marimi[i] * 0.36            # un pixel de 60 m = 0,36 ha
        coordonate.append((lat, lon))
        print(f"  #{rang}: {ha:6.1f} ha  ->  lat {lat:.4f}, lon {lon:.4f}")

    # ZONA de ~5 x 5 km in jurul celei mai mari taieri: copiaz-o in pasul 2
    lat, lon = coordonate[0]
    sugestie = {"west": round(lon - 0.03, 4), "south": round(lat - 0.02, 4),
                "east": round(lon + 0.03, 4), "north": round(lat + 0.02, 4)}
    log(f"ZONA sugerata pentru #1 (verifica intai in Copernicus Browser): {sugestie}")

if ARATA_GRAFIC_SCANARE:
    plt.figure(figsize=(10, 8))
    plt.imshow(inainte_s - dupa_s, cmap="Reds", vmin=0, vmax=0.5)
    plt.title("Scadere NDVI - scanare larga"); plt.colorbar()
    plt.savefig("scan_comparatie.png", dpi=150)
    log("Salvat scan_comparatie.png")
    plt.show()

log("Gata")