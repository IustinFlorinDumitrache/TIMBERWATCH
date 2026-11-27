
import openeo
import rasterio
import matplotlib.pyplot as plt

conn = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
print(conn.describe_account())

# %% 2. Zona-pilot (inlocuieste cu coordonatele tale, ~5 x 5 km)
ZONA = {"west": 24.60, "south": 45.55, "east": 24.67, "north": 45.60}

# %% 3. Test: o singura imagine RGB dintr-o zi senina
test = conn.load_collection(
    "SENTINEL2_L2A",
    spatial_extent=ZONA,
    temporal_extent=["2026-07-10", "2026-07-11"],  # o zi senina vazuta in Copernicus Browser
    bands=["B04", "B03", "B02"],
)
test.download("test_rgb.tif")
print("Descarcat test_rgb.tif - deschide-l in QGIS")

# %% 4. Functia NDVI median pe o perioada, fara nori
def ndvi_median(conn, zona, perioada):
    cube = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=zona,
        temporal_extent=perioada,
        bands=["B04", "B08", "SCL"],
        max_cloud_cover=40,
    )
    scl = cube.band("SCL")
    nori = (scl == 0) | (scl == 1) | (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 11)
    curat = cube.filter_bands(["B04", "B08"]).mask(nori)
    return curat.ndvi(nir="B08", red="B04").reduce_dimension(dimension="t", reducer="median")

# %% 5. Descarcare NDVI inainte / dupa (dureaza cateva minute)
ndvi_median(conn, ZONA, ["2025-06-01", "2025-08-31"]).download("ndvi_before.tif")
ndvi_median(conn, ZONA, ["2026-06-01", "2026-08-31"]).download("ndvi_after.tif")
print("Descarcat ndvi_before.tif si ndvi_after.tif")

# %% 6. Verificare vizuala
with rasterio.open("ndvi_before.tif") as a, rasterio.open("ndvi_after.tif") as b:
    inainte, dupa = a.read(1), b.read(1)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(inainte, cmap="RdYlGn", vmin=0, vmax=1); ax[0].set_title("Inainte")
ax[1].imshow(dupa, cmap="RdYlGn", vmin=0, vmax=1); ax[1].set_title("Dupa")
ax[2].imshow(inainte - dupa, cmap="Reds", vmin=0, vmax=0.6); ax[2].set_title("Scadere NDVI")
plt.tight_layout()
plt.savefig("ndvi_comparatie.png", dpi=150)  # buna pentru pitch
plt.show()