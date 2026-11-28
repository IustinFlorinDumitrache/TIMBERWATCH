"""Functii comune ML: citirea rasterelor, indici, candidati si caracteristici.

IMPORTANT: aceleasi functii sunt folosite si la antrenare, si la predictie,
ca modelul sa vada mereu exact aceleasi caracteristici.
"""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from scipy import ndimage
from shapely.geometry import shape

import config

# Ordinea benzilor in s2_*.tif (aceeasi ca in config.ML_BENZI)
B = {name: i for i, name in enumerate(config.ML_BENZI)}

FEATURES = [
    "ndvi_before", "ndvi_after", "ndvi_drop",
    "nbr_before", "nbr_after", "nbr_drop",
    "ndmi_before", "ndmi_after", "ndmi_drop",
    "b04_after", "b11_after", "b12_after",
    "area_ha", "compactness", "rectangularity",
    "elevation", "slope", "forest_frac",
]


# ---------------------------------------------------------------- rastere
def _read_aligned(path, ref, resampling):
    """Citeste un raster si il aliniaza pe grila de referinta (ref = profil rasterio)."""
    with rasterio.open(path) as src:
        out = np.full((src.count, ref["height"], ref["width"]), np.nan, dtype="float32")
        for i in range(src.count):
            reproject(
                source=rasterio.band(src, i + 1), destination=out[i],
                dst_transform=ref["transform"], dst_crs=ref["crs"],
                resampling=resampling, dst_nodata=np.nan,
            )
    return out


def load_stack():
    """Returneaza un dict cu toate rasterele pe aceeasi grila + profilul grilei."""
    d = config.ML_DIR
    with rasterio.open(d / "s2_before.tif") as src:
        ref = src.profile.copy()
        before = src.read().astype("float32")
    after = _read_aligned(d / "s2_after.tif", ref, Resampling.nearest)
    wc = _read_aligned(d / "worldcover.tif", ref, Resampling.nearest)[0]
    dem = _read_aligned(d / "dem.tif", ref, Resampling.bilinear)[0]
    for arr in (before, after):
        arr[arr <= 0] = np.nan  # valorile 0 = fara date
    return {"before": before, "after": after, "worldcover": wc, "dem": dem, "profile": ref}


def _nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def indices(img):
    """NDVI, NBR, NDMI pentru un stack de 6 benzi."""
    return {
        "ndvi": _nd(img[B["B08"]], img[B["B04"]]),
        "nbr": _nd(img[B["B08"]], img[B["B12"]]),
        "ndmi": _nd(img[B["B08"]], img[B["B11"]]),
    }


def slope_deg(dem, pixel_m):
    gy, gx = np.gradient(dem, pixel_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


# ---------------------------------------------------------------- candidati
def candidate_labels(stack):
    """Regulile LARGI: era vegetatie, NDVI a scazut. Returneaza (etichete, n)."""
    ib, ia = indices(stack["before"]), indices(stack["after"])
    drop = ib["ndvi"] - ia["ndvi"]
    valid = np.isfinite(drop)
    cand = valid & (ib["ndvi"] > config.ML_NDVI_INAINTE_MIN) & (drop > config.ML_SCADERE_MIN)
    cand = ndimage.binary_opening(cand, iterations=1)
    cand = ndimage.binary_closing(cand, iterations=1)
    labels, n = ndimage.label(cand)
    return labels, n


def _polygons(labels, profile):
    feats = [{"geometry": shape(g), "label": int(v)}
             for g, v in shapes(labels.astype("int32"), mask=labels > 0,
                                transform=profile["transform"])]
    return gpd.GeoDataFrame(feats, crs=profile["crs"]).dissolve(by="label").reset_index()


def build_candidates(stack):
    """Candidati (poligoane) + toate caracteristicile, intr-un GeoDataFrame."""
    labels, n = candidate_labels(stack)
    if n == 0:
        return gpd.GeoDataFrame(columns=["label", "geometry"] + FEATURES, crs=stack["profile"]["crs"])
    ids = np.arange(1, n + 1)
    px = abs(stack["profile"]["transform"].a)

    def zmean(arr):
        a = np.where(np.isfinite(arr), arr, 0)
        cnt = ndimage.sum(np.isfinite(arr), labels, ids)
        with np.errstate(invalid="ignore", divide="ignore"):
            return ndimage.sum(a, labels, ids) / cnt

    ib, ia = indices(stack["before"]), indices(stack["after"])
    vals = {}
    for k in ("ndvi", "nbr", "ndmi"):
        vals[f"{k}_before"] = zmean(ib[k])
        vals[f"{k}_after"] = zmean(ia[k])
        vals[f"{k}_drop"] = vals[f"{k}_before"] - vals[f"{k}_after"]
    for band in ("B04", "B11", "B12"):
        vals[f"{band.lower()}_after"] = zmean(stack["after"][B[band]])
    vals["elevation"] = zmean(stack["dem"])
    vals["slope"] = zmean(slope_deg(stack["dem"], px))
    vals["forest_frac"] = zmean((stack["worldcover"] == 10).astype("float32"))

    gdf = _polygons(labels, stack["profile"])
    for k, v in vals.items():
        gdf[k] = gdf["label"].map(dict(zip(ids, v)))

    # forma (in metri, grila e in UTM)
    gdf["area_ha"] = gdf.area / 10_000
    gdf["compactness"] = 4 * np.pi * gdf.area / gdf.length ** 2
    gdf["rectangularity"] = gdf.area / gdf.geometry.minimum_rotated_rectangle().area

    gdf = gdf[(gdf["area_ha"] >= config.ML_ARIE_MIN_HA)
              & (gdf["forest_frac"] >= config.ML_PADURE_MIN)].copy()

    cent = gdf.geometry.centroid.to_crs(4326)
    gdf["lat"], gdf["lon"] = cent.y.round(5), cent.x.round(5)
    gdf["cand_id"] = [f"C{int(l):05d}" for l in gdf["label"]]
    return gdf[["cand_id", "lat", "lon"] + FEATURES + ["geometry"]].reset_index(drop=True)