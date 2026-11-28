
import time
from datetime import datetime

import openeo

import config

START = time.time()


def run(cube, name):
    out = config.ML_DIR / name
    if out.exists():
        log(f"Exista deja {name} - sar peste"); return
    log(f"Trimit jobul pentru {name} ...")
    cube.execute_batch(outputfile=out, title=f"TimberWatch ML {name}", print=print)
    log(f"Descarcat -> {out}")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] (+{time.time() - START:5.0f}s) {msg}", flush=True)


def scl_mask(cube):
    scl = cube.band("SCL")
    m = scl == config.SCL_INVALID[0]
    for v in config.SCL_INVALID[1:]:
        m = m | (scl == v)
    return m


def s2_composite(conn, period):
    cube = conn.load_collection(
        config.COLECTIE, spatial_extent=config.ML_ZONA, temporal_extent=period,
        bands=config.ML_BENZI + ["SCL"], max_cloud_cover=config.MAX_NORI,
    )
    clean = cube.filter_bands(config.ML_BENZI).mask(scl_mask(cube))
    comp = clean.reduce_dimension(dimension="t", reducer="median")
    return comp.resample_spatial(resolution=config.ML_REZOLUTIE, projection=config.ML_EPSG,
                                 method="bilinear")


def static_layer(conn, collection, band, method):
    cube = conn.load_collection(collection, spatial_extent=config.ML_ZONA, bands=[band])
    try:
        cube = cube.reduce_dimension(dimension="t", reducer="max")  # unele colectii au timp
    except Exception:
        pass
    return cube.resample_spatial(resolution=config.ML_REZOLUTIE, projection=config.ML_EPSG,
                                 method=method)


def run(cube, name):
    out = config.ML_DIR / name
    log(f"Trimit jobul pentru {name} ...")
    cube.execute_batch(outputfile=out, title=f"TimberWatch ML {name}", print=print)
    log(f"Descarcat -> {out}")


def main():
    config.ML_DIR.mkdir(parents=True, exist_ok=True)
    log("Conectare la Copernicus...")
    conn = openeo.connect(config.OPENEO_URL).authenticate_oidc()

    ids = conn.list_collection_ids()
    for col in (config.COLECTIE_WORLDCOVER, config.COLECTIE_DEM):
        if col not in ids:
            sugestii = [c for c in ids if col.split("_")[0] in c]
            raise SystemExit(f"Colectia {col} nu exista. Posibile: {sugestii}. Schimbati in config.py.")

    run(s2_composite(conn, config.PERIOADA_INAINTE), "s2_before.tif")
    run(s2_composite(conn, config.PERIOADA_DUPA), "s2_after.tif")
    run(static_layer(conn, config.COLECTIE_WORLDCOVER, "MAP", "near"), "worldcover.tif")
    run(static_layer(conn, config.COLECTIE_DEM, "DEM", "bilinear"), "dem.tif")
    log("Gata.")


if __name__ == "__main__":
    main()