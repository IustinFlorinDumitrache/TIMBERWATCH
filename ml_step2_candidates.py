"""ML PASUL 2: gaseste candidatii (reguli largi) si calculeaza caracteristicile.

    python ml_step2_candidates.py

Iesire: data/ml/candidates.gpkg  (poligoane + caracteristici, se deschide si in QGIS)
"""
import config
import ml_features


def main():
    print("Citesc rasterele...")
    stack = ml_features.load_stack()
    print("Caut candidatii si calculez caracteristicile...")
    gdf = ml_features.build_candidates(stack)
    gdf.to_file(config.ML_CANDIDATI, driver="GPKG")
    print(f"Candidati: {len(gdf)} -> {config.ML_CANDIDATI}")
    if len(gdf):
        print(gdf[["cand_id", "area_ha", "ndvi_drop", "nbr_drop", "rectangularity", "slope"]]
              .sort_values("area_ha", ascending=False).head(10).round(2).to_string(index=False))


if __name__ == "__main__":
    main()