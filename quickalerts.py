import geopandas as gpd
import config

c = gpd.read_file(config.ML_CANDIDATI)
a = c[(c["ndvi_drop"] > 0.25) & (c["slope"] > 10)].copy()     # filtru simplu, provizoriu
a = a.sort_values("area_ha", ascending=False).reset_index(drop=True)
a["id"] = [f"TW-{i + 1:03d}" for i in range(len(a))]
a["probability"] = None
a["area_ha"] = a["area_ha"].round(2)
a["ndvi_drop"] = a["ndvi_drop"].round(2)
a["date_before"], a["date_after"] = config.ETICHETA_INAINTE, config.ETICHETA_DUPA
a["protected_area"], a["priority"] = None, "unknown"
a["img_before"] = "img/" + a["id"] + "_before.png"
a["img_after"] = "img/" + a["id"] + "_after.png"
cols = ["id", "cand_id", "probability", "area_ha", "ndvi_drop", "date_before", "date_after",
        "protected_area", "priority", "img_before", "img_after", "geometry"]
a[cols].to_crs(4326).to_file(config.ALERTS_FILE, driver="GeoJSON")
print(f"{len(a)} alerte -> {config.ALERTS_FILE}")