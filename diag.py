import numpy as np
import ml_features

s = ml_features.load_stack()
ib = ml_features.indices(s["before"])
ia = ml_features.indices(s["after"])
drop = ib["ndvi"] - ia["ndvi"]
wc = s["worldcover"]

print("Dimensiune grila:", s["before"].shape)
print("Pixeli valizi inainte / dupa:",
      round(np.isfinite(ib["ndvi"]).mean(), 3), "/", round(np.isfinite(ia["ndvi"]).mean(), 3))
print("NDVI median inainte / dupa:",
      round(float(np.nanmedian(ib["ndvi"])), 3), "/", round(float(np.nanmedian(ia["ndvi"])), 3))
print("Valori benzi (B08) min/median/max:",
      np.nanmin(s["before"][3]), np.nanmedian(s["before"][3]), np.nanmax(s["before"][3]))
print("Procent pixeli cu scadere NDVI > 0.2:", round(float(np.nanmean(drop > 0.2)) * 100, 2), "%")
print("Procent pixeli cu scadere NDVI > 0.15:", round(float(np.nanmean(drop > 0.15)) * 100, 2), "%")
vals, counts = np.unique(wc[np.isfinite(wc)], return_counts=True)
print("WorldCover (clasa: procent):", {int(v): round(c / counts.sum() * 100, 1) for v, c in zip(vals, counts)})