"""ML PASUL 3: unealta de etichetare manuala.

    python ml_step3_label.py

Pentru fiecare candidat vezi: inainte (RGB) | dupa (RGB) | scaderea NDVI, cu conturul candidatului.
Taste:   y = taiere reala    n = nu e taiere    s = sari peste    q = iesire
Etichetele se salveaza dupa fiecare tasta in data/ml/labels.csv (poti relua oricand).
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.transform import rowcol

import config
import ml_features

CHIP = 50  # pixeli in jurul centrului (50 x 20 m = 1 km in fiecare directie)


def rgb(img, r0, r1, c0, c1):
    b = ml_features.B
    x = np.stack([img[b["B04"], r0:r1, c0:c1], img[b["B03"], r0:r1, c0:c1],
                  img[b["B02"], r0:r1, c0:c1]], axis=-1)
    return np.clip(np.nan_to_num(x) / 2500, 0, 1) ** 0.8


def main():
    stack = ml_features.load_stack()
    tr = stack["profile"]["transform"]
    H, W = stack["before"].shape[1:]
    drop = ml_features.indices(stack["before"])["ndvi"] - ml_features.indices(stack["after"])["ndvi"]

    cand = gpd.read_file(config.ML_CANDIDATI)
    done = pd.read_csv(config.ML_ETICHETE) if config.ML_ETICHETE.exists() else \
        pd.DataFrame(columns=["cand_id", "label"])
    todo = cand[~cand["cand_id"].isin(done["cand_id"])].sample(frac=1, random_state=42)
    print(f"Etichetate: {len(done)}   De etichetat: {len(todo)}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))
    state = {"i": 0, "labels": done}

    def show():
        if state["i"] >= len(todo):
            plt.close(fig); return
        row = todo.iloc[state["i"]]
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        r, c = rowcol(tr, cx, cy)
        r0, r1 = max(r - CHIP, 0), min(r + CHIP, H)
        c0, c1 = max(c - CHIP, 0), min(c + CHIP, W)
        ext = (c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5)
        for a in ax:
            a.clear(); a.set_xticks([]); a.set_yticks([])
        ax[0].imshow(rgb(stack["before"], r0, r1, c0, c1), extent=ext); ax[0].set_title("Inainte")
        ax[1].imshow(rgb(stack["after"], r0, r1, c0, c1), extent=ext); ax[1].set_title("Dupa")
        ax[2].imshow(drop[r0:r1, c0:c1], cmap="Reds", vmin=0, vmax=0.5, extent=ext)
        ax[2].set_title("Scadere NDVI")
        geoms = [row.geometry] if row.geometry.geom_type == "Polygon" else list(row.geometry.geoms)
        for g in geoms:
            xs, ys = g.exterior.xy
            rr, cc = rowcol(tr, list(xs), list(ys))
            for a in ax:
                a.plot(cc, rr, color="cyan", lw=1.2)
        n = len(state["labels"])
        fig.suptitle(f"{row.cand_id}  |  {row.area_ha:.1f} ha  |  lat {row.lat}, lon {row.lon}  |  "
                     f"etichetate: {n}    [y] taiere  [n] nu  [s] sari  [q] iesire")
        fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key not in ("y", "n", "s", "q"):
            return
        if ev.key == "q":
            plt.close(fig); return
        if ev.key in ("y", "n"):
            row = todo.iloc[state["i"]]
            new = pd.DataFrame([{"cand_id": row.cand_id, "label": 1 if ev.key == "y" else 0}])
            state["labels"] = pd.concat([state["labels"], new], ignore_index=True)
            state["labels"].to_csv(config.ML_ETICHETE, index=False)
        state["i"] += 1
        show()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()
    lab = state["labels"]
    print(f"Salvat {len(lab)} etichete -> {config.ML_ETICHETE}  "
          f"(taieri: {int((lab['label'] == 1).sum())}, nu: {int((lab['label'] == 0).sum())})")


if __name__ == "__main__":
    main()