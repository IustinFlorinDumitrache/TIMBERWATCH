# TimberWatch (carcasă UI)

Interfața e gata; datele satelitare se conectează într-un singur loc: **`src/api.js`**.
Fiecare funcție de acolo are contractul documentat (formatul GeoJSON, parametri, atribuiri).
Până se implementează, aplicația afișează "—" și un mesaj, fără date inventate.

    npm install
    npm run dev      # dezvoltare
    npm run build    # produce dist/ (static, merge pe GitHub Pages / Netlify)

| Funcție în api.js      | Ce alimentează în interfață                         |
|------------------------|------------------------------------------------------|
| listPeriods            | dropdown-ul „Perioadă analiză”                       |
| getLossZones           | poligoanele roșii + KPI-urile                        |
| getProtectedAreas      | stratul Natura 2000 + intersecția/alertele           |
| getSatelliteLayer      | basemap „Satelit” + comparația Înainte/După          |
