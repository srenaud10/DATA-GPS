# Application de rapport événementiel GPS

## Lancement

```bash
pip install -r requirements.txt
streamlit run streamlit_event_report.py
```

## Données attendues

Le CSV doit contenir (au minimum) des colonnes correspondant à :

- joueur
- temps
- position X
- position Y
- vitesse (km/h)

L'app permet de mapper manuellement les colonnes si les noms diffèrent.
Tu peux importer **plusieurs CSV** (un par joueur).

## Fonctionnalités

- Heatmap de position sur terrain.
- Heatmap par zones (temps passé dans les zones du terrain).
- Détection des sprints avec seuil configurable (25 km/h par défaut).
- Flèches de direction pour les événements de sprint (couleur = intensité, largeur = distance).
- Liste déroulante pour sélectionner le joueur à étudier.
- Conversion vitesse (m/s -> km/h) et remise à l'échelle automatique des coordonnées X/Y.
- Tableau résumé multi-joueurs exportable en CSV.
