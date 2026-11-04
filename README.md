# Projet Machine Learning — 4 modèles dans une seule app

| Modèle | Type | Cible | Notebook |
|---|---|---|---|
| Téléphone (déployé) | Classification | `etat` | `telephone_classification.ipynb` |
| Car Data | Classification | `Venant_Occasion` | `car_data_classification.ipynb` |
| Taxi trip (déployé) | Régression | `Trip_Price` | `taxi_trip_price_regression.ipynb` |
| Car Dataset | Régression | `Selling_Price` | `car_dataset_regression.ipynb` |

Les notebooks suivent le code vu en classe. Le nombre de voisins du `KNNImputer` est choisi avec `GridSearchCV`.
Chaque notebook sauvegarde `encoders`, `uniques`, `scaler` et `rf_model` (`.joblib`), rangés dans `models/<nom>/`.

`app.py` (Streamlit) regroupe les 4 modèles ; chacun a une interface **simple** et une interface **multiple** (csv).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Déploiement : pousser ce dépôt sur GitHub, puis sur https://share.streamlit.io → *New app* → choisir le dépôt, la branche `main` et le fichier `app.py`.
