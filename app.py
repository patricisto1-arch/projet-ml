"""Application Streamlit : 4 modèles (2 classification, 2 régression).
Chaque modèle a 2 interfaces : prédiction simple et prédiction multiple (csv).
Même logique que le code du cours : encoders, scaler, best modèle, uniques."""
import html
import os

import joblib as jb
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Projet ML — 4 modèles", page_icon="🤖", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {background: linear-gradient(135deg, #0f766e 0%, #0e7490 60%, #1d4ed8 100%); padding: 1.6rem 2rem;
       border-radius: 16px; margin-bottom: 1.2rem; box-shadow: 0 4px 14px rgba(0,0,0,.18);}
.hero-title {color: #ffffff; font-size: 2rem; font-weight: 800; line-height: 1.2;}
.hero-sub {color: #e6fffb; font-size: 1.02rem; margin-top: .35rem;}
.badge {display: inline-block; padding: .22rem .75rem; border-radius: 999px; font-size: .82rem;
        font-weight: 700; margin-right: .45rem;}
.badge-clf {background: #dbeafe; color: #1e3a8a;}
.badge-reg {background: #dcfce7; color: #14532d;}
.badge-cible {background: #e2e8f0; color: #0f172a;}
.result {background: linear-gradient(135deg, #0f766e, #14b8a6); border-radius: 16px; padding: 1.4rem 1.8rem;
         text-align: center; margin-top: 1rem; box-shadow: 0 4px 14px rgba(0,0,0,.18);}
.result-lab {color: #e6fffb; font-size: .95rem; letter-spacing: .06em; text-transform: uppercase;}
.result-val {color: #ffffff; font-size: 2.9rem; font-weight: 800; line-height: 1.15; margin-top: .2rem;}
button[data-baseweb="tab"] {font-size: 1.02rem; font-weight: 600;}
div[data-testid="stForm"] {border-radius: 14px; padding: 1.2rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@st.cache_resource
def charger(nom):
    d = os.path.join(BASE, nom)
    return (jb.load(os.path.join(d, "encoders.joblib")), jb.load(os.path.join(d, "rf_model.joblib")),
            jb.load(os.path.join(d, "uniques.joblib")), jb.load(os.path.join(d, "scaler.joblib")))


# Description de chaque modèle : colonnes dans l'ordre d'entraînement, variables catégorielles
# (nom -> position de l'encodeur), widgets du formulaire (type, valeur par défaut ou None).
MODELES = {
    "telephone": dict(
        onglet="📱 Téléphone", type="classification", cible="etat", unite="", target_enc=2,
        titre="Prédire l'état d'un téléphone (Occasion, Neuf, Reconditionné, Venant)",
        colonnes=["prix", "adresse", "marque", "dim_ecr", "ram", "stockage"],
        cats={"adresse": 0, "marque": 1},
        defauts={"prix": 300000.0, "dim_ecr": 6.0, "ram": 8.0, "stockage": 128.0}),
    "car_data": dict(
        onglet="🚗 Car Data", type="classification", cible="Venant_Occasion", unite="", target_enc=3,
        titre="Prédire si une voiture est « Venant » ou « D'occasion »",
        colonnes=["Marque", "Année", "Boite_Vitesse", "Quartier", "Prix"],
        cats={"Marque": 0, "Boite_Vitesse": 1, "Quartier": 2},
        defauts={"Année": 2015.0, "Prix": 8000000.0}),
    "taxi_trip": dict(
        onglet="🚕 Taxi trip", type="regression", cible="Trip_Price", unite="$",
        titre="Prédire le prix d'une course de taxi",
        colonnes=["Trip_Distance_km", "Time_of_Day", "Day_of_Week", "Passenger_Count", "Traffic_Conditions",
                  "Weather", "Base_Fare", "Per_Km_Rate", "Per_Minute_Rate", "Trip_Duration_Minutes"],
        cats={"Time_of_Day": 0, "Day_of_Week": 1, "Traffic_Conditions": 2, "Weather": 3},
        defauts={"Trip_Distance_km": 20.0, "Passenger_Count": 2.0, "Base_Fare": 3.0, "Per_Km_Rate": 1.0,
                 "Per_Minute_Rate": 0.3, "Trip_Duration_Minutes": 30.0}),
    "car_dataset": dict(
        onglet="🚙 Car Dataset", type="regression", cible="Selling_Price", unite="",
        titre="Prédire le prix de vente d'une voiture",
        colonnes=["Kms_Driven", "Present_Price", "Fuel_Type", "Seller_Type", "Transmission", "Age"],
        cats={"Fuel_Type": 0, "Seller_Type": 1, "Transmission": 2},
        defauts={"Kms_Driven": 50000.0, "Present_Price": 10.0, "Age": 5.0}),
}


@st.cache_data
def valeurs_par_defaut(nom):
    """Valeurs de remplacement des cases vides : mode (catégorielles) et médiane (numériques),
    calculés sur les données d'entraînement."""
    m = MODELES[nom]
    ref = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"{nom}.csv"))
    return {c: (ref[c].mode()[0] if c in m["cats"] else float(ref[c].median())) for c in m["colonnes"]}


def predire(nom, valeurs):
    """Prédit pour une ligne (dict colonne -> valeur), comme Pred_func du cours."""
    m = MODELES[nom]
    encoders, model, uniques, scaler = charger(nom)
    x_new = []
    for col in m["colonnes"]:
        if col in m["cats"]:
            x_new.append(encoders[m["cats"][col]].transform([valeurs[col]])[0])
        else:
            x_new.append(valeurs[col])
    x_new = scaler.transform(np.array(x_new, dtype=float).reshape(1, -1))
    y_pred = model.predict(x_new)
    if m["type"] == "classification":
        return encoders[m["target_enc"]].inverse_transform([int(y_pred[0])])[0]
    return round(float(y_pred[0]), 2)


def carte_resultat(libelle, valeur):
    """Grande carte colorée (lisible en thème clair et sombre) pour afficher la prédiction."""
    st.markdown(f'<div class="result"><div class="result-lab">{html.escape(str(libelle))}</div>'
                f'<div class="result-val">{html.escape(str(valeur))}</div></div>', unsafe_allow_html=True)


def badges(nom):
    m = MODELES[nom]
    clf = m["type"] == "classification"
    st.markdown(f'<span class="badge {"badge-clf" if clf else "badge-reg"}">'
                f'{"Classification" if clf else "Régression"}</span>'
                f'<span class="badge badge-cible">Cible : {html.escape(m["cible"])}</span>', unsafe_allow_html=True)


@st.cache_data
def exemple_csv(nom):
    """Petit fichier csv d'exemple (5 lignes du jeu de données) au bon format."""
    m = MODELES[nom]
    ref = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"{nom}.csv"))
    if nom == "car_data":
        ref["Quartier"] = ref["Adresse"].astype(str).str.split(",").str[0].str.strip()
    return ref[m["colonnes"]].head(5).to_csv(index=False).encode("utf-8")


def interface_simple(nom):
    m = MODELES[nom]
    uniques = charger(nom)[2]
    with st.form(f"form_{nom}"):
        cols = st.columns(3)
        valeurs = {}
        for i, col in enumerate(m["colonnes"]):
            with cols[i % 3]:
                if col in m["cats"]:
                    valeurs[col] = st.selectbox(col, uniques[m["cats"][col]], key=f"{nom}_{col}")
                else:
                    valeurs[col] = st.number_input(col, value=m["defauts"][col], key=f"{nom}_{col}")
        ok = st.form_submit_button("🔮 Prédire", type="primary", width="stretch")
    if ok:
        res = predire(nom, valeurs)
        carte_resultat(f"{m['cible']} prédit", f"{res}{m['unite']}")


def interface_multiple(nom):
    m = MODELES[nom]
    st.write("Importez un fichier csv contenant les colonnes : `" + "`, `".join(m["colonnes"]) + "`")
    st.download_button("📄 Télécharger un exemple de csv", exemple_csv(nom), file_name=f"exemple_{nom}.csv",
                       mime="text/csv", key=f"ex_{nom}")
    fichier = st.file_uploader("Fichier csv", type=["csv"], key=f"csv_{nom}")
    if fichier is None:
        return
    df = pd.read_csv(fichier)
    # Car Data : si le csv contient « Adresse » (Quartier,Ville), on en déduit le quartier
    if nom == "car_data" and "Quartier" not in df.columns and "Adresse" in df.columns:
        df["Quartier"] = df["Adresse"].astype(str).str.split(",").str[0].str.strip()
    manquantes = [c for c in m["colonnes"] if c not in df.columns]
    if manquantes:
        st.error("Colonnes manquantes : " + ", ".join(manquantes))
        return
    entree = df[m["colonnes"]].copy()
    nb_vides = int(entree.isna().sum().sum())
    if nb_vides:
        entree = entree.fillna(valeurs_par_defaut(nom))
        st.warning(f"{nb_vides} valeur(s) manquante(s) remplacée(s) avant la prédiction : mode pour les "
                   "variables catégorielles, médiane pour les variables numériques.")
    predictions = []
    with st.spinner("Prédiction en cours..."):
        for i, ligne in entree.iterrows():
            try:
                predictions.append(predire(nom, ligne.to_dict()))
            except ValueError as e:
                st.error(f"Ligne {i + 2} : valeur inconnue ({e})")
                return
    col_pred = m["cible"] + "_prediction"
    df[col_pred] = predictions

    # Résumé des prédictions
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes traitées", len(df))
    if m["type"] == "classification":
        comptes = df[col_pred].value_counts()
        c2.metric("Classe la plus fréquente", str(comptes.index[0]))
        c3.metric("Classes prédites", int(comptes.size))
        st.caption("Répartition des prédictions")
        st.bar_chart(comptes)
    else:
        s = df[col_pred].astype(float)
        c2.metric("Moyenne prédite", f"{s.mean():.2f}{m['unite']}")
        c3.metric("Min – Max", f"{s.min():.2f} – {s.max():.2f}")

    # Tableau : colonne de prédiction mise en évidence (couleurs explicites, lisibles en thème clair/sombre)
    st.markdown("**Résultats : prédiction pour chaque ligne**")
    styler = df.style.set_properties(subset=[col_pred], **{"background-color": "#fef08a", "color": "#111827",
                                                           "font-weight": "bold"})
    st.dataframe(styler, width="stretch")
    st.download_button("⬇️ Télécharger les prédictions (csv)", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{nom}.csv", mime="text/csv", key=f"dl_{nom}", type="primary")


with st.sidebar:
    st.header("À propos")
    st.write("Une seule application pour **4 modèles** de machine learning. "
             "Pour chacun : un **formulaire** (prédiction simple) ou un **fichier csv** (prédiction multiple).")
    for n, mm in MODELES.items():
        st.markdown(f"- {mm['onglet']} — {'classification' if mm['type'] == 'classification' else 'régression'}")

st.markdown('<div class="hero"><div class="hero-title">🤖 Projet Machine Learning — 4 modèles</div>'
            '<div class="hero-sub">Classification : téléphones, voitures · Régression : taxi, voitures</div></div>',
            unsafe_allow_html=True)
onglets = st.tabs([MODELES[n]["onglet"] for n in MODELES])
for onglet, nom in zip(onglets, MODELES):
    with onglet:
        st.subheader(MODELES[nom]["titre"])
        badges(nom)
        simple, multiple = st.tabs(["✍️ Prédiction simple (formulaire)", "📁 Prédiction multiple (csv)"])
        with simple:
            interface_simple(nom)
        with multiple:
            interface_multiple(nom)
