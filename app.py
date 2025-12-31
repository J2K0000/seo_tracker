#!/usr/bin/env python3
"""
SEO Tracker Multi-Sites - Version Streamlit
"""

import os
import io
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor, as_completed

# Charger les variables d'environnement
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configuration
LANGUAGE_CODE = os.getenv("DEFAULT_LANGUAGE_CODE", "fr")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))

# API Endpoints
API_URL = "https://api.dataforseo.com/v3/serp/google/organic/live/regular"
SEARCH_VOLUME_API_URL = "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live"


def get_auth_header(login, password):
    """Génère le header d'authentification pour DataForSEO."""
    credentials = f"{login}:{password}"
    encoded = b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}", "Content-Type": "application/json"}


def extract_domain(url_or_domain):
    """Extrait le domaine principal d'une URL ou d'un domaine."""
    if not url_or_domain.startswith(("http://", "https://")):
        url_or_domain = "https://" + url_or_domain

    parsed = urlparse(url_or_domain)
    domain = parsed.netloc or parsed.path.split("/")[0]

    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def get_search_volumes(keywords, headers, location_code):
    """Récupère les volumes de recherche pour une liste de mots-clés."""
    volumes = {kw: None for kw in keywords}

    payload = [
        {
            "keywords": keywords,
            "location_code": location_code,
            "language_code": LANGUAGE_CODE,
        }
    ]

    try:
        response = requests.post(SEARCH_VOLUME_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()

        if data.get("status_code") != 20000:
            return volumes

        tasks = data.get("tasks", [])
        if not tasks or not tasks[0].get("result"):
            return volumes

        for item in tasks[0]["result"]:
            keyword = item.get("keyword")
            volume = item.get("search_volume")
            if keyword:
                volumes[keyword] = volume

    except Exception:
        pass

    return volumes


def get_keyword_positions_multi(keyword, target_domains, headers, location_code):
    """Recherche les positions de plusieurs domaines pour un mot-clé donné."""
    payload = [
        {
            "keyword": keyword,
            "location_code": location_code,
            "language_code": LANGUAGE_CODE,
            "depth": 100,
        }
    ]

    results = {domain: (None, None) for domain in target_domains}

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("status_code") != 20000:
            return keyword, results

        tasks = data.get("tasks", [])
        if not tasks or not tasks[0].get("result"):
            return keyword, results

        items = tasks[0]["result"][0].get("items", [])

        for item in items:
            if item.get("type") == "organic":
                item_domain = item.get("domain", "").lower()
                for target_domain in target_domains:
                    if target_domain.lower() in item_domain and results[target_domain][0] is None:
                        results[target_domain] = (item.get("rank_group"), item.get("url", ""))

        return keyword, results

    except Exception:
        return keyword, results


def get_positions_parallel_multi(keywords, target_domains, headers, location_code, progress_bar, status_text):
    """Recherche les positions en parallèle pour plusieurs domaines."""
    results = {}
    completed = 0
    total = len(keywords)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_keyword_positions_multi, kw, target_domains, headers, location_code): kw
            for kw in keywords
        }

        for future in as_completed(futures):
            keyword, positions = future.result()
            results[keyword] = positions
            completed += 1

            progress_bar.progress(completed / total)
            status_text.text(f"Analyse des positions: {completed}/{total} mots-clés")

    return results


def build_dataframe(keywords, sites, positions_dict, volumes_dict):
    """Construit un DataFrame avec les résultats."""
    data = []
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    for kw in keywords:
        row = {
            "Mot-clé": kw,
            "Volume": volumes_dict.get(kw) if volumes_dict.get(kw) is not None else "N/A"
        }

        pos_dict = positions_dict.get(kw, {})
        for site in sites:
            pos, url = pos_dict.get(site, (None, None))
            row[f"Position ({site})"] = pos if pos else ">100"
            row[f"URL ({site})"] = url if url else ""

        row["Date"] = date_str
        data.append(row)

    return pd.DataFrame(data)


def convert_df_to_csv(df):
    """Convertit un DataFrame en CSV pour le téléchargement."""
    return df.to_csv(index=False, sep=";").encode("utf-8")


# ============ INTERFACE STREAMLIT ============

st.set_page_config(
    page_title="SEO Tracker Multi-Sites",
    page_icon="📊",
    layout="wide"
)

st.title("📊 SEO Tracker Multi-Sites")
st.markdown("Relevé de positions pour un ou plusieurs sites via DataForSEO API")

# Récupérer les credentials (priorité: Streamlit Secrets > .env)
def get_credentials():
    """Récupère les credentials depuis Streamlit Secrets ou .env"""
    login = None
    password = None

    # Essayer Streamlit Secrets d'abord (pour déploiement cloud)
    try:
        login = st.secrets.get("DATAFORSEO_LOGIN")
        password = st.secrets.get("DATAFORSEO_PASSWORD")
    except Exception:
        pass

    # Sinon, utiliser les variables d'environnement (.env local)
    if not login:
        login = os.getenv("DATAFORSEO_LOGIN", "")
    if not password:
        password = os.getenv("DATAFORSEO_PASSWORD", "")

    return login, password

default_login, default_password = get_credentials()
credentials_configured = bool(default_login and default_password)

# Sidebar pour les paramètres
with st.sidebar:
    st.header("Configuration API")

    if credentials_configured:
        st.success("API configurée via Secrets")
        api_login = default_login
        api_password = default_password
    else:
        st.warning("API non configurée")
        api_login = st.text_input("DataForSEO Login", type="default")
        api_password = st.text_input("DataForSEO Password", type="password")

    st.divider()
    st.header("Paramètres")

    # Liste complète des pays supportés par DataForSEO
    location_codes = {
        # Europe francophone
        "France": 2250,
        "Belgique": 2056,
        "Suisse": 2756,
        "Luxembourg": 2442,
        "Monaco": 2492,
        # Europe
        "Allemagne": 2276,
        "Royaume-Uni": 2826,
        "Espagne": 2724,
        "Italie": 2380,
        "Portugal": 2620,
        "Pays-Bas": 2528,
        "Autriche": 2040,
        "Pologne": 2616,
        "Suède": 2752,
        "Norvège": 2578,
        "Danemark": 2208,
        "Finlande": 2246,
        "Irlande": 2372,
        "Grèce": 2300,
        "République Tchèque": 2203,
        "Roumanie": 2642,
        "Hongrie": 2348,
        # Amérique du Nord
        "États-Unis": 2840,
        "Canada": 2124,
        "Mexique": 2484,
        # Amérique du Sud
        "Brésil": 2076,
        "Argentine": 2032,
        "Colombie": 2170,
        "Chili": 2152,
        "Pérou": 2604,
        # Asie
        "Japon": 2392,
        "Corée du Sud": 2410,
        "Inde": 2356,
        "Singapour": 2702,
        "Hong Kong": 2344,
        "Taïwan": 2158,
        "Thaïlande": 2764,
        "Vietnam": 2704,
        "Indonésie": 2360,
        "Malaisie": 2458,
        "Philippines": 2608,
        # Océanie
        "Australie": 2036,
        "Nouvelle-Zélande": 2554,
        # Afrique
        "Afrique du Sud": 2710,
        "Maroc": 2504,
        "Algérie": 2012,
        "Tunisie": 2788,
        "Égypte": 2818,
        "Nigeria": 2566,
        "Kenya": 2404,
        # Moyen-Orient
        "Émirats Arabes Unis": 2784,
        "Arabie Saoudite": 2682,
        "Israël": 2376,
        "Turquie": 2792,
    }

    location = st.selectbox("Pays", list(location_codes.keys()), index=0)

# Layout principal en deux colonnes
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sites à analyser")
    st.caption("Un site par ligne (le premier = client, les autres = concurrents)")

    # Charger les sites par défaut depuis le fichier s'il existe
    default_sites = ""
    sites_file = os.path.join(os.path.dirname(__file__), "sites.txt")
    if os.path.exists(sites_file):
        with open(sites_file, "r", encoding="utf-8") as f:
            default_sites = f.read()

    sites_input = st.text_area("Sites", value=default_sites, height=150, placeholder="monsite.fr\nconcurrent1.fr\nconcurrent2.com")

with col2:
    st.subheader("Mots-clés")
    st.caption("Un mot-clé par ligne")

    # Charger les mots-clés par défaut depuis le fichier s'il existe
    default_keywords = ""
    keywords_file = os.path.join(os.path.dirname(__file__), "keywords.txt")
    if os.path.exists(keywords_file):
        with open(keywords_file, "r", encoding="utf-8") as f:
            default_keywords = f.read()

    keywords_input = st.text_area("Mots-clés", value=default_keywords, height=150, placeholder="mot clé 1\nmot clé 2\nmot clé 3")

# Bouton de lancement
st.divider()

if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):

    # Validation des entrées
    if not api_login or not api_password:
        st.error("Veuillez renseigner vos identifiants DataForSEO.")
    elif not sites_input.strip():
        st.error("Veuillez entrer au moins un site à analyser.")
    elif not keywords_input.strip():
        st.error("Veuillez entrer au moins un mot-clé.")
    else:
        # Parser les entrées
        sites = [extract_domain(line.strip()) for line in sites_input.strip().split("\n") if line.strip()]
        keywords = [line.strip() for line in keywords_input.strip().split("\n") if line.strip()]

        st.info(f"Analyse de **{len(keywords)}** mots-clés pour **{len(sites)}** site(s)")

        # Afficher les sites
        with st.expander("Sites analysés"):
            for i, site in enumerate(sites):
                label = "🏠 Client" if i == 0 else "🏢 Concurrent"
                st.write(f"{label}: **{site}**")

        headers = get_auth_header(api_login, api_password)

        # Barre de progression pour les positions
        st.subheader("Progression")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Récupérer le code de localisation sélectionné
        selected_location_code = location_codes[location]

        # Récupérer les positions
        status_text.text("Analyse des positions...")
        positions_dict = get_positions_parallel_multi(keywords, sites, headers, selected_location_code, progress_bar, status_text)

        # Récupérer les volumes
        status_text.text("Récupération des volumes de recherche...")
        volumes_dict = get_search_volumes(keywords, headers, selected_location_code)

        progress_bar.progress(100)
        status_text.text("Analyse terminée !")

        # Construire le DataFrame
        df = build_dataframe(keywords, sites, positions_dict, volumes_dict)

        # Afficher les résultats
        st.divider()
        st.subheader("Résultats")

        # Résumé
        col_summary = st.columns(len(sites))
        for i, site in enumerate(sites):
            with col_summary[i]:
                pos_col = f"Position ({site})"
                found = df[df[pos_col] != ">100"].shape[0]
                st.metric(
                    label=site,
                    value=f"{found}/{len(keywords)}",
                    help="Positions trouvées (top 100)"
                )

        # Tableau des résultats
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Bouton de téléchargement CSV
        csv_data = convert_df_to_csv(df)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        st.download_button(
            label="📥 Télécharger le CSV",
            data=csv_data,
            file_name=f"positions_multi_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True
        )
