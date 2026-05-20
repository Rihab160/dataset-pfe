"""
═══════════════════════════════════════════════════════════════════════
API Backend Flask — PFE AWS CPU Monitor
───────────────────────────────────────────────────────────────────────
Charge les meilleurs modèles exportés (Phase 3 et Phase 4) et expose
des endpoints pour la prédiction CPU et la détection d'anomalies.

Endpoints :
  GET  /                       → statut de l'API
  GET  /servers                → liste des CSVs disponibles
  GET  /cpu/metrics            → données CPU temps réel (CSV → JSON)
  POST /predict                → prédiction XGBoost sur données envoyées
  POST /detect                 → détection anomalies sur données envoyées
  POST /pipeline               → pipeline complet : Phase2 + Phase3 + Phase4

Lancement :
  pip install flask flask-cors pandas numpy scikit-learn xgboost joblib
  python api_server.py
═══════════════════════════════════════════════════════════════════════
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os, glob, joblib, json
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR                                # dossier des modèles .pkl
CSV_DIR   = BASE_DIR                                # dossier des CSVs

FEAT_XGB = ["lag_1","lag_2","moyenne_mobile_5","rolling_std_5",
            "diff_1","weekday_cos","net_io_proxy","disk_io_proxy"]
AFEATS   = ["value","rolling_std_5","diff_1"]
CONT     = 0.05
SEUIL    = 0.60


# ═══════════════════════════════════════════════════════════════════
# CHARGEMENT DES MEILLEURS MODÈLES (Phase 3 et Phase 4 exportés)
# ═══════════════════════════════════════════════════════════════════
MODELS = {"xgboost": None, "metadata": {}}

def charger_modeles():
    """Charge les modèles sauvegardés depuis les notebooks Phase 3 et Phase 4."""
    # Modèle XGBoost (Phase 3)
    for nom in ["best_model.pkl", "xgboost_best.pkl", "model_xgb.pkl"]:
        chemin = os.path.join(MODEL_DIR, nom)
        if os.path.exists(chemin):
            MODELS["xgboost"] = joblib.load(chemin)
            print(f"[OK] Modèle XGBoost chargé depuis {nom}")
            break

    # Métadonnées (MAE baseline, features, hyperparamètres)
    for nom in ["metadata.json", "model_metadata.json"]:
        chemin = os.path.join(MODEL_DIR, nom)
        if os.path.exists(chemin):
            with open(chemin) as f:
                MODELS["metadata"] = json.load(f)
            print(f"[OK] Métadonnées chargées depuis {nom}")
            break

    if MODELS["xgboost"] is None:
        print("[WARN] Aucun modèle XGBoost trouvé — un modèle sera entraîné à la volée")


# ═══════════════════════════════════════════════════════════════════
# PHASE 2 — DOUBLE PIPELINE (appelée si fichier brut)
# ═══════════════════════════════════════════════════════════════════
def _normalise(df, nom):
    """Normalise les colonnes et value en [0,1]. Tolérant aux variations."""
    df = df.copy()

    # Renommage des colonnes alternatives
    renamings = [
        ("cpu","value"), ("server_id","serveur_id"), ("host","serveur_id"),
        ("time","timestamp"), ("date","timestamp"), ("datetime","timestamp"),
        ("ts","timestamp")
    ]
    for old, new in renamings:
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    print(f"[_normalise] Colonnes après renommage : {list(df.columns)}")

    # Colonne timestamp obligatoire — essayer de la déduire si absente
    if "timestamp" not in df.columns:
        # Chercher une colonne qui ressemble à un timestamp
        for c in df.columns:
            sample = df[c].iloc[0] if len(df) > 0 else None
            if sample and any(sep in str(sample) for sep in [":", "-", "/"]):
                df = df.rename(columns={c: "timestamp"})
                print(f"[_normalise] Colonne '{c}' renommée en 'timestamp'")
                break

    if "timestamp" not in df.columns:
        raise ValueError(f"Colonne timestamp manquante. Colonnes reçues : {list(df.columns)}")

    # serveur_id
    if "serveur_id" not in df.columns:
        df["serveur_id"] = nom.replace(".csv","").replace("ec2_cpu_utilization_","")

    # value
    if "value" not in df.columns:
        nc = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
              and c not in ["timestamp","serveur_id"]]
        if nc:
            df = df.rename(columns={nc[0]: "value"})
        else:
            raise ValueError(f"Aucune colonne numérique trouvée pour 'value'. "
                            f"Colonnes : {list(df.columns)}")

    # Conversion timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_avant = len(df)
    df = df.dropna(subset=["timestamp"])
    if len(df) < n_avant:
        print(f"[_normalise] {n_avant - len(df)} lignes avec timestamp invalide supprimées")

    # Conversion value en float et normalisation
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    if df["value"].max() > 1.5:
        df["value"] = df["value"] / 100.0
    df["value"] = df["value"].clip(0.02, 0.98)

    df = df.sort_values(["serveur_id","timestamp"]).reset_index(drop=True)
    return df

def build_df_features(df_base):
    """Pipeline ML : log1p + features temporelles pour XGBoost."""
    df = df_base.copy()
    sc = "serveur_id"
    df["value_log"]        = np.log1p(df["value"])
    df["rolling_std_5"]    = df.groupby(sc)["value_log"].transform(
        lambda x: x.rolling(5, min_periods=1).std()).fillna(0)
    df["diff_1"]           = df.groupby(sc)["value_log"].diff(1).fillna(0)
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_{lag}"]   = df.groupby(sc)["value_log"].shift(lag)
    df["moyenne_mobile_5"] = df.groupby(sc)["value_log"].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    df["target"]           = df.groupby(sc)["value_log"].shift(-1)
    df["weekday_cos"]      = np.cos(2*np.pi*pd.to_datetime(df["timestamp"]).dt.weekday/7)
    # Proxies multivariés
    np.random.seed(42)
    cpu_v = df["value"].values; n = len(cpu_v)
    lag_v = np.roll(cpu_v, 1); lag_v[0] = cpu_v[0]
    net   = np.clip(0.45*cpu_v+0.20*lag_v+np.random.normal(0,.20,n), 0, None)
    win   = pd.Series(cpu_v).rolling(5, min_periods=1).mean().values
    dsk   = np.clip(0.30*cpu_v+0.20*win+np.random.normal(0,.25,n), 0, None)
    for col, vals in [("net_io_proxy",net), ("disk_io_proxy",dsk)]:
        mn, mx = vals.min(), vals.max()
        df[col] = (vals - mn) / (mx - mn + 1e-8)
    return df

def build_df_original(df_base):
    """Pipeline anomalies : valeurs brutes + rolling_std_5 + diff_1."""
    df = df_base.copy()
    sc = "serveur_id"
    df["rolling_std_5"] = df.groupby(sc)["value"].transform(
        lambda x: x.rolling(5, min_periods=1).std()).fillna(0)
    df["diff_1"] = df.groupby(sc)["value"].diff(1).fillna(0)
    return df


# ═══════════════════════════════════════════════════════════════════
# PHASE 3 — PRÉDICTION XGBOOST
# ═══════════════════════════════════════════════════════════════════
def predire_xgboost(df_feat):
    """Prédit le CPU avec le modèle XGBoost chargé (ou entraîne si absent)."""
    from sklearn.metrics import mean_absolute_error, r2_score
    from xgboost import XGBRegressor

    feats = [f for f in FEAT_XGB if f in df_feat.columns]
    if not feats or "target" not in df_feat.columns:
        return None, 0.0, 0.0

    mask = df_feat["target"].notna()
    X    = df_feat[feats].fillna(0).values
    y    = df_feat["target"].values
    if mask.sum() < 20:
        return None, 0.0, 0.0

    # Utiliser le modèle pré-entraîné si disponible et compatible
    model = MODELS["xgboost"]
    if model is None:
        model = XGBRegressor(n_estimators=100, learning_rate=0.05,
                             max_depth=4, random_state=42, verbosity=0)
        model.fit(X[mask], y[mask])

    pred_log  = np.clip(model.predict(X), 0, None)
    pred_vals = np.clip(np.expm1(pred_log), 0, 1)
    mae = float(mean_absolute_error(y[mask], pred_log[mask]))
    try:    r2 = float(r2_score(y[mask], pred_log[mask]))
    except: r2 = 0.0
    return pred_vals.tolist(), mae, r2


def predire_futur(df_feat, n_steps):
    """
    Rolling forecast XGBoost avec recalcul complet des features à chaque pas.

    Principe : la fenêtre glissante d'historique est étendue à chaque pas
    avec la prédiction précédente, puis TOUTES les features sont recalculées
    sur cette nouvelle fenêtre — y compris rolling_std_5, diff_1, moyenne_mobile_5.

    Cela permet à XGBoost de capter l'évolution dynamique des features
    plutôt que de prédire toujours la même valeur avec des features figées.

    Pas 1 : lags + features sur historique réel → p1
    Pas 2 : lags décalés (lag_1=p1) + features recalculées sur hist+p1 → p2
    Pas k : lags décalés + features recalculées sur hist+p1+...+p(k-1) → pk
    """
    from xgboost import XGBRegressor

    feats = [f for f in FEAT_XGB if f in df_feat.columns]
    if not feats or "target" not in df_feat.columns or n_steps < 1:
        return []

    mask = df_feat["target"].notna()
    X    = df_feat[feats].fillna(0).values
    y    = df_feat["target"].values
    if mask.sum() < 20:
        return []

    model = MODELS["xgboost"]
    if model is None:
        model = XGBRegressor(n_estimators=100, learning_rate=0.05,
                             max_depth=4, random_state=42, verbosity=0)
        model.fit(X[mask], y[mask])

    # Historique en espace log (étendu à chaque itération)
    hist_log = df_feat["value_log"].fillna(0).tolist()
    last_ts  = pd.to_datetime(df_feat["timestamp"].iloc[-1])

    if len(hist_log) < 10:
        return []

    pred_futur = []

    for k in range(n_steps):
        # ── Lags : valeurs passées (incluant prédictions précédentes)
        row = {}
        for lag_k in [1, 2, 3, 5, 10]:
            col = f"lag_{lag_k}"
            if col in feats:
                idx_h = len(hist_log) - lag_k
                row[col] = hist_log[idx_h] if idx_h >= 0 else 0.0

        # ── Features dynamiques recalculées sur la fenêtre glissante
        # rolling_std_5 sur les 5 derniers points (réels + prédits)
        if "rolling_std_5" in feats:
            window = hist_log[-5:]
            row["rolling_std_5"] = float(np.std(window)) if len(window) >= 2 else 0.0

        # diff_1 : variation par rapport au dernier point
        if "diff_1" in feats:
            row["diff_1"] = hist_log[-1] - hist_log[-2] if len(hist_log) >= 2 else 0.0

        # moyenne_mobile_5 : moyenne glissante sur les 5 derniers
        if "moyenne_mobile_5" in feats:
            window = hist_log[-5:]
            row["moyenne_mobile_5"] = float(np.mean(window)) if window else 0.0

        # net_io_proxy et disk_io_proxy : proxies recalculés sur fenêtre glissante
        if "net_io_proxy" in feats:
            window_val = [float(np.expm1(v)) for v in hist_log[-5:]]
            row["net_io_proxy"] = float(np.mean(window_val)) if window_val else 0.0
        if "disk_io_proxy" in feats:
            row["disk_io_proxy"] = row.get("net_io_proxy", 0.0)

        # weekday_cos : timestamp futur
        if "weekday_cos" in feats:
            ts_next            = last_ts + pd.Timedelta(minutes=5*(k+1))
            row["weekday_cos"] = float(np.cos(2*np.pi*ts_next.weekday()/7))

        # ── Prédiction XGBoost
        X_row = np.array([row.get(f, 0.0) for f in feats]).reshape(1, -1)
        p_log = float(np.clip(model.predict(X_row)[0], 0, None))
        p_cpu = float(np.clip(np.expm1(p_log), 0, 1))
        pred_futur.append(p_cpu)

        # ── EXTENSION de la fenêtre glissante avec la prédiction
        # C'est ce qui permet aux features rolling/diff/moy d'évoluer
        # au pas suivant — sans cela elles resteraient figées
        new_val = (0.7 * hist_log[-1]+ 0.3 * p_log + np.random.normal(0, 0.01))
        hist_log.append(new_val)

    return pred_futur


# ═══════════════════════════════════════════════════════════════════
# PHASE 4 — DÉTECTION D'ANOMALIES
# ═══════════════════════════════════════════════════════════════════
def detecter_anomalies(df_orig):
    """Applique IF + LOF + SVM avec split temporel 70/30 et score de fusion."""
    fo = [f for f in AFEATS if f in df_orig.columns]
    n  = len(df_orig)
    if not fo:
        return {"IF":[], "LOF":[], "SVM":[]}, [], [], []

    X  = df_orig[fo].fillna(df_orig[fo].median()).fillna(0).values
    sc = StandardScaler()
    split = int(n * 0.7)
    Xtr   = sc.fit_transform(X[:split])
    Xall  = np.vstack([Xtr, sc.transform(X[split:])])

    cpu_v = df_orig["value"].values * 100
    iqr   = float(np.percentile(cpu_v, 75) - np.percentile(cpu_v, 25))
    cont  = 0.08 if iqr < 5.0 else CONT

    # Isolation Forest
    mdl_if = IsolationForest(contamination=cont, random_state=42, n_estimators=100)
    mdl_if.fit(Xtr)
    lab_if = (mdl_if.predict(Xall) == -1).astype(int)

    # LOF (novelty=True pour fit/predict séparés)
    mdl_lof = LocalOutlierFactor(n_neighbors=min(20, split-1),
                                  contamination=cont, novelty=True)
    mdl_lof.fit(Xtr)
    lab_lof = (mdl_lof.predict(Xall) == -1).astype(int)

    # One-Class SVM
    mdl_svm = OneClassSVM(nu=cont, kernel="rbf", gamma="scale")
    mdl_svm.fit(Xtr)
    lab_svm = (mdl_svm.predict(Xall) == -1).astype(int)

    nb    = lab_if + lab_lof + lab_svm
    score = 0.6 * (nb / 3)
    conf  = (score >= SEUIL).astype(int)

    return {"IF":lab_if.tolist(), "LOF":lab_lof.tolist(), "SVM":lab_svm.tolist()}, \
           nb.tolist(), score.tolist(), conf.tolist()


# ═══════════════════════════════════════════════════════════════════
# UTILITAIRE — Recherche de fichier CSV
# ═══════════════════════════════════════════════════════════════════
def trouver_csv(nom_fichier):
    chemin = os.path.join(CSV_DIR, nom_fichier)
    if os.path.exists(chemin): return chemin
    matches = glob.glob(os.path.join(CSV_DIR, "**", nom_fichier), recursive=True)
    return matches[0] if matches else None


# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Statut de l'API et endpoints disponibles."""
    return jsonify({
        "service":  "PFE AWS CPU Monitor — API Backend",
        "status":   "ok",
        "models":   {"xgboost": MODELS["xgboost"] is not None},
        "metadata": MODELS["metadata"],
        "endpoints": {
            "/servers"     : "GET — Liste des CSVs disponibles",
            "/cpu/metrics" : "GET ?file=... — Données CPU temps réel",
            "/predict"     : "POST {data} — Prédiction XGBoost",
            "/detect"      : "POST {data} — Détection anomalies",
            "/pipeline"    : "POST {data} — Pipeline complet"
        }
    })


@app.route("/servers")
def servers():
    """Liste tous les CSVs disponibles dans CSV_DIR."""
    csvs = glob.glob(os.path.join(CSV_DIR, "**", "*.csv"), recursive=True)
    return jsonify([os.path.basename(f) for f in csvs])


@app.route("/cpu/metrics")
def metrics():
    """
    Renvoie les données CPU d'un fichier CSV en simulation temps réel.
    Les timestamps sont décalés pour que le dernier point = maintenant.
    La fenêtre d'affichage est contrôlée par le dashboard, pas par l'API.
    """
    fichier = request.args.get("file", "ec2_cpu_utilization_77c1ca.csv")
    chemin  = trouver_csv(fichier)
    if chemin is None:
        return jsonify({
            "error": f"Fichier '{fichier}' introuvable",
            "disponibles": [os.path.basename(f) for f in
                glob.glob(os.path.join(CSV_DIR, "**", "*.csv"), recursive=True)]
        }), 404

    try:
        df = pd.read_csv(chemin)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Simulation temps réel : recaler ts_max sur maintenant
        now   = pd.Timestamp.utcnow().tz_localize(None)
        shift = now - df["timestamp"].max()
        df["timestamp"] = df["timestamp"] + shift

        # Normalisation value
        if "value" not in df.columns:
            nc = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                  and c not in ["timestamp","serveur_id"]]
            if nc: df = df.rename(columns={nc[0]:"value"})
        if "value" in df.columns and df["value"].max() > 1.5:
            df["value"] = df["value"] / 100.0

        if "serveur_id" not in df.columns:
            srv = fichier.replace(".csv","").replace("ec2_cpu_utilization_","")
            df["serveur_id"] = srv

        df["timestamp"] = df["timestamp"].astype(str)
        return jsonify(df[["timestamp","value","serveur_id"]].to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Reçoit des données JSON brutes (timestamp + value) et renvoie
    les prédictions XGBoost calculées via le modèle chargé.
    """
    try:
        data = request.get_json()
        df_raw = pd.DataFrame(data)
        df_base = _normalise(df_raw, "api_predict")
        df_feat = build_df_features(df_base)

        pred_vals, mae, r2 = predire_xgboost(df_feat)
        return jsonify({
            "predictions": pred_vals,
            "mae": mae,
            "r2": r2,
            "model": "XGBoost",
            "features": [f for f in FEAT_XGB if f in df_feat.columns]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/detect", methods=["POST"])
def detect():
    """
    Reçoit des données JSON brutes (timestamp + value) et renvoie
    les anomalies détectées par IF + LOF + SVM avec score de fusion.
    """
    try:
        data = request.get_json()
        df_raw = pd.DataFrame(data)
        df_base = _normalise(df_raw, "api_detect")
        df_orig = build_df_original(df_base)

        labs, nb, score, conf = detecter_anomalies(df_orig)
        return jsonify({
            "labels"   : labs,
            "consensus": nb,
            "score"    : score,
            "confirmed": conf,
            "n_anomalies": int(sum(conf)),
            "models"   : ["IsolationForest", "LOF", "OneClassSVM"],
            "seuil"    : SEUIL
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pipeline", methods=["POST"])
def pipeline():
    """
    Pipeline complet : Phase2 + Phase3 + Phase4 sur les données reçues.
    """
    import traceback
    try:
        body = request.get_json(force=True, silent=False)
        if body is None:
            return jsonify({"error": "Body JSON vide ou invalide"}), 400

        # Accepter les deux formats : liste directe OU dict avec data + n_steps_future
        if isinstance(body, list):
            data = body
            n_steps_future = 0
        else:
            data = body.get("data", [])
            n_steps_future = int(body.get("n_steps_future", 0))

        print(f"[/pipeline] Reçu {len(data)} points, n_steps_future={n_steps_future}")

        if not data:
            return jsonify({"error": "Aucune donnée reçue"}), 400

        df_raw = pd.DataFrame(data)
        print(f"[/pipeline] Colonnes brutes : {list(df_raw.columns)}")

        df_base = _normalise(df_raw, "api_pipeline")
        print(f"[/pipeline] Après normalisation : {len(df_base)} lignes")

        df_feat = build_df_features(df_base)
        df_orig = build_df_original(df_base)

        # Prédictions sur l'historique
        pred_vals, mae, r2 = predire_xgboost(df_feat)
        print(f"[/pipeline] Prédiction historique : MAE={mae:.4f}, R²={r2:.4f}")

        # Prédictions futures pas à pas
        pred_future = predire_futur(df_feat, n_steps_future) if n_steps_future > 0 else []
        print(f"[/pipeline] Prédictions futures : {len(pred_future)} pts")

        # Détection d'anomalies
        labs, nb, score, conf = detecter_anomalies(df_orig)
        print(f"[/pipeline] Anomalies détectées : {sum(conf)}")

        return jsonify({
            "prediction": {
                "values"     : pred_vals if pred_vals else [],
                "future"     : pred_future,
                "mae"        : mae,
                "r2"         : r2,
                "model"      : "XGBoost"
            },
            "detection": {
                "labels"     : labs,
                "consensus"  : nb,
                "score"      : score,
                "confirmed"  : conf,
                "n_anomalies": int(sum(conf)),
                "seuil"      : SEUIL
            }
        })
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)}"
        tb      = traceback.format_exc()
        print(f"[/pipeline] ERREUR : {err_msg}")
        print(tb)
        return jsonify({"error": err_msg, "traceback": tb.split(chr(10))[-5:]}), 500


# ═══════════════════════════════════════════════════════════════════
# LANCEMENT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 60)
    print("  PFE AWS CPU Monitor — API Backend")
    print("═" * 60)
    print(f"  Dossier modèles : {MODEL_DIR}")
    print(f"  Dossier CSVs    : {CSV_DIR}")
    print("─" * 60)
    charger_modeles()
    print("─" * 60)
    print("  Endpoints disponibles :")
    print("    GET  /                     → statut API")
    print("    GET  /servers              → liste CSVs")
    print("    GET  /cpu/metrics?file=... → données temps réel")
    print("    POST /predict              → prédiction XGBoost")
    print("    POST /detect               → détection anomalies")
    print("    POST /pipeline             → pipeline complet")
    print("═" * 60)
    print("  Démarrage sur http://127.0.0.1:5000")
    print("═" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)