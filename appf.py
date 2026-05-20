"""
 Dashboard PFE - AWS CPU Monitor
Pipeline complet : Phase2 + Phase3 + Phase4 sur chaque fichier importe
"""
import streamlit as st
import pandas as pd
import numpy as np
import json, joblib, smtplib, warnings, requests
from datetime import datetime, date, time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import streamlit.components.v1 as components
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PFE - AWS CPU Monitor", page_icon="CPU",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html, body, .stApp { background: var(--color-background-tertiary, #F1EFE8) !important; font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="block-container"] { padding: 0.3rem 1.5rem 2rem !important; max-width: 100% !important; }
[data-testid="stHeader"]  { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.topbar-pill { display:none; }
.topbar-pill-warn { display:none; }
.sec-label { font-size:10px; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:.10em; margin-bottom:12px; margin-top:16px; display:flex; align-items:center; gap:8px; }
.sec-label::before { content:""; display:inline-block; width:3px; height:12px; background:#185FA5; border-radius:2px; }
.card { background:#FFFFFF; border:1px solid #E8ECF0; border-radius:16px; padding:18px 20px; margin-bottom:6px; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.mcard { background:var(--color-background-secondary); border-radius:8px; padding:12px 14px; }
.mlabel { font-size:11px; color:var(--color-text-secondary); margin-bottom:5px; text-transform:uppercase; letter-spacing:.04em; }
.mval { font-size:22px; font-weight:500; color:var(--color-text-primary); }
.msub { font-size:11px; margin-top:3px; }
.divider { height:0.5px; background:var(--color-border-tertiary); margin:18px 0; }
.badge { font-size:10px; padding:2px 8px; border-radius:6px; font-weight:500; white-space:nowrap; }
.badge-ok      { background:#EAF3DE; color:#3B6D11; }
.badge-warn    { background:#FAEEDA; color:#854F0B; }
.badge-crit    { background:#FCEBEB; color:#A32D2D; }
.badge-suspect { background:#F1F0EE; color:#6B6B6B; border:0.5px solid #B4B2A9; }
.arow { display:flex; align-items:center; gap:8px; padding:7px 10px; background:var(--color-background-secondary); border-radius:8px; margin-bottom:4px; }
.dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.tog { display:flex; align-items:center; gap:10px; padding:7px 0; border-bottom:0.5px solid var(--color-border-tertiary); }
.tog:last-of-type { border-bottom:none; }
.tog-label { font-size:12px; color:var(--color-text-primary); flex:1; }
.switch { width:32px; height:17px; border-radius:9px; cursor:pointer; display:flex; align-items:center; padding:2px; flex-shrink:0; transition:background .15s; }
.switch.on { background:#639922; } .switch.off { background:#B4B2A9; }
.knob { width:13px; height:13px; border-radius:50%; background:#fff; transition:transform .15s; }
.switch.on .knob { transform:translateX(15px); }
.stButton > button { background:#185FA5 !important; color:#E6F1FB !important; border:none !important; border-radius:8px !important; font-size:12px !important; font-weight:500 !important; padding:7px 18px !important; }
.stTextInput input, .stNumberInput input { border:0.5px solid var(--color-border-secondary) !important; border-radius:8px !important; font-size:13px !important; background:var(--color-background-primary) !important; color:var(--color-text-primary) !important; }
.stSelectbox [data-baseweb="select"] > div { border-radius:8px !important; font-size:13px !important; }
[data-testid="stFileUploadDropzone"] { border-radius:8px !important; border:0.5px dashed var(--color-border-secondary) !important; }
.footer { display:flex; justify-content:space-between; padding:8px 0; margin-top:16px; border-top:0.5px solid var(--color-border-tertiary); font-size:10px; color:var(--color-text-secondary); }
.welcome { display:flex; flex-direction:column; align-items:center; justify-content:center; padding:60px 20px; text-align:center; }
.welcome h2 { font-size:1.4rem; font-weight:600; color:var(--color-text-primary); margin-bottom:10px; }
.welcome p { font-size:13px; color:var(--color-text-secondary); margin-bottom:24px; max-width:500px; }
</style>
""", unsafe_allow_html=True)
# ── Constantes
AFEATS   = ["value","rolling_std_5","diff_1"]
FEAT_XGB = ["lag_1","lag_2","lag_3","lag_5","lag_10","rolling_std_5","diff_1","hour_cos","weekday_cos"]
CONT  = 0.05
SEUIL = 0.60
# ══ PHASE 2 ══════════════════════════════════════════════
def phase2_features(df, nom="i-import"):
    df = df.copy()
    for old,new in [("cpu","value"),("server_id","serveur_id"),("host","serveur_id")]:
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old:new})
    if "serveur_id" not in df.columns:
        df["serveur_id"] = nom.replace(".csv","").replace("ec2_cpu_utilization_","")
    if "value" not in df.columns:
        nc = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ["timestamp","serveur_id"]]
        if nc: df = df.rename(columns={nc[0]:"value"})
    if "value" in df.columns:
        if df["value"].max() > 1.5:
            df["value"] = df["value"] / 100.0
        df["value"] = df["value"].clip(0, 1)
    ts_candidates = [c for c in df.columns if any(k in c.lower() for k in ["time","date","ts","stamp"])]
    if "timestamp" not in df.columns and ts_candidates:
        df = df.rename(columns={ts_candidates[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["serveur_id","timestamp"]).reset_index(drop=True)
    sc = "serveur_id"
    df["value_log"]        = np.log1p(df["value"])
    df["rolling_std_5"]    = df.groupby(sc)["value_log"].transform(lambda x: x.rolling(5,min_periods=1).std()).fillna(0)
    df["diff_1"]           = df.groupby(sc)["value_log"].diff(1).fillna(0)
    for lag in [1,2,3,5,10]: df[f"lag_{lag}"] = df.groupby(sc)["value_log"].shift(lag)
    df["moyenne_mobile_5"] = df.groupby(sc)["value_log"].transform(lambda x: x.rolling(5,min_periods=1).mean())
    df["target"]           = df.groupby(sc)["value_log"].shift(-1)
    ts_dt = pd.to_datetime(df["timestamp"])
    df["hour_cos"]         = np.cos(2*np.pi*ts_dt.dt.hour/24)
    df["weekday_cos"]      = np.cos(2*np.pi*ts_dt.dt.weekday/7)
    return df
# ══ PHASE 3 ══════════════════════════════════════════════
def phase3_train_predict(df_srv):
    feats = [f for f in FEAT_XGB if f in df_srv.columns]
    if not feats or "target" not in df_srv.columns: return None, 0, 0
    mask = df_srv["target"].notna()
    X = df_srv[feats].fillna(0).values; y = df_srv["target"].values
    X_tr = X[mask]; y_tr = y[mask]
    if len(X_tr) < 20: return None, 0, 0
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0)
    model.fit(X_tr, y_tr)
    pred_log  = np.clip(model.predict(X), 0, None)
    pred_vals = np.clip(np.expm1(pred_log), 0, 1)
    mae = float(mean_absolute_error(y_tr, pred_log[mask]))
    try: r2 = float(r2_score(y_tr, pred_log[mask]))
    except: r2 = 0.0
    return pred_vals, mae, r2
# ══ PHASE 4 ══════════════════════════════════════════════
def phase4_detection(df_srv):
    fo = [f for f in AFEATS if f in df_srv.columns]
    if not fo:
        n = len(df_srv); return {}, np.zeros(n,int), np.zeros(n), np.zeros(n,int), fo
    n = len(df_srv); sc = StandardScaler()
    X = df_srv[fo].fillna(df_srv[fo].median()).fillna(0).values
    split_prod = int(len(X) * 0.7)
    Xtr = sc.fit_transform(X[:split_prod])      # fit sur train uniquement
    Xall = np.vstack([Xtr, sc.transform(X[split_prod:])])  # transform sur reste
    cpu_v = df_srv["value"].values * 100
    iqr   = float(np.percentile(cpu_v,75) - np.percentile(cpu_v,25))
    cont  = 0.08 if iqr < 5.0 else CONT
    labs  = {}
    # Isolation Forest — fit train, predict all
    mdl_if = IsolationForest(contamination=cont, random_state=42, n_estimators=100)
    mdl_if.fit(Xtr)
    labs["IF"] = (mdl_if.predict(Xall) == -1).astype(int)
    # LOF — novelty=True : fit train, predict all
    mdl_lof = LocalOutlierFactor(n_neighbors=min(20, split_prod-1),contamination=cont, novelty=True)
    mdl_lof.fit(Xtr)
    labs["LOF"] = (mdl_lof.predict(Xall) == -1).astype(int)
    # One-Class SVM — fit train, predict all
    mdl_svm = OneClassSVM(nu=cont, kernel="rbf", gamma="scale")
    mdl_svm.fit(Xtr)
    labs["SVM"] = (mdl_svm.predict(Xall) == -1).astype(int)
    nb    = labs["IF"] + labs["LOF"] + labs["SVM"]
    score = 0.6*(nb/3); conf = (score >= SEUIL).astype(int)
    return labs, nb, score, conf, fo
def send_email(to, frm, pwd, subject, body):
    try:
        msg = MIMEMultipart(); msg["From"]=frm; msg["To"]=to; msg["Subject"]=subject
        msg.attach(MIMEText(body,"html"))
        with smtplib.SMTP("smtp.gmail.com",587) as s:
            s.starttls(); s.login(frm,pwd); s.sendmail(frm,to,msg.as_string())
        return True
    except: return False
def make_labels(timestamps):
    return json.dumps([str(t)[11:16] for t in timestamps])
# ══ SESSION STATE ═════════════════════════════════════════
if "df_raw" not in st.session_state:
    st.session_state.df_raw      = None
    st.session_state.source_name = None
if "email_cfg" not in st.session_state:
    st.session_state.email_cfg = {"to":"","from":"","pwd":"","seuil_w":75,"seuil_c":90,"notif_crit":True,"notif_warn":True,"saved":False}
# ══ TOPBAR ═══════════════════════════════════════════════
now_str = datetime.now().strftime("%d %B %Y · %H:%M")
st.markdown(f"""
<div style="padding:4px 4px 8px 4px;margin-bottom:8px;margin-top:-80px;">
  <div style="font-size:20px;font-weight:700;color:#1E293B;letter-spacing:-0.3px;line-height:1.2">
    AWS CPU Monitor
    <span style="font-size:12px;font-weight:400;color:#94A3B8;margin-left:10px;letter-spacing:0">Prediction &amp; Anomaly Detection</span>
  </div>
  <div style="font-size:11px;color:#CBD5E1;margin-top:4px;font-weight:400">{now_str}</div>
</div>
""", unsafe_allow_html=True)
# ══ IMPORT ═══════════════════════════════════════════════
st.markdown('<div class="sec-label">Source de donnees</div>', unsafe_allow_html=True)
with st.expander("Importer un fichier CSV ou se connecter à une API", expanded=st.session_state.df_raw is None):
    c1,c2 = st.columns([4,1])
    with c1: uploaded = st.file_uploader("Glissez un fichier CSV ou cliquez pour parcourir",type=["csv"],label_visibility="visible")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        imp_btn  = st.button("Importer", use_container_width=True)
    st.markdown('<div style="height:1px;background:#F1F5F9;margin:8px 0"></div>', unsafe_allow_html=True)
    c3,c4 = st.columns([4,1])
    with c3: api_url = st.text_input("URL API",placeholder="http://127.0.0.1:5000/cpu/metrics?file=ec2_cpu_utilization_xxx.csv",label_visibility="visible")
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        conn_btn = st.button("Connecter", use_container_width=True)
    if uploaded and imp_btn:
        with st.spinner("Phase 2 en cours..."):
            df_tmp = pd.read_csv(uploaded, sep=None, engine='python')            
            df_up = phase2_features(df_tmp, uploaded.name)        
        st.session_state.df_raw = df_up; st.session_state.source_name = uploaded.name
        st.success(f"{len(df_up):,} lignes | Phase 2 appliquee")
        st.rerun()
    if conn_btn and api_url:
        try:
            with st.spinner("Connexion API + Phase 2..."):
                r = requests.get(api_url, timeout=15); r.raise_for_status()
                data = r.json()
                df_api = pd.DataFrame(data if isinstance(data,list) else data.get("data",[data]))
                nom = api_url.split("?file=")[-1].split("/")[-1] if "?file=" in api_url else "api"
                df_api = phase2_features(df_api, nom)
            st.session_state.df_raw = df_api; st.session_state.source_name = f"API: {nom}"
            st.success(f"{len(df_api):,} lignes | Phase 2 appliquee"); st.rerun()
        except Exception as e: st.error(f"Erreur API : {e}")
if st.session_state.df_raw is None:
    st.markdown("""<div class="welcome"><h2>Bienvenue sur le Dashboard CPU Monitor</h2>
    <p>Importez un fichier CSV CloudWatch ou connectez une API.</p>
    <p style="font-size:12px;color:#94A3B8">Format : CSV avec colonnes timestamp + value</p></div>""", unsafe_allow_html=True)
    st.stop()
df_raw   = st.session_state.df_raw
serveurs = sorted(df_raw["serveur_id"].unique())
srv_sel  = st.selectbox("", serveurs, format_func=lambda x: f"EC2 — {x}", label_visibility="collapsed")
# ══ FENETRE TEMPORELLE ═══════════════════════════════════
df_srv_full = df_raw[df_raw["serveur_id"]==srv_sel].sort_values("timestamp").reset_index(drop=True)
ts_all = pd.to_datetime(df_srv_full["timestamp"])
ts_min = ts_all.min(); ts_max = ts_all.max()
# ── Durées sur une seule ligne : CPU réel à gauche, CPU prédit à droite
c_left, c_right = st.columns(2)
with c_left:
    st.markdown('<div style="font-size:11px;font-weight:600;color:#374151;margin-bottom:8px;display:flex;align-items:center;gap:6px"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#378ADD"></span>Duree CPU reel</div>', unsafe_allow_html=True)
    preset_reel = st.selectbox("", ["6h","12h","24h","48h","Tout"],
                                index=1, key="pr_reel", label_visibility="collapsed")
with c_right:
    st.markdown('<div style="font-size:11px;font-weight:600;color:#374151;margin-bottom:8px;display:flex;align-items:center;gap:6px"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#1D9E75"></span>Duree CPU predit</div>', unsafe_allow_html=True)
    preset_pred = st.selectbox("", ["1h","3h","6h","12h","24h"],
                                index=2, key="pr_pred", label_visibility="collapsed")
# Calcul timestamps — preset uniquement (pas de calendrier)
durees_map = {"1h":1,"3h":3,"6h":6,"12h":12,"24h":24,"48h":48}
if preset_reel == "Tout":
    dt_reel_debut, dt_reel_fin = ts_min, ts_max
else:
    dt_reel_debut = ts_min
    dt_reel_fin   = ts_min + pd.Timedelta(hours=durees_map[preset_reel])
dt_pred_debut = dt_reel_fin
dt_pred_fin   = dt_pred_debut + pd.Timedelta(hours=durees_map.get(preset_pred, 6))
dt_reel_fin   = min(dt_reel_fin,   ts_max)
dt_pred_debut = min(dt_pred_debut, ts_max)
dt_pred_fin   = min(dt_pred_fin,   ts_max)
mask_all = (ts_all >= dt_reel_debut) & (ts_all <= dt_pred_fin)
df_srv   = df_srv_full[mask_all].reset_index(drop=True)
if len(df_srv) < 20:
    st.warning(f"Fenetre trop courte ({len(df_srv)} obs)."); st.stop()
st.markdown(
    f'<div style="font-size:10px;color:#64748B;margin-top:12px;padding:8px 12px;background:#F8FAFC;border-radius:8px;border:1px solid #E8ECF0;">' +
    f'<span style="color:#185FA5;font-weight:600">CPU reel</span> : {dt_reel_debut.strftime("%d/%m %H:%M")} → {dt_reel_fin.strftime("%d/%m %H:%M")}' +
    f' &nbsp;&nbsp;·&nbsp;&nbsp; <span style="color:#1D9E75;font-weight:600">CPU predit</span> : {dt_pred_debut.strftime("%d/%m %H:%M")} → {dt_pred_fin.strftime("%d/%m %H:%M")}</div>',
    unsafe_allow_html=True)
# ══ PIPELINE ═════════════════════════════════════════════
with st.spinner(f"Phase 3 — XGBoost sur {srv_sel}..."):
    pred_vals, mae_srv, r2_srv = phase3_train_predict(df_srv)
with st.spinner(f"Phase 4 — Detection anomalies sur {srv_sel}..."):
    labs, nb_accord, score, confirmed, feat_ok = phase4_detection(df_srv)
cpu       = np.clip(df_srv["value"].values * 100, 0, 100)
ts        = df_srv["timestamp"]
mu        = float(cpu.mean())
split_idx = int(len(cpu)*0.7)
seuil_vis = float(np.percentile(cpu[:split_idx],95)) if split_idx>0 else float(cpu.max()*0.9)
n_anom    = int(confirmed.sum())
n_crit    = int((nb_accord==3).sum())
cpu_now   = float(cpu[-1])
cpu_pred  = min(float(pred_vals[-1])*100,100.0) if pred_vals is not None else cpu_now
dispo     = round(100 - n_anom/len(df_srv)*100, 1)
cpu_trend = cpu_now - float(cpu[-6]) if len(cpu)>5 else 0
pct_anom  = n_anom/len(df_srv)*100
# ══ ARRAYS JS — deux fenetres séparées ═══════════════════
lbl_js   = make_labels(ts.values)
lbl_list = json.loads(lbl_js)
N        = len(lbl_list)
ts_dt        = pd.to_datetime(ts)
mask_reel    = (ts_dt >= dt_reel_debut) & (ts_dt <= dt_reel_fin)
mask_pred    = (ts_dt >= dt_pred_debut) & (ts_dt <= dt_pred_fin)
mask_visible = mask_reel | mask_pred
# Série réel
indices_reel = [i for i in range(N) if mask_reel.iloc[i]]
lbl_reel     = [lbl_list[i] for i in indices_reel]
real_vals    = [round(float(cpu[i]),2) for i in indices_reel]
thr_reel     = [round(seuil_vis,1)] * len(lbl_reel)
anom_reel    = [{"x": lbl_list[i], "y": max(round(float(cpu[i]),2), 8.0)}
                for i in indices_reel if confirmed[i]==1]
# Série prédit
indices_pred  = [i for i in range(N) if mask_pred.iloc[i]]
lbl_pred      = [lbl_list[i] for i in indices_pred]
pred_vals_win = [round(float(pred_vals[i])*100,2) for i in indices_pred] if pred_vals is not None else [0]*len(indices_pred)
thr_pred      = [round(seuil_vis,1)] * len(lbl_pred)
lbl_reel_js  = json.dumps(lbl_reel)
real_js      = json.dumps(real_vals)
thr_reel_js  = json.dumps(thr_reel)
anom_reel_js = json.dumps(anom_reel)
lbl_pred_js  = json.dumps(lbl_pred)
pred_js      = json.dumps(pred_vals_win)
thr_pred_js  = json.dumps(thr_pred)
# ── Axe Y dynamique par graphique ──
y_max_reel = max(real_vals) if real_vals else 100
y_max_pred = max(pred_vals_win) if pred_vals_win else 100
y_max_reel = (max(y_max_reel * 1.15, y_max_reel + 5))# marge 15%, plafond 100
y_max_pred = (max(y_max_pred * 1.15, y_max_pred + 5))
y_max_reel = round(y_max_reel, 0)
y_max_pred = round(y_max_pred, 0)
# Anomalies recentes HTML
anom_rows_html = ""
for idx in np.where(confirmed==1)[0][::-1]:
    i = int(idx)
    if i >= len(mask_visible) or not bool(mask_visible.iloc[i]): continue
    nb    = int(nb_accord[i]); cpu_v = float(cpu[i]); sc_v = float(score[i])
    ts_v  = str(df_srv["timestamp"].iloc[i])[11:16]
    srv_n = str(df_srv["serveur_id"].iloc[i])
    if nb==3:   dc,bc,bl = "E24B4A","badge-crit","critique"
    elif nb==2: dc,bc,bl = "EF9F27","badge-warn","warning"
    else:       dc,bc,bl = "6B6B6B","badge-suspect","suspect"
    anom_rows_html += (
        f'<div class="arow"><div class="dot" style="background:#{dc}"></div>' +
        f'<div style="flex:1"><div style="font-size:12px;font-weight:500">{srv_n}</div>' +
        f'<div style="font-size:11px;color:#64748B">{ts_v} · score {sc_v:.2f} · CPU {cpu_v:.1f}%</div>' +
        f'</div><span class="badge {bc}">{bl}</span></div>'
    )
if not anom_rows_html:
    anom_rows_html = '<div class="arow"><div class="dot" style="background:#639922"></div><div style="font-size:12px;color:#64748B">Aucune anomalie detectee</div></div>'
col_cpu  = "#A32D2D" if cpu_now>=90 else "#854F0B" if cpu_now>=75 else "#3B6D11"
col_anom = "#A32D2D" if n_anom>10  else "#854F0B" if n_anom>3   else "#3B6D11"
t_sym    = "↑" if cpu_trend>0 else "↓"
t_col    = "#A32D2D" if cpu_trend>0 else "#3B6D11"
mae_display = f"{mae_srv:.5f}" if mae_srv > 0 else "N/A"
r2_display  = f"{r2_srv:.4f}"  if r2_srv  != 0 else "N/A"
# ══ SECTION 1 — KPI ══════════════════════════════════════
st.markdown('<div class="sec-label">section 1 — indicateurs cles</div>', unsafe_allow_html=True)
st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:4px">
  <div class="mcard"><div class="mlabel">CPU moyen</div>
    <div class="mval" style="color:{col_cpu}">{mu:.1f}%</div>
    <div class="msub" style="color:{t_col}">{t_sym} {abs(cpu_trend):.1f}% vs 30 min</div></div>
  <div class="mcard"><div class="mlabel">Disponibilite</div>
    <div class="mval" style="color:#3B6D11">{dispo}%</div>
    <div class="msub" style="color:var(--color-text-secondary)">{len(serveurs)} serveurs</div></div>
  <div class="mcard"><div class="mlabel">Anomalies detectees</div>
    <div class="mval" style="color:{col_anom}">{n_anom}</div>
    <div class="msub" style="color:#A32D2D">{n_crit} critiques · score≥0.6</div></div>
  <div class="mcard"><div class="mlabel">XGBoost(T+5mins)</div>
    <div class="mval" style="color:#185FA5">{cpu_pred:.1f}%</div>
    <div class="msub" style="color:var(--color-text-secondary)">MAE {mae_display} · R² {r2_display}</div></div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
# ══ SECTIONS 2-3 — Graphiques ════════════════════════════
CHARTS_HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
*{{box-sizing:border-box;margin:0;padding:0;font-family:'Inter',sans-serif;}}
body{{background:transparent;padding:0;}}
.sl{{font-size:10px;font-weight:500;color:#64748B;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;margin-top:16px;}}
.card{{background:#fff;border:0.5px solid #E2E8F0;border-radius:12px;padding:14px 16px;margin-bottom:10px;}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:4px;}}
.ct{{font-size:12px;font-weight:500;color:#64748B;margin-bottom:10px;}}
.legend{{display:flex;gap:14px;margin-top:10px;flex-wrap:wrap;}}
.li{{display:flex;align-items:center;gap:5px;font-size:11px;color:#64748B;}}
.ll{{width:14px;height:2px;}} .ld{{width:8px;height:8px;border-radius:50%;}}
.dv{{height:0.5px;background:#E2E8F0;margin:16px 0;}}
.arow{{display:flex;align-items:center;gap:8px;padding:7px 10px;background:#F8FAFC;border-radius:8px;margin-bottom:4px;}}
.dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0;}}
.badge{{font-size:10px;padding:2px 8px;border-radius:6px;font-weight:500;}}
.badge-ok{{background:#EAF3DE;color:#3B6D11;}}
.badge-warn{{background:#FAEEDA;color:#854F0B;}}
.badge-crit{{background:#FCEBEB;color:#A32D2D;}}
.badge-suspect{{background:#F1F0EE;color:#6B6B6B;border:0.5px solid #B4B2A9;}}
</style></head><body>
<div class="sl">section 2 — cpu reel (historique) et cpu predit (futur)</div>
<div class="card">
  <div class="ct">CPU reel — {srv_sel} · {dt_reel_debut.strftime("%d/%m %H:%M")} → {dt_reel_fin.strftime("%d/%m %H:%M")}</div>
  <div style="position:relative;width:100%;height:180px"><canvas id="c1"></canvas></div>
  <div class="legend">
    <div class="li"><div class="ll" style="background:#378ADD"></div>CPU reel (historique)</div>
    <div class="li"><div class="ll" style="background:#E24B4A"></div>seuil p95</div>
  </div>
</div>
<div class="card">
  <div class="ct">CPU predit XGBoost — {srv_sel} · {dt_pred_debut.strftime("%d/%m %H:%M")} → {dt_pred_fin.strftime("%d/%m %H:%M")} · MAE {mae_display} · R² {r2_display}</div>
  <div style="position:relative;width:100%;height:180px"><canvas id="c2"></canvas></div>
  <div class="legend">
    <div class="li"><div class="ll" style="background:#1D9E75"></div>CPU predit (horizon futur)</div>
    <div class="li"><div class="ll" style="background:#E24B4A"></div>seuil p95</div>
  </div>
</div>
<div class="dv"></div>
<div class="sl">section 3 — detection anomalies (Phase 4 · IF + LOF + SVM · score fusion 0.6)</div>
<div class="g2">
  <div class="card">
    <div class="ct">{n_anom} anomalies confirmees sur {N} observations</div>
    <div style="position:relative;width:100%;height:160px"><canvas id="c3"></canvas></div>
    <div class="legend">
      <div class="li"><div class="ll" style="background:#378ADD"></div>CPU reel</div>
      <div class="li"><div class="ll" style="background:#E24B4A"></div>seuil p95</div>
      <div class="li"><div class="ld" style="background:#E24B4A"></div>anomalie</div>
    </div>
  </div>
  <div class="card">
    <div class="ct">Anomalies recentes — score fusion ≥ 0.6</div>
    <div style="display:flex;flex-direction:column;gap:4px;max-height:220px;overflow-y:auto;">
      {anom_rows_html}
    </div>
  </div>
</div>
<script>
const lbl_reel  = {lbl_reel_js};
const real      = {real_js};
const thr_reel  = {thr_reel_js};
const anom      = {anom_reel_js};
const lbl_pred  = {lbl_pred_js};
const pred      = {pred_js};
const thr_pred  = {thr_pred_js};
const gridC = "rgba(128,128,128,0.12)";
const tFont = {{size:10}};
const mkX = () => ({{grid:{{color:gridC}},ticks:{{font:tFont,maxRotation:0,autoSkip:true,maxTicksLimit:72,
  callback:function(val,idx){{const lv=this.getLabelForValue(val);return lv?lv:"";}}
}}}});
const yAxisReel = {{min:0,max:{y_max_reel},grid:{{color:gridC}},ticks:{{font:tFont}}}};
const yAxisPred = {{min:0,max:{y_max_pred},grid:{{color:gridC}},ticks:{{font:tFont}}}};
const mkBase = (yAx) => ({{responsive:true,maintainAspectRatio:false,animation:false,
  plugins:{{legend:{{display:false}},tooltip:{{bodyFont:{{size:11}}}}}},scales:{{x:mkX(),y:yAx}}}});
// c1 — CPU reel sur sa propre fenetre
new Chart(document.getElementById("c1"),{{type:"line",data:{{labels:lbl_reel,datasets:[
 {{data:real,borderColor:"#378ADD",borderWidth:1.5,pointRadius:0,tension:.2,fill:"origin"}},
  {{data:thr_reel,borderColor:"#E24B4A",borderWidth:1.2,borderDash:[5,3],pointRadius:0,fill:false}},
]}},options:mkBase(yAxisReel)}});

// c2 — CPU predit sur sa propre fenetre
new Chart(document.getElementById("c2"),{{type:"line",data:{{labels:lbl_pred,datasets:[
  {{data:pred,borderColor:"#1D9E75",borderWidth:1.5,pointRadius:0,tension:.2,fill:"origin"}},
  {{data:thr_pred,borderColor:"#E24B4A",borderWidth:1.2,borderDash:[5,3],pointRadius:0,fill:false}}
]}},options:mkBase(yAxisPred)}});
// c3 — detection sur fenetre reelle
new Chart(document.getElementById("c3"),{{type:"line",data:{{labels:lbl_reel,datasets:[
  {{data:real,borderColor:"#378ADD",borderWidth:1.2,pointRadius:0,tension:.2,fill:false}},
  {{data:thr_reel,borderColor:"#E24B4A",borderWidth:1.2,borderDash:[5,3],pointRadius:0,fill:false}},
  {{type:"scatter",data:anom,backgroundColor:"#E24B4A",pointRadius:5}}
]}},options:mkBase(yAxisReel)}});

</script></body></html>"""
components.html(CHARTS_HTML, height=980, scrolling=False)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
# ══ SECTION 4 — Alertes email ════════════════════════════
st.markdown('<div class="sec-label">section 4 — configuration des alertes email</div>', unsafe_allow_html=True)
c_cfg1, c_cfg2 = st.columns([1,1], gap="large")
with c_cfg1:
    st.markdown('<div style="font-size:11px;color:var(--color-text-secondary);margin-bottom:5px">Destinataire</div>', unsafe_allow_html=True)
    cfg_to   = st.text_input("",placeholder="admin@example.com",key="e_to",value=st.session_state.email_cfg["to"],label_visibility="collapsed")
    st.markdown('<div style="font-size:11px;color:var(--color-text-secondary);margin-bottom:5px">Expediteur Gmail</div>', unsafe_allow_html=True)
    cfg_from = st.text_input("",placeholder="monitor@gmail.com",key="e_from",value=st.session_state.email_cfg["from"],label_visibility="collapsed")
    st.markdown('<div style="font-size:11px;color:var(--color-text-secondary);margin-bottom:5px">Mot de passe application Gmail</div>', unsafe_allow_html=True)
    cfg_pwd  = st.text_input("",placeholder="xxxx xxxx xxxx xxxx",type="password",key="e_pwd",label_visibility="collapsed")
    cn1,cn2  = st.columns(2)
    with cn1:
        st.markdown('<div style="font-size:11px;color:var(--color-text-secondary);margin-bottom:5px">Seuil warning (%)</div>', unsafe_allow_html=True)
        seuil_w = st.number_input("",value=st.session_state.email_cfg["seuil_w"],min_value=0,max_value=99,key="sw",label_visibility="collapsed")
    with cn2:
        st.markdown('<div style="font-size:11px;color:var(--color-text-secondary);margin-bottom:5px">Seuil critique (%)</div>', unsafe_allow_html=True)
        seuil_c = st.number_input("",value=st.session_state.email_cfg["seuil_c"],min_value=0,max_value=100,key="sc",label_visibility="collapsed")
with c_cfg2:
    st.markdown('<div style="font-size:11px;color:var(--color-text-secondary);margin-bottom:8px">Notifications actives</div>', unsafe_allow_html=True)
    notif_crit = st.checkbox("Alerte email — CPU CRITIQUE", value=st.session_state.email_cfg["notif_crit"])
    notif_warn = st.checkbox("Alerte email — CPU WARNING",  value=st.session_state.email_cfg["notif_warn"])
    notif_anom = st.checkbox("Alerte email — Anomalie confirmee (score >= 0.6)", value=True)
    st.markdown("<br>", unsafe_allow_html=True)
    b1,b2 = st.columns(2)
    with b1:
        if st.button("Tester l'envoi"):
            to_v=st.session_state.email_cfg.get("to",""); from_v=st.session_state.email_cfg.get("from",""); pwd_v=st.session_state.email_cfg.get("pwd","")
            if to_v and from_v and pwd_v:
                body_t = f"""<h2 style='color:#185FA5'>Test — CPU Monitor PFE</h2>
                <p><b>Serveur :</b> {srv_sel}</p><p><b>CPU actuel :</b> {cpu_now:.1f}%</p>
                <p><b>Anomalies :</b> {n_anom} ({n_crit} critiques)</p>
                <p>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p><hr><small>PFE</small>"""
                ok = send_email(to_v,from_v,pwd_v,f"[TEST] CPU Monitor — {srv_sel}",body_t)
                if ok: st.success(f"Email envoye a {to_v}")
                else: st.error("Echec — verifier les identifiants Gmail")
            else: st.warning("Sauvegarder d'abord les parametres")
    with b2:
        if st.button("Sauvegarder"):
            st.session_state.email_cfg.update({"to":cfg_to,"from":cfg_from,"pwd":cfg_pwd,"seuil_w":seuil_w,"seuil_c":seuil_c,"notif_crit":notif_crit,"notif_warn":notif_warn,"saved":True})
            st.success("Parametres sauvegardes")
cfg = st.session_state.email_cfg
if cfg["saved"] and cfg["to"] and cfg["from"] and cfg["pwd"]:
    alert_sent = False
    if cfg["notif_crit"] and cpu_now >= cfg["seuil_c"]:
        body = f"""<h2 style='color:#A32D2D'>ALERTE CRITIQUE — CPU {cpu_now:.1f}%</h2>
        <table style='border-collapse:collapse;font-size:13px'>
        <tr><td style='padding:4px 12px;color:#666'>Serveur</td><td><b>{srv_sel}</b></td></tr>
        <tr><td style='padding:4px 12px;color:#666'>CPU actuel</td><td><b style='color:#A32D2D'>{cpu_now:.1f}%</b></td></tr>
        <tr><td style='padding:4px 12px;color:#666'>Anomalies</td><td>{n_anom} ({n_crit} critiques)</td></tr>
        <tr><td style='padding:4px 12px;color:#666'>Timestamp</td><td>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
        </table><hr><small>PFE</small>"""
        ok = send_email(cfg["to"],cfg["from"],cfg["pwd"],f"CRITIQUE CPU {cpu_now:.1f}% — {srv_sel}",body)
        if ok: st.error(f"ALERTE CRITIQUE envoyee a {cfg['to']}"); alert_sent = True
    elif cfg["notif_warn"] and cpu_now >= cfg["seuil_w"]:
        body = f"""<h2 style='color:#854F0B'>WARNING — CPU {cpu_now:.1f}%</h2>
        <table style='border-collapse:collapse;font-size:13px'>
        <tr><td style='padding:4px 12px;color:#666'>Serveur</td><td><b>{srv_sel}</b></td></tr>
        <tr><td style='padding:4px 12px;color:#666'>CPU actuel</td><td><b>{cpu_now:.1f}%</b></td></tr>
        <tr><td style='padding:4px 12px;color:#666'>Timestamp</td><td>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
        </table><hr><small>PFE</small>"""
        ok = send_email(cfg["to"],cfg["from"],cfg["pwd"],f"WARNING CPU {cpu_now:.1f}% — {srv_sel}",body)
        if ok: st.warning(f"WARNING envoye a {cfg['to']}"); alert_sent = True
    if notif_anom and n_anom>0 and pct_anom>5 and not alert_sent:
        body = f"""<h2 style='color:#854F0B'>ANOMALIES — {n_anom} evenements</h2>
        <table style='border-collapse:collapse;font-size:13px'>
        <tr><td style='padding:4px 12px;color:#666'>Serveur</td><td><b>{srv_sel}</b></td></tr>
        <tr><td style='padding:4px 12px;color:#666'>Anomalies</td><td><b>{n_anom}</b> ({pct_anom:.1f}%)</td></tr>
        <tr><td style='padding:4px 12px;color:#666'>Timestamp</td><td>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
        </table><hr><small>PFE</small>"""
        ok = send_email(cfg["to"],cfg["from"],cfg["pwd"],f"ANOMALIES {n_anom} — {srv_sel}",body)
        if ok: st.warning(f"Alerte anomalies envoyee a {cfg['to']}")
st.markdown(f"""
<div class="footer">
  <span>PFE — Genie Logiciel & SI — Dashboard ML CPU</span>
  <span>Phase2 + Phase3 (XGBoost) + Phase4 (IF+LOF+SVM) · Score fusion 0.6</span>
  <span>AWS CloudWatch · {st.session_state.source_name or "Aucune source"} · {datetime.now().strftime("%Y-%m-%d")}</span>
</div>
""", unsafe_allow_html=True)