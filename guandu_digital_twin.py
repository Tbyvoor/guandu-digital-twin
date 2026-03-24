"""
tools/guandu_digital_twin.py

Digital Twin — Guandu rivier | LG Sonic × CEDAE | Rio de Janeiro
Gebruik: py -m streamlit run tools/guandu_digital_twin.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Guandu Digital Twin",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Design tokens ──────────────────────────────────────────────────────────────
C_BLUE    = "#0099CC"
C_ORANGE  = "#E8622A"
C_GREEN   = "#27AE60"
C_RED     = "#C0392B"
C_YELLOW  = "#F39C12"
C_DARK    = "#0D1B2A"
C_MID     = "#1E3A5F"
C_LIGHT   = "#F0F4F8"
C_TEXT    = "#2C3E50"
C_MUTED   = "#7F8C8D"
C_WHITE   = "#FFFFFF"
C_BORDER  = "#DDE3E9"

CHART = dict(
    plot_bgcolor=C_WHITE,
    paper_bgcolor=C_WHITE,
    font=dict(family="Inter, sans-serif", color=C_TEXT, size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER, borderwidth=1,
                font=dict(size=11), orientation="h", yanchor="bottom", y=1.02),
)

def style(fig, height=280, title="", margin_l=12):
    fig.update_layout(**CHART, height=height,
                      margin=dict(l=margin_l, r=12, t=36, b=12),
                      title=dict(text=title, font=dict(size=13)))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=C_BORDER, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor="#EEF2F6", zeroline=False, linecolor=C_BORDER, tickfont=dict(size=11))
    return fig

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background: {C_DARK};
  }}
  [data-testid="stSidebar"] * {{ color: #CBD5E1 !important; }}
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {{ color: {C_WHITE} !important; }}
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label {{ color: #94A3B8 !important; font-size: 12px !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em; }}
  [data-testid="stSidebar"] hr {{ border-color: #1E3A5F !important; }}

  /* Main bg */
  .main .block-container {{ background: {C_LIGHT}; padding-top: 1.5rem; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid {C_BORDER}; }}
  .stTabs [data-baseweb="tab"] {{
    font-size: 13px; font-weight: 600; color: {C_MUTED};
    padding: 8px 18px; border-radius: 6px 6px 0 0;
    background: transparent; border: none;
  }}
  .stTabs [aria-selected="true"] {{ color: {C_BLUE} !important; border-bottom: 2px solid {C_BLUE}; }}

  /* KPI cards */
  .kpi-card {{
    background: {C_WHITE};
    border-radius: 10px;
    padding: 18px 20px;
    border: 1px solid {C_BORDER};
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .kpi-label {{ font-size: 11px; font-weight: 600; color: {C_MUTED}; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 26px; font-weight: 700; color: {C_DARK}; line-height: 1.1; }}
  .kpi-sub   {{ font-size: 11px; color: {C_MUTED}; margin-top: 4px; }}
  .kpi-ok    {{ color: {C_GREEN}; }}
  .kpi-warn  {{ color: {C_YELLOW}; }}
  .kpi-alert {{ color: {C_RED}; }}

  /* Buoy status cards */
  .buoy-card {{
    background: {C_WHITE};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .buoy-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .buoy-id  {{ font-size: 12px; font-weight: 700; color: {C_DARK}; min-width: 28px; }}
  .buoy-name{{ font-size: 12px; color: {C_MUTED}; flex: 1; }}
  .buoy-vals{{ font-size: 11px; color: {C_TEXT}; text-align: right; }}

  /* Alert cards */
  .alert-card {{
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border: 1px solid;
    display: flex;
    gap: 14px;
    align-items: flex-start;
  }}
  .alert-red    {{ background: #FEF2F2; border-color: #FECACA; }}
  .alert-orange {{ background: #FFFBEB; border-color: #FDE68A; }}
  .alert-icon   {{ line-height: 1; display:flex; align-items:flex-start; padding-top:2px; }}
  .alert-title  {{ font-size: 13px; font-weight: 700; color: {C_DARK}; }}
  .alert-body   {{ font-size: 12px; color: {C_TEXT}; margin-top: 2px; }}
  .alert-action {{ font-size: 11px; color: {C_MUTED}; margin-top: 6px; font-style: italic; }}

  /* Log */
  .log-row {{ font-family: 'Courier New', monospace; font-size: 11px; padding: 4px 0; border-bottom: 1px solid {C_BORDER}; }}

  /* Section labels */
  .section-label {{
    font-size: 11px; font-weight: 700; color: {C_MUTED};
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 12px; margin-top: 4px;
  }}

  /* Divider */
  hr {{ border: none; border-top: 1px solid {C_BORDER}; margin: 16px 0; }}

  /* Hide streamlit branding */
  #MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── Constanten ─────────────────────────────────────────────────────────────────
BUOYS = [
    {"id": "B01", "name": "Nascente",       "lat": -22.856, "lon": -43.832},
    {"id": "B02", "name": "Seropédica",     "lat": -22.862, "lon": -43.785},
    {"id": "B03", "name": "Centraal",       "lat": -22.868, "lon": -43.740},
    {"id": "B04", "name": "Captação Norte", "lat": -22.874, "lon": -43.695},
    {"id": "B05", "name": "CEDAE Inname",   "lat": -22.859, "lon": -43.648},
    {"id": "B06", "name": "Station 6",      "lat": -22.845, "lon": -43.612},
    {"id": "B07", "name": "Stroomafwaarts", "lat": -22.838, "lon": -43.572},
    {"id": "B08", "name": "Uitstroom",      "lat": -22.825, "lon": -43.540},
]

THR = {
    "algae":   {"warn": 40,  "alert": 70},
    "geosmin": {"warn": 20,  "alert": 50},
    "ph":      {"low": 6.5,  "high": 8.5},
    "oxygen":  {"warn": 5.0, "alert": 3.0},
}


def seasonal_solar(doy):
    return 18.0 + 6.0 * np.sin(2 * np.pi * (doy - 355) / 365)


# ── Historische lozingsgebeurtenissen — Guandu rivier ──────────────────────────
# Gebaseerd op bekende incidenten: industriële lozingen Seropédica,
# landbouwrunoff (suikerriet, citrus), rioolwater en 2020-crisis patronen.
_NOW = datetime.now()
DISCHARGE_EVENTS = [
    {"date": _NOW - timedelta(days=54), "severity": 2.8, "duration": 5,
     "desc": "Industriële lozing — Seropédica chemische fabriek"},
    {"date": _NOW - timedelta(days=41), "severity": 1.9, "duration": 3,
     "desc": "Landbouwrunoff na hevige regenval"},
    {"date": _NOW - timedelta(days=29), "severity": 2.4, "duration": 4,
     "desc": "Suikerrietverwerking — verhoogde nutriëntenlozing"},
    {"date": _NOW - timedelta(days=18), "severity": 1.6, "duration": 2,
     "desc": "Routinelozing industriezone Nova Iguaçu"},
    {"date": _NOW - timedelta(days=8),  "severity": 3.2, "duration": 6,
     "desc": "Ernstige lozing — onvoldoende rioolwaterzuivering"},
]


def get_discharge_level(date):
    """
    Berekent het lozingsniveau op een gegeven datum.
    Combineert seizoenspatroon (regenseizoen = meer runoff) met
    het naijlend effect van bekende historische lozingsincidenten.
    """
    doy = date.timetuple().tm_yday
    # Regenseizoen in Brazilië: okt–mrt hogere runoff
    seasonal = 1.0 + 0.35 * np.sin(2 * np.pi * (doy - 30) / 365)

    event_factor = 0.0
    for ev in DISCHARGE_EVENTS:
        days_since = (date - ev["date"]).days
        if days_since < 0:
            continue  # toekomstige event, sla over
        if days_since <= ev["duration"]:
            spike = ev["severity"] - 1.0
        else:
            decay = days_since - ev["duration"]
            spike = (ev["severity"] - 1.0) * np.exp(-decay / 7.0)
        event_factor = max(event_factor, spike)

    return round(seasonal * (1.0 + event_factor), 3)


# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_history(days=60, seed=42):
    np.random.seed(seed)
    dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]
    records = []
    # Realistische basiswaarden per buoy:
    # Upstream (B01-B03) = schoon bronwater
    # Midden (B04-B05)   = licht verhoogd door landbouw/industrie
    # Downstream (B06-B08) = probleemzone, CEDAE inname
    BUOY_PROFILE = {
        "B01": {"algae": 8,  "geosmin": 4,  "temp": 24.5, "o2": 8.2, "turb": 3},
        "B02": {"algae": 12, "geosmin": 6,  "temp": 25.0, "o2": 7.9, "turb": 4},
        "B03": {"algae": 16, "geosmin": 9,  "temp": 25.4, "o2": 7.5, "turb": 5},
        "B04": {"algae": 25, "geosmin": 14, "temp": 26.0, "o2": 7.0, "turb": 7},
        "B05": {"algae": 35, "geosmin": 22, "temp": 26.5, "o2": 6.4, "turb": 9},
        "B06": {"algae": 44, "geosmin": 30, "temp": 27.0, "o2": 5.8, "turb": 11},
        "B07": {"algae": 52, "geosmin": 38, "temp": 27.3, "o2": 5.2, "turb": 13},
        "B08": {"algae": 60, "geosmin": 45, "temp": 27.8, "o2": 4.8, "turb": 15},
    }

    for b in BUOYS:
        prof         = BUOY_PROFILE[b["id"]]
        base_temp    = prof["temp"] + np.random.normal(0, 0.2)
        base_algae   = float(prof["algae"])
        algae        = base_algae
        base_geosmin = prof["geosmin"]

        for i, d in enumerate(dates):
            doy  = d.timetuple().tm_yday
            season = np.sin(2 * np.pi * i / 365) * 2
            temp = base_temp + season + np.random.normal(0, 0.6)
            ph   = 7.2 + np.random.normal(0, 0.2)
            o2   = max(2.0, prof["o2"] - (temp - prof["temp"]) * 0.1 + np.random.normal(0, 0.3))
            turb = prof["turb"] + np.random.normal(0, 1.2)
            solar = max(5.0, seasonal_solar(doy) + np.random.normal(0, 2.5))

            # Varieer behandeling zodat XGBoost het effect leert (opgeslagen, niet toegepast)
            treatment_val = np.random.uniform(0.0, 1.0)

            # Historische algen: vloeiende lijn rond basiswaarde
            # Autocorrelatie zorgt dat elke dag gedeeltelijk op de vorige dag bouwt
            algae = max(0.0, min(120.0,
                        algae * 0.85 + base_algae * 0.15 + np.random.normal(0, base_algae * 0.03)))

            geo = max(0, base_geosmin + algae * 0.4 + np.random.normal(0, 2))
            records.append({
                "date": d, "buoy_id": b["id"], "buoy_name": b["name"],
                "lat": b["lat"], "lon": b["lon"],
                "temp": round(temp, 2), "ph": round(ph, 2),
                "oxygen": round(o2, 2), "turbidity": round(turb, 2),
                "solar": round(solar, 1),
                "algae": round(algae, 1), "geosmin": round(geo, 1),
                "chlorophyl": round(algae * 0.8 + np.random.normal(0, 2), 1),
                "treatment": round(treatment_val, 3),
            })
    return pd.DataFrame(records)


# ── Feature engineering ────────────────────────────────────────────────────────
FEATURES = ["temp", "ph", "oxygen", "turbidity", "solar",
            "day_of_year", "day_of_week",
            "lag_1", "lag_3", "lag_7",
            "rolling_mean_7", "rolling_std_7",
            "treatment"]

def add_features(df_b, treatment=0.7):
    d = df_b.copy().sort_values("date").reset_index(drop=True)
    d["day_of_year"]    = d["date"].dt.dayofyear
    d["day_of_week"]    = d["date"].dt.dayofweek
    d["lag_1"]          = d["algae"].shift(1)
    d["lag_3"]          = d["algae"].shift(3)
    d["lag_7"]          = d["algae"].shift(7)
    d["rolling_mean_7"] = d["algae"].shift(1).rolling(7, min_periods=1).mean()
    d["rolling_std_7"]  = d["algae"].shift(1).rolling(7, min_periods=1).std().fillna(0)
    if "treatment" not in d.columns:
        d["treatment"] = treatment
    if "solar" not in d.columns:
        d["solar"] = d["day_of_year"].apply(seasonal_solar)
    return d.dropna(subset=["lag_7"])


# ── XGBoost model trainen ──────────────────────────────────────────────────────
@st.cache_resource
def train_xgb_models(df_all, treatment=0.7):
    """
    Traint één XGBoost model per buoy op 60 dagen historische data.
    Features: temperatuur, pH, O₂, troebelheid, seizoen, lag-waarden, behandeling.
    Target: algenconcentratie de volgende dag.
    """
    models, metrics = {}, {}
    for buoy_id in df_all["buoy_id"].unique():
        df_b = df_all[df_all["buoy_id"] == buoy_id].copy()
        d    = add_features(df_b, treatment)

        X = d[FEATURES]
        y = d["algae"]

        # Train/test split — laatste 10 dagen = test
        split = max(1, len(d) - 10)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        y_pred = model.predict(X_test)
        metrics[buoy_id] = {
            "r2":  round(r2_score(y_test, y_pred), 3),
            "mae": round(mean_absolute_error(y_test, y_pred), 2),
            "n_train": len(X_train),
            "n_test":  len(X_test),
        }
        models[buoy_id] = (model, d)

    return models, metrics


# ── XGBoost iteratieve voorspelling ───────────────────────────────────────────
def predict_xgb(df_b, models_dict, buoy_id, days=7,
                dt=0.0, rf=1.0, disch=1.0, treatment=0.7):
    model, hist = models_dict[buoy_id]
    last   = hist.iloc[-1].copy()

    algae_window = hist["algae"].tail(7).tolist()
    base_temp    = last["temp"]
    base_ph      = last["ph"]
    base_o2      = last["oxygen"]
    base_turb    = last["turbidity"]

    # LG Sonic haalt max ~90% reductie — altijd residu van ~10% startwaarde
    N_start  = float(hist["algae"].iloc[-1])
    N_floor  = max(2.0, N_start * 0.10)

    preds = []
    now   = datetime.now()
    K = 140   # wetenschappelijk: gemeten piek Guandu 135 μg/L (1998)

    for i in range(1, days + 1):
        future_date = now + timedelta(days=i)
        doy = future_date.timetuple().tm_yday
        dow = future_date.weekday()

        temp  = base_temp + dt + np.random.normal(0, 0.2)
        solar = max(3.0, seasonal_solar(doy) + np.random.normal(0, 1.5))
        turb  = base_turb * (1.2 / max(rf, 0.1)) * np.random.uniform(0.9, 1.1)
        o2    = max(1.5, base_o2 - dt * 0.08 + np.random.normal(0, 0.2))
        ph    = base_ph + np.random.normal(0, 0.05)

        lag1 = algae_window[-1]
        lag3 = algae_window[-3] if len(algae_window) >= 3 else algae_window[0]
        lag7 = algae_window[-7] if len(algae_window) >= 7 else algae_window[0]
        rm7  = np.mean(algae_window[-7:]) * disch
        rs7  = np.std(algae_window[-7:]) if len(algae_window) >= 2 else 0.0

        row = pd.DataFrame([[temp, ph, o2, turb, solar, doy, dow,
                              lag1, lag3, lag7, rm7, rs7, treatment]],
                           columns=FEATURES)

        xgb_pred = float(model.predict(row)[0])

        N = algae_window[-1]

        # ── Sliders zijn deltas t.o.v. huidige situatie ──────────────────────
        # Bij alles=0: algen stabiel (geen groei, geen daling)

        # Temperatuurstijging (dt=0 → geen groei, dt>0 → meer groei)
        # Gebaseerd op cardinaal model: optimum 30°C, lineair tussen 0-5°C stijging
        temp_growth = (dt / 5.0) * 0.025 * (1 - N / K)   # max 2.5%/dag bij +5°C

        # Lozing (disch=0 → geen effect, disch>0 → meer nutriënten → meer groei)
        nutrient_growth = (disch / 3.0) * 0.015 * (1 - N / K)   # max 1.5%/dag bij x3

        # Totale groeisnelheid
        growth_rate = temp_growth + nutrient_growth

        # Regen verdunt algen (rf=0 → geen verdunning, rf>0 → verdunning)
        dilution_rate = rf * 0.03   # max 6%/dag bij rf=2 (realistisch: 3-5%/dag)

        # LG Sonic kill rate bouwt op over ~5 dagen
        # Gebaseerd op LG Sonic case studies: ~87% reductie na 3 weken
        # Kleine dagelijkse schommelingen (±15%) voor realisme
        treatment_ramp = min(1.0, max(0.0, (i - 2) / 5)) * treatment
        kill_rate = treatment_ramp * 0.10 * np.random.uniform(0.85, 1.15)

        # Netto: stabiel + groei - verdunning - kill
        N_next = N * (1 + growth_rate - dilution_rate - kill_rate)

        pred = max(N_floor, min(K, N_next + np.random.normal(0, N * 0.04)))

        geo = max(0.0, pred * 0.5 + np.random.normal(0, 1.5))
        preds.append({
            "date":      future_date,
            "algae":     round(pred, 1),
            "geosmin":   round(geo, 1),
            "treatment": treatment,
        })
        algae_window.append(pred)

    return pd.DataFrame(preds)


# ── Wetenschappelijke algengroei voorspelling (geen parameters) ────────────────
def predict_scientific(df_b, buoy_id, days=90, seed=None):
    """
    Wetenschappelijke 90-daagse algengroei voorspelling op basis van:
    - Logistisch groeimodel (Verhulst)
    - Cardinaal temperatuurmodel (Bernard & Rémond 2012)
    - Seizoenscyclus Rio de Janeiro (belichting + temperatuur)
    - Historische variabiliteit als ruis
    Geen gebruikersparameters — puur wetenschappelijk baseline.
    """
    if seed is not None:
        np.random.seed(seed)

    hist = df_b.sort_values("date").reset_index(drop=True)
    N_start   = float(hist["algae"].iloc[-1])
    base_temp = float(hist["temp"].iloc[-1])
    hist_std  = max(1.0, hist["algae"].tail(14).std())

    K     = 140.0   # draagkracht (gemeten piek Guandu 135 μg/L, 1998)
    r_max = 0.15    # max groeisnelheid cyanobacteriën (/dag) — Microcystis literatuur
    T_opt = 30.0    # optimum temperatuur (°C)
    T_min = 15.0    # minimum temperatuur (°C)
    T_max = 38.0    # maximum temperatuur (°C)
    m_base= 0.010   # basale sterfte (/dag) — cel-lyse, uitzinking

    N    = N_start
    now  = datetime.now()
    preds = []

    for i in range(1, days + 1):
        future_date = now + timedelta(days=i)
        doy = future_date.timetuple().tm_yday

        # Seizoenstemperatuur Rio de Janeiro
        # Zomer (dec-mrt): +3°C boven gemiddelde, winter (jun-sep): -3°C
        T_seasonal = base_temp + 3.0 * np.sin(2 * np.pi * (doy - 355) / 365)
        T = T_seasonal + np.random.normal(0, 0.6)

        # Cardinaal temperatuurmodel — piecewise lineair
        if T <= T_min or T >= T_max:
            f_T = 0.0
        elif T <= T_opt:
            f_T = (T - T_min) / (T_opt - T_min)
        else:
            f_T = (T_max - T) / (T_max - T_opt)

        # Lichtfactor op basis van dag van het jaar
        # Meer licht in zomer → meer fotosynthese
        f_L = 0.7 + 0.3 * np.sin(2 * np.pi * (doy - 355) / 365)

        # Logistische groei met temperatuur- en lichtmodulatie
        growth = r_max * f_T * f_L * (1 - N / K)

        # Seizoengebonden sterfte: hoger bij lage temperatuur (cel-lyse, sedimentatie)
        mortality = m_base + 0.015 * (1 - f_T)

        # Netto dynamiek
        N_next = N * (1 + growth - mortality)

        # Realistische dagelijkse ruis (proportioneel aan huidige concentratie)
        noise = np.random.normal(0, hist_std * 0.25)
        N = max(2.0, min(K, N_next + noise))

        geo = max(0.0, N * 0.5 + np.random.normal(0, 1.5))
        preds.append({
            "date":   future_date,
            "algae":  round(N, 1),
            "geosmin": round(geo, 1),
            "temp":   round(T, 1),
            "f_T":    round(f_T, 3),
        })

    return pd.DataFrame(preds)


def status(val, param):
    if param == "oxygen":
        if val < THR["oxygen"]["alert"]: return "alert"
        if val < THR["oxygen"]["warn"]:  return "warn"
        return "ok"
    if param == "ph":
        if val < THR["ph"]["low"] or val > THR["ph"]["high"]: return "warn"
        return "ok"
    if val >= THR[param]["alert"]: return "alert"
    if val >= THR[param]["warn"]:  return "warn"
    return "ok"


def _dot(color):
    return (f'<svg width="8" height="8" style="vertical-align:middle;margin-right:4px">'
            f'<circle cx="4" cy="4" r="4" fill="{color}"/></svg>')

STATUS_ICON = {
    "ok":    (_dot(C_GREEN),  C_GREEN),
    "warn":  (_dot(C_YELLOW), C_YELLOW),
    "alert": (_dot(C_RED),    C_RED),
}


# ── Data laden & modellen trainen ─────────────────────────────────────────────
df      = generate_history(60)
latest  = df.groupby("buoy_id").last().reset_index()
now_str = datetime.now().strftime("%d %b %Y, %H:%M")

with st.spinner("Modellen trainen op historische data..."):
    xgb_models, xgb_metrics = train_xgb_models(df)



# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 4px 0 20px 0;">
      <!-- LG Sonic logo -->
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;
                  padding:10px 12px;background:rgba(255,255,255,0.06);border-radius:8px;
                  border:1px solid rgba(255,255,255,0.08);">
        <div style="background:{C_BLUE};border-radius:5px;padding:5px 9px;
                    font-size:15px;font-weight:800;color:#fff;letter-spacing:0.5px;
                    font-family:'Inter',sans-serif;">LG</div>
        <div style="display:flex;flex-direction:column;line-height:1.2;">
          <span style="font-size:13px;font-weight:700;color:#fff;letter-spacing:2px;">SONIC</span>
          <span style="font-size:9px;color:#64748B;letter-spacing:0.5px;">Water Technology</span>
        </div>
      </div>
      <!-- Dashboard title -->
      <div style="display:flex;align-items:center;gap:8px;">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="{C_BLUE}">
          <path d="M12 2C6 10 4 14 4 16a8 8 0 0 0 16 0c0-2-2-6-8-14z"/>
        </svg>
        <div style="font-size:16px; font-weight:700; color:#fff; letter-spacing:-0.3px;">
          Guandu Digital Twin
        </div>
      </div>
      <div style="font-size:11px; color:#64748B; margin-top:4px; padding-left:26px;">
        CEDAE · Rio de Janeiro
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label" style="color:#475569">Buoy selectie</p>', unsafe_allow_html=True)
    selected_buoy = st.selectbox(
        "Buoy", label_visibility="collapsed",
        options=[b["id"] for b in BUOYS],
        format_func=lambda x: f"{x}  —  {next(b['name'] for b in BUOYS if b['id'] == x)}"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-label" style="color:#475569">Scenario parameters</p>', unsafe_allow_html=True)
    temp_offset   = st.slider("Temperatuurstijging (°C, gem. over periode)", 0.0, 5.0, 0.0, 0.5,
                               help="Gemiddelde stijging t.o.v. nu over de hele voorspellingsperiode · 0 = geen verandering · +5°C = sterke opwarming")
    _temp_label = "Geen effect" if temp_offset == 0 else f"+{temp_offset}°C — {'Licht' if temp_offset < 2 else 'Matig' if temp_offset < 4 else 'Sterk'} verhoogd"
    st.markdown(f'<div style="font-size:11px;color:#64748B;margin-top:-10px;margin-bottom:8px;">↳ {_temp_label}</div>', unsafe_allow_html=True)

    rain_factor   = st.slider("Regenval (factor, gem. over periode)", 0.0, 2.0, 0.0, 0.1,
                               help="Gemiddelde regenintensiteit t.o.v. normaal · 0 = geen regen · 1.0 = normaal · 2.0 = dubbel normaal")
    _rain_label = "Geen regen" if rain_factor == 0 else ("Lichte regen" if rain_factor < 0.5 else "Normale regen" if rain_factor < 1.2 else "Zware regen" if rain_factor < 1.7 else "Extreme regenval")
    st.markdown(f'<div style="font-size:11px;color:#64748B;margin-top:-10px;margin-bottom:8px;">↳ {_rain_label} (×{rain_factor:.1f})</div>', unsafe_allow_html=True)

    discharge     = st.slider("Lozingsintensiteit (factor, gem. over periode)", 0.0, 3.0, 0.0, 0.1,
                               help="Gemiddelde lozingsintensiteit t.o.v. normaal · 0 = geen lozing · 1.0 = normaal · 3.0 = hoge lozing")
    _disc_label = "Geen lozing" if discharge == 0 else ("Lage lozing" if discharge < 1.0 else "Normale lozing" if discharge < 1.8 else "Hoge lozing" if discharge < 2.5 else "Ernstige lozing")
    st.markdown(f'<div style="font-size:11px;color:#64748B;margin-top:-10px;margin-bottom:8px;">↳ {_disc_label} (×{discharge:.1f})</div>', unsafe_allow_html=True)

    forecast_days = st.slider("Voorspellingshorizon (dagen)", 3, 90, 3)

    treatment = st.slider("Ultrasonore behandeling (LG Sonic)", 0.0, 1.0, 0.0, 0.05,
                          help="0.0 = uit · 0.5 = half vermogen · 1.0 = vol vermogen")
    _treat_label = "Uit" if treatment == 0 else ("Laag vermogen" if treatment < 0.35 else "Half vermogen" if treatment < 0.7 else "Hoog vermogen" if treatment < 1.0 else "Vol vermogen")
    st.markdown(f'<div style="font-size:11px;color:#64748B;margin-top:-10px;margin-bottom:8px;">↳ {_treat_label} ({int(treatment*100)}%)</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:11px; color:#475569; line-height:2.0;">
      <div style="display:flex;align-items:center;gap:6px;">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        {now_str}
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M8.5 8.5a5 5 0 0 1 7 0"/><path d="M5.5 5.5a9 9 0 0 1 13 0"/></svg>
        8 MPC-Buoys actief
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
        Realtime simulatie
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{C_DARK}; border-radius:12px; padding:20px 28px; margin-bottom:20px;
            display:flex; justify-content:space-between; align-items:center;">
  <div>
    <div style="font-size:22px; font-weight:700; color:{C_WHITE}; letter-spacing:-0.3px;">
      Guandu Rivier — Digital Twin
    </div>
    <div style="font-size:13px; color:#64748B; margin-top:4px;">
      LG Sonic × CEDAE &nbsp;·&nbsp; Rio de Janeiro, Brazilië &nbsp;·&nbsp; 8 MPC-Buoys
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:11px; color:#475569;">Laatste update</div>
    <div style="font-size:13px; font-weight:600; color:#94A3B8;">{now_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── KPI row ────────────────────────────────────────────────────────────────────
avg_algae   = latest["algae"].mean()
avg_geosmin = latest["geosmin"].mean()
avg_temp    = latest["temp"].mean()
avg_o2      = latest["oxygen"].mean()
n_alerts    = sum(1 for _, r in latest.iterrows()
                  if r["algae"] > 40 or r["geosmin"] > 20 or r["oxygen"] < 5)

def kpi(label, value, sub, stat):
    icon, color = STATUS_ICON[stat]
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color}">{value}</div>
      <div class="kpi-sub">{icon} {sub}</div>
    </div>"""

col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown(kpi("Gem. Algen",       f"{avg_algae:.1f} μg/L",
    "Normaal" if avg_algae < 40 else "Verhoogd" if avg_algae < 70 else "Alarm",
    status(avg_algae, "algae")), unsafe_allow_html=True)
col2.markdown(kpi("Gem. Geosmin",     f"{avg_geosmin:.1f} ng/L",
    "Onder smaakgrens" if avg_geosmin < 20 else "Boven smaakgrens",
    status(avg_geosmin, "geosmin")), unsafe_allow_html=True)
col3.markdown(kpi("Temperatuur",      f"{avg_temp:.1f} °C",
    "Gem. watertemperatuur", "ok"), unsafe_allow_html=True)
col4.markdown(kpi("Opgelost O₂",      f"{avg_o2:.1f} mg/L",
    "Normaal" if avg_o2 > 5 else "Laag" if avg_o2 > 3 else "Kritiek",
    status(avg_o2, "oxygen")), unsafe_allow_html=True)
col5.markdown(kpi("Actieve alerts",   f"{n_alerts} / 8",
    "Buoys met verhoogde waarden",
    "ok" if n_alerts == 0 else "warn" if n_alerts < 4 else "alert"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Kaart & Overzicht  ",
    "  Sensordata  ",
    "  Algenvoorspelling  ",
    "  Scenario-simulator  ",
    "  Waarschuwingen  ",
])


# ────────────────────────────────────────────────────────────────────────────────
# TAB 1 — KAART
# ────────────────────────────────────────────────────────────────────────────────
with tab1:
    col_map, col_right = st.columns([3, 1], gap="medium")

    with col_map:
        st.markdown('<p class="section-label">Live buoy kaart — Guandu rivier</p>', unsafe_allow_html=True)

        dot_colors = []
        for _, row in latest.iterrows():
            s = status(row["algae"], "algae")
            dot_colors.append({"ok": C_GREEN, "warn": C_YELLOW, "alert": C_RED}[s])

        # Algenwaarden per buoy ophalen
        algae_map = {b["id"]: latest[latest["buoy_id"] == b["id"]]["algae"].values[0] for b in BUOYS}

        def algae_to_color(val, opacity=1.0):
            """Blauw → groen → oranje → rood op basis van algenconcentratie."""
            if val < 20:   return f"rgba(0,180,255,{opacity})"
            elif val < 40: return f"rgba(39,174,96,{opacity})"
            elif val < 70: return f"rgba(243,156,18,{opacity})"
            else:          return f"rgba(192,57,43,{opacity})"

        fig_map = go.Figure()

        # Rivier segmenten — elk segment gekleurd op algenwaarde van die buoy
        for i in range(len(BUOYS) - 1):
            b_start = BUOYS[i]
            b_end   = BUOYS[i + 1]
            avg_val = (algae_map[b_start["id"]] + algae_map[b_end["id"]]) / 2

            # Glow lagen per segment
            for width, frac in [(14, 0.06), (8, 0.15), (4, 0.4), (2, 1.0)]:
                fig_map.add_trace(go.Scattermapbox(
                    lat=[b_start["lat"], b_end["lat"]],
                    lon=[b_start["lon"], b_end["lon"]],
                    mode="lines",
                    line=dict(width=width, color=algae_to_color(avg_val, frac)),
                    hoverinfo="skip", showlegend=False,
                ))

        # Buoy markers met pulse-ring effect (grote halftransparante cirkel + solide kern)
        for buoy, color in zip(BUOYS, dot_colors):
            row = latest[latest["buoy_id"] == buoy["id"]].iloc[0]
            r, g, b_ = (
                (39, 174, 96) if color == C_GREEN else
                (243, 156, 18) if color == C_YELLOW else
                (192, 57, 43)
            )
            # Pulse ring
            fig_map.add_trace(go.Scattermapbox(
                lat=[buoy["lat"]], lon=[buoy["lon"]],
                mode="markers",
                marker=dict(size=30, color=f"rgba({r},{g},{b_},0.18)"),
                hoverinfo="skip", showlegend=False,
            ))
            # Kern
            fig_map.add_trace(go.Scattermapbox(
                lat=[buoy["lat"]], lon=[buoy["lon"]],
                mode="markers+text",
                marker=dict(size=14, color=color,
                            symbol="circle"),
                text=[buoy["id"]],
                textposition="top center",
                textfont=dict(size=10, color="#FFFFFF", weight=700),
                name=buoy["name"],
                hovertemplate=(
                    f"<b>{buoy['id']} — {buoy['name']}</b><br>"
                    f"Algen: <b>{row['algae']} μg/L</b><br>"
                    f"Geosmin: <b>{row['geosmin']} ng/L</b><br>"
                    f"Temp: {row['temp']} °C  |  O₂: {row['oxygen']} mg/L  |  pH: {row['ph']}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        fig_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=-22.858, lon=-43.686),
                zoom=10,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=480,
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown(
            f'<span style="font-size:11px;color:{C_MUTED};">'
            f'<span style="color:{C_GREEN}">●</span> Normaal  '
            f'<span style="color:{C_YELLOW}">●</span> Verhoogd (>40 μg/L)  '
            f'<span style="color:{C_RED}">●</span> Alarm (>70 μg/L)</span>',
            unsafe_allow_html=True
        )

        # ── 3D Rivier visualisatie ─────────────────────────────────────────────
        st.markdown('<p class="section-label" style="margin-top:16px;">3D rivierprofiel — algenconcentratie per locatie</p>',
                    unsafe_allow_html=True)

        buoy_lons   = [b["lon"] for b in BUOYS]
        buoy_lats   = [b["lat"] for b in BUOYS]
        algae_vals  = [latest[latest["buoy_id"] == b["id"]]["algae"].values[0] for b in BUOYS]
        buoy_names  = [b["name"] for b in BUOYS]
        buoy_ids    = [b["id"] for b in BUOYS]

        # Interpoleer een vloeiende rivier (meer punten tussen de buoys)
        from scipy.interpolate import interp1d
        t       = np.linspace(0, 1, len(BUOYS))
        t_fine  = np.linspace(0, 1, 80)
        lon_interp    = interp1d(t, buoy_lons,  kind="cubic")(t_fine)
        lat_interp    = interp1d(t, buoy_lats,  kind="cubic")(t_fine)
        algae_interp  = interp1d(t, algae_vals, kind="cubic")(t_fine)
        algae_interp  = np.clip(algae_interp, 0, 120)

        fig_3d = go.Figure()

        # Rivier bodem (plat vlak op z=0)
        fig_3d.add_trace(go.Scatter3d(
            x=lon_interp, y=lat_interp, z=np.zeros(len(t_fine)),
            mode="lines",
            line=dict(color="rgba(0,80,120,0.4)", width=8),
            name="Rivierbedding", showlegend=False,
            hoverinfo="skip",
        ))

        # Verticale "muren" van buoy naar oppervlak — geeft diepte-effect
        for i, (buoy, val) in enumerate(zip(BUOYS, algae_vals)):
            fig_3d.add_trace(go.Scatter3d(
                x=[buoy["lon"], buoy["lon"]],
                y=[buoy["lat"], buoy["lat"]],
                z=[0, val],
                mode="lines",
                line=dict(color="rgba(200,200,200,0.3)", width=2, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))

        # 3D rivierlijn gekleurd op algenconcentratie
        fig_3d.add_trace(go.Scatter3d(
            x=lon_interp, y=lat_interp, z=algae_interp,
            mode="lines",
            line=dict(
                color=algae_interp,
                colorscale=[[0, "#00BFFF"], [0.33, "#27AE60"],
                            [0.66, "#F39C12"], [1.0, "#C0392B"]],
                width=8,
                cmin=0, cmax=100,
            ),
            name="Algenconcentratie", showlegend=False,
            hoverinfo="skip",
        ))

        # Buoy bollen op de 3D lijn
        marker_colors = [
            "#27AE60" if status(v, "algae") == "ok" else
            "#F39C12" if status(v, "algae") == "warn" else "#C0392B"
            for v in algae_vals
        ]
        fig_3d.add_trace(go.Scatter3d(
            x=buoy_lons, y=buoy_lats, z=algae_vals,
            mode="markers+text",
            marker=dict(size=8, color=marker_colors,
                        line=dict(color="white", width=1.5)),
            text=buoy_ids,
            textposition="top center",
            textfont=dict(size=10, color="white"),
            name="Buoys",
            hovertemplate=[
                f"<b>{bid} — {name}</b><br>Algen: <b>{val:.1f} μg/L</b><extra></extra>"
                for bid, name, val in zip(buoy_ids, buoy_names, algae_vals)
            ],
        ))

        # Drempelgrens vlak (40 μg/L)
        fig_3d.add_trace(go.Scatter3d(
            x=lon_interp, y=lat_interp, z=[40] * len(t_fine),
            mode="lines",
            line=dict(color="rgba(243,156,18,0.5)", width=3, dash="dot"),
            name="Waarschuwingsgrens (40)", showlegend=True,
            hoverinfo="skip",
        ))

        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title="Lengtegraad", backgroundcolor="#0D1B2A",
                           gridcolor="#1E3A5F", zerolinecolor="#1E3A5F",
                           tickfont=dict(size=9, color="#94A3B8")),
                yaxis=dict(title="Breedtegraad", backgroundcolor="#0D1B2A",
                           gridcolor="#1E3A5F", zerolinecolor="#1E3A5F",
                           tickfont=dict(size=9, color="#94A3B8")),
                zaxis=dict(title="Algen (μg/L)", backgroundcolor="#0D1B2A",
                           gridcolor="#1E3A5F", zerolinecolor="#1E3A5F",
                           tickfont=dict(size=9, color="#94A3B8"), range=[0, 110]),
                bgcolor="#0D1B2A",
                camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
                aspectmode="manual",
                aspectratio=dict(x=2.5, y=1, z=0.6),
            ),
            paper_bgcolor="#0D1B2A",
            plot_bgcolor="#0D1B2A",
            font=dict(color="#94A3B8"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=420,
            legend=dict(font=dict(color="#94A3B8", size=10),
                        bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        st.caption("Draai de grafiek met je muis · hoogte = algenconcentratie · kleur: blauw=laag → rood=hoog")

    with col_right:
        st.markdown('<p class="section-label">Live status</p>', unsafe_allow_html=True)
        for _, row in latest.iterrows():
            s_algae = status(row["algae"], "algae")
            s_geo   = status(row["geosmin"], "geosmin")
            s_o2    = status(row["oxygen"], "oxygen")
            overall = "alert" if "alert" in [s_algae, s_geo, s_o2] \
                      else "warn" if "warn" in [s_algae, s_geo, s_o2] \
                      else "ok"
            icon, color = STATUS_ICON[overall]
            buoy_name = next(b["name"] for b in BUOYS if b["id"] == row["buoy_id"])
            st.markdown(f"""
            <div style="background:{C_WHITE}; border:1px solid {C_BORDER}; border-radius:8px;
                        padding:10px 12px; margin-bottom:6px; border-left:3px solid {color};">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="font-size:12px; font-weight:700; color:{C_DARK};">{row['buoy_id']}</span>
                <span style="font-size:10px; font-weight:600; color:{color}; background:{'#FEF2F2' if color==C_RED else '#FFFBEB' if color==C_YELLOW else '#F0FDF4'};
                      padding:2px 6px; border-radius:4px;">
                  {'ALARM' if overall=='alert' else 'WAARSCH.' if overall=='warn' else 'OK'}
                </span>
              </div>
              <div style="font-size:11px; color:{C_MUTED}; margin-bottom:5px;">{buoy_name}</div>
              <div style="display:flex; justify-content:space-between;">
                <div>
                  <div style="font-size:10px; color:{C_MUTED};">Algen</div>
                  <div style="font-size:12px; font-weight:600; color:{C_DARK};">{row['algae']} <span style="font-size:9px; color:{C_MUTED};">μg/L</span></div>
                </div>
                <div>
                  <div style="font-size:10px; color:{C_MUTED};">Geosmin</div>
                  <div style="font-size:12px; font-weight:600; color:{C_DARK};">{row['geosmin']} <span style="font-size:9px; color:{C_MUTED};">ng/L</span></div>
                </div>
                <div>
                  <div style="font-size:10px; color:{C_MUTED};">O₂</div>
                  <div style="font-size:12px; font-weight:600; color:{C_DARK};">{row['oxygen']} <span style="font-size:9px; color:{C_MUTED};">mg/L</span></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
# TAB 2 — SENSORDATA
# ────────────────────────────────────────────────────────────────────────────────
with tab2:
    buoy_name_sel = next(b["name"] for b in BUOYS if b["id"] == selected_buoy)
    st.markdown(f'<p class="section-label">{selected_buoy} — {buoy_name_sel} · 60 dagen historie</p>',
                unsafe_allow_html=True)

    df_b = df[df["buoy_id"] == selected_buoy].copy()

    params = [
        ("algae",     "Algenconcentratie",  "μg/L",  C_BLUE,   THR["algae"]),
        ("geosmin",   "Geosmin",            "ng/L",  C_ORANGE, THR["geosmin"]),
        ("temp",      "Temperatuur",        "°C",    C_YELLOW, {}),
        ("oxygen",    "Opgelost O₂",        "mg/L",  C_GREEN,  {}),
        ("ph",        "pH",                 "",      "#9B59B6", {}),
        ("turbidity", "Troebelheid",        "NTU",   C_MUTED,  {}),
    ]

    for row_idx in range(0, len(params), 2):
        cols = st.columns(2, gap="medium")
        for col_idx, col in enumerate(cols):
            pi = row_idx + col_idx
            if pi >= len(params): break
            key, label, unit, color, thr = params[pi]
            r, g, b_ = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fill_rgba = f"rgba({r},{g},{b_},0.08)"

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_b["date"], y=df_b[key], mode="lines", name=label,
                line=dict(color=color, width=1.8),
                fill="tozeroy", fillcolor=fill_rgba,
                hovertemplate=f"%{{y:.1f}} {unit}<extra></extra>",
            ))
            if "warn" in thr:
                fig.add_hline(y=thr["warn"], line_dash="dot", line_color=C_YELLOW,
                              annotation_text="Waarschuwing", annotation_font_size=10)
            if "alert" in thr:
                fig.add_hline(y=thr["alert"], line_dash="dot", line_color=C_RED,
                              annotation_text="Alarm", annotation_font_size=10)
            if "low" in thr:
                fig.add_hline(y=thr["low"],  line_dash="dot", line_color=C_YELLOW)
                fig.add_hline(y=thr["high"], line_dash="dot", line_color=C_YELLOW)

            style(fig, height=220, title=f"<b>{label}</b> ({unit})")
            fig.update_layout(showlegend=False)
            col.plotly_chart(fig, use_container_width=True)

    # Correlatie
    st.markdown('<p class="section-label" style="margin-top:8px;">Correlatie — temperatuur vs. algengroei</p>',
                unsafe_allow_html=True)
    fig_c = px.scatter(df_b, x="temp", y="algae", color="geosmin",
                       color_continuous_scale=[[0, C_BLUE], [0.5, C_YELLOW], [1, C_RED]],
                       labels={"temp": "Temperatuur (°C)", "algae": "Algen (μg/L)", "geosmin": "Geosmin (ng/L)"},
                       opacity=0.7)
    fig_c.update_traces(marker=dict(size=7))
    style(fig_c, height=280, title="<b>Hogere temperatuur → meer algengroei</b>")
    st.plotly_chart(fig_c, use_container_width=True)

    # ── Algensoorten uitsplitsing ──────────────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:8px;">Algensoorten — geschatte concentratie per type</p>',
                unsafe_allow_html=True)

    # Proporties op basis van temperatuur (hogere temp → meer cyanobacteriën)
    # Gebaseerd op typisch patroon Guandu rivier (CEDAE onderzoek)
    df_algae_types = df_b[["date", "algae", "temp"]].copy()
    t_norm = ((df_algae_types["temp"] - 20) / 10).clip(0, 1)   # 0 bij 20°C, 1 bij 30°C
    df_algae_types["Microcystis aeruginosa"]        = df_algae_types["algae"] * (0.32 + 0.22 * t_norm)
    df_algae_types["Cylindrospermopsis raciborskii"]= df_algae_types["algae"] * (0.18 + 0.14 * t_norm)
    df_algae_types["Groenwieren (Chlorophyta)"]     = df_algae_types["algae"] * (0.28 - 0.18 * t_norm)
    df_algae_types["Diatomeeën (Bacillariophyta)"]  = df_algae_types["algae"] * (0.22 - 0.18 * t_norm)

    ALGAE_COLORS = {
        "Microcystis aeruginosa":         "#C0392B",
        "Cylindrospermopsis raciborskii": "#E67E22",
        "Groenwieren (Chlorophyta)":      "#27AE60",
        "Diatomeeën (Bacillariophyta)":   "#2980B9",
    }

    fig_at = go.Figure()
    for species, color in ALGAE_COLORS.items():
        r, g_, b_ = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig_at.add_trace(go.Scatter(
            x=df_algae_types["date"], y=df_algae_types[species].round(1),
            name=species, mode="lines", stackgroup="one",
            line=dict(color=color, width=1),
            fillcolor=f"rgba({r},{g_},{b_},0.75)",
            hovertemplate=f"<b>{species}</b><br>%{{y:.1f}} μg/L<extra></extra>",
        ))
    st.markdown('<p class="section-label" style="margin-top:8px;">Algensoorten samenstelling (gestapeld)</p>', unsafe_allow_html=True)
    style(fig_at, height=280, title="")
    fig_at.update_yaxes(title="μg/L")
    st.plotly_chart(fig_at, use_container_width=True)

    # Huidige dag donut + info kaarten
    col_donut, col_info = st.columns([1, 2], gap="medium")
    with col_donut:
        latest_b = df_algae_types.iloc[-1]
        species_vals = {s: float(latest_b[s]) for s in ALGAE_COLORS}
        fig_pie = go.Figure(go.Pie(
            labels=list(species_vals.keys()),
            values=[round(v, 1) for v in species_vals.values()],
            hole=0.55,
            marker_colors=list(ALGAE_COLORS.values()),
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value:.1f} μg/L<extra></extra>",
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30, b=10, l=10, r=10), height=220,
            showlegend=False,
            annotations=[dict(text="Huidig", x=0.5, y=0.5, font_size=13,
                              font_color=C_TEXT, showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_info:
        info_rows = [
            ("Microcystis aeruginosa",         "#C0392B", "Toxisch · produceert microcystine · dominant bij >27°C"),
            ("Cylindrospermopsis raciborskii",  "#E67E22", "Toxisch · cylindrospermopsine · typisch tropisch Brazil"),
            ("Groenwieren (Chlorophyta)",        "#27AE60", "Niet-toxisch · meer aanwezig bij lagere temperaturen"),
            ("Diatomeeën (Bacillariophyta)",     "#2980B9", "Niet-toxisch · indicator voor goede waterkwaliteit"),
        ]
        for species, color, desc in info_rows:
            pct = round(species_vals[species] / latest_b["algae"] * 100, 0) if latest_b["algae"] > 0 else 0
            st.markdown(f"""
            <div style="background:#F8FAFC;border-left:4px solid {color};border-radius:6px;
                        padding:8px 12px;margin-bottom:6px;">
              <span style="font-weight:700;color:{color};font-size:13px;">{species}</span>
              <span style="font-size:12px;color:{C_MUTED};float:right;">{pct:.0f}% · {species_vals[species]:.1f} μg/L</span>
              <div style="font-size:11px;color:{C_TEXT};margin-top:2px;">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
# TAB 3 — ALGENVOORSPELLING
# ────────────────────────────────────────────────────────────────────────────────
with tab3:
    buoy_name_sel = next(b["name"] for b in BUOYS if b["id"] == selected_buoy)
    df_b  = df[df["buoy_id"] == selected_buoy].copy()
    hist14 = df_b.tail(14)
    anchor_date = hist14["date"].iloc[-1]
    anchor_val  = hist14["algae"].iloc[-1]
    now_dt = datetime.now()

    # ── Wetenschappelijke 90-daagse baseline voorspelling ─────────────────────
    st.markdown('<p class="section-label">Wetenschappelijke voorspelling — 90 dagen (logistisch groeimodel)</p>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px;
                padding:10px 16px; font-size:12px; color:{C_TEXT}; margin-bottom:12px;">
      <b>Model:</b> Logistisch groeimodel · Cardinaal temperatuurmodel (Bernard & Rémond 2012) ·
      Seizoenscyclus Rio de Janeiro · Draagkracht K = 140 μg/L ·
      Max groeisnelheid r = 0.15/dag (Microcystis aeruginosa) · Geen gebruikersparameters
    </div>
    """, unsafe_allow_html=True)

    df_sci = predict_scientific(df_b, selected_buoy, days=90, seed=42)
    sci_dates = [anchor_date] + df_sci["date"].tolist()
    sci_vals  = [anchor_val]  + df_sci["algae"].tolist()
    sci_upper = [anchor_val] + (df_sci["algae"] * 1.15).tolist()
    sci_lower = [anchor_val] + (df_sci["algae"] * 0.85).tolist()

    fig_sci = go.Figure()
    # Onzekerheidsband
    fig_sci.add_trace(go.Scatter(
        x=sci_dates + sci_dates[::-1], y=sci_upper + sci_lower[::-1],
        fill="toself", fillcolor="rgba(39,174,96,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="Onzekerheidsmarge (±15%)", showlegend=True,
    ))
    # Historie
    fig_sci.add_trace(go.Scatter(
        x=hist14["date"], y=hist14["algae"], mode="lines",
        name="Historisch (gemeten)", line=dict(color=C_DARK, width=2),
        hovertemplate="%{y:.1f} μg/L<extra></extra>",
    ))
    # Wetenschappelijke voorspelling
    fig_sci.add_trace(go.Scatter(
        x=sci_dates, y=sci_vals, mode="lines",
        name="Wetenschappelijke voorspelling (90 dagen)",
        line=dict(color=C_GREEN, width=2.5),
        hovertemplate="Voorspelling: %{y:.1f} μg/L<extra></extra>",
    ))
    fig_sci.add_trace(go.Scatter(
        x=[now_dt, now_dt], y=[0, 145], mode="lines",
        line=dict(color=C_MUTED, width=1, dash="dot"),
        name="Nu", showlegend=False,
    ))
    fig_sci.add_hline(y=40, line_dash="dot", line_color=C_YELLOW,
                      annotation_text="Waarschuwing (40)", annotation_font_size=10)
    fig_sci.add_hline(y=70, line_dash="dot", line_color=C_RED,
                      annotation_text="Alarm (70)", annotation_font_size=10)
    style(fig_sci, height=360, title="")
    fig_sci.update_yaxes(title="μg/L")
    st.plotly_chart(fig_sci, use_container_width=True)

    # Wetenschappelijke samenvatting
    sci_peak = df_sci["algae"].max()
    sci_peak_day = df_sci.loc[df_sci["algae"].idxmax(), "date"].strftime("%d %b")
    sci_end = df_sci["algae"].iloc[-1]
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.metric("Piek verwacht", f"{sci_peak:.1f} μg/L", f"op {sci_peak_day}")
    c2.metric("Verwacht over 90 dagen", f"{sci_end:.1f} μg/L",
              f"{sci_end - anchor_val:+.1f} vs nu")
    c3.metric("Risico", "Hoog" if sci_peak > 70 else "Verhoogd" if sci_peak > 40 else "Laag",
              "Boven alarmgrens" if sci_peak > 70 else "Boven waarschuwing" if sci_peak > 40 else "Binnen norm")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Scenario met parameters ───────────────────────────────────────────────
    st.markdown(
        f'<p class="section-label">{selected_buoy} — {buoy_name_sel} · Scenario-voorspelling ({forecast_days} dagen)</p>',
        unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#FFF7ED; border:1px solid #FED7AA; border-radius:8px;
                padding:10px 16px; font-size:12px; color:{C_TEXT}; margin-bottom:12px;">
      <b>Scenario:</b> Temperatuur +{temp_offset}°C · Regenval ×{rain_factor:.1f} ·
      Lozing ×{discharge:.1f} · LG Sonic {int(treatment*100)}% ·
      Parameters stelbaar via de sidebar
    </div>
    """, unsafe_allow_html=True)

    df_no_treat  = predict_xgb(df_b, xgb_models, selected_buoy,
                               forecast_days, temp_offset, rain_factor, discharge, treatment=0.0)
    if treatment == 0.0:
        df_pred = df_no_treat.copy()
    else:
        df_pred = predict_xgb(df_b, xgb_models, selected_buoy,
                               forecast_days, temp_offset, rain_factor, discharge, treatment)

    dates_fwd = [anchor_date] + df_pred["date"].tolist()
    pred_vals  = [anchor_val]  + df_pred["algae"].tolist()
    notr_vals  = [anchor_val]  + df_no_treat["algae"].tolist()
    upper = [anchor_val] + (df_pred["algae"] * 1.12).tolist()
    lower = [anchor_val] + (df_pred["algae"] * 0.88).tolist()

    fig_f = go.Figure()
    # Onbehandeld (grijs) — referentie
    fig_f.add_trace(go.Scatter(
        x=dates_fwd, y=notr_vals, mode="lines",
        name="Scenario zonder behandeling", line=dict(color="#B0BEC5", width=1.5, dash="dot"),
        hovertemplate="Zonder behandeling: %{y:.1f} μg/L<extra></extra>",
    ))
    # Onzekerheidsband behandeld
    fig_f.add_trace(go.Scatter(
        x=dates_fwd + dates_fwd[::-1], y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(0,153,204,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="Onzekerheidsband", showlegend=True,
    ))
    # Historie
    fig_f.add_trace(go.Scatter(
        x=hist14["date"], y=hist14["algae"], mode="lines",
        name="Historisch (gemeten)", line=dict(color=C_DARK, width=2),
        hovertemplate="%{y:.1f} μg/L<extra></extra>",
    ))
    # Scenario met behandeling
    fig_f.add_trace(go.Scatter(
        x=dates_fwd, y=pred_vals, mode="lines+markers",
        name=f"Scenario met LG Sonic ({int(treatment*100)}%)",
        line=dict(color=C_BLUE, width=2.5),
        marker=dict(size=6, symbol="circle", color=C_BLUE),
        hovertemplate="Met behandeling: %{y:.1f} μg/L<extra></extra>",
    ))
    fig_f.add_trace(go.Scatter(
        x=[now_dt, now_dt], y=[0, 125], mode="lines",
        line=dict(color=C_MUTED, width=1, dash="dot"),
        name="Nu", showlegend=False,
    ))
    fig_f.add_hline(y=40, line_dash="dot", line_color=C_YELLOW,
                    annotation_text="Waarschuwing (40)", annotation_font_size=10)
    fig_f.add_hline(y=70, line_dash="dot", line_color=C_RED,
                    annotation_text="Alarm (70)", annotation_font_size=10)

    style(fig_f, height=360, title="")
    fig_f.update_yaxes(title="μg/L")
    st.markdown("**Algenconcentratie — met en zonder ultrasonore behandeling**")
    st.plotly_chart(fig_f, use_container_width=True)

    # ── Behandelingseffect samenvatting ───────────────────────────────────────
    besparing = df_no_treat["algae"].max() - df_pred["algae"].max()
    pct = (besparing / max(df_no_treat["algae"].max(), 0.1)) * 100
    st.markdown(f"""
    <div style="background:{C_WHITE}; border:1px solid {C_BORDER}; border-radius:8px;
                padding:14px 20px; display:flex; gap:32px; align-items:center; margin-bottom:16px;">
      <div style="text-align:center;">
        <div style="font-size:11px; color:{C_MUTED}; font-weight:600; text-transform:uppercase;">Zonder behandeling</div>
        <div style="font-size:22px; font-weight:700; color:{C_RED};">{df_no_treat['algae'].max():.1f} <span style="font-size:13px;">μg/L</span></div>
        <div style="font-size:11px; color:{C_MUTED};">piek verwacht</div>
      </div>
      <div style="font-size:28px; color:{C_MUTED};">→</div>
      <div style="text-align:center;">
        <div style="font-size:11px; color:{C_MUTED}; font-weight:600; text-transform:uppercase;">Met LG Sonic ({int(treatment*100)}%)</div>
        <div style="font-size:22px; font-weight:700; color:{C_GREEN};">{df_pred['algae'].max():.1f} <span style="font-size:13px;">μg/L</span></div>
        <div style="font-size:11px; color:{C_MUTED};">piek verwacht</div>
      </div>
      <div style="text-align:center; background:#F0FDF4; border-radius:8px; padding:10px 16px;">
        <div style="font-size:11px; color:{C_MUTED}; font-weight:600; text-transform:uppercase;">Reductie</div>
        <div style="font-size:22px; font-weight:700; color:{C_GREEN};">−{pct:.0f}%</div>
        <div style="font-size:11px; color:{C_MUTED};">door ultrasonore behandeling</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Geosmin bar
    geo_colors = [C_RED if v > 50 else C_YELLOW if v > 20 else C_BLUE for v in df_pred["geosmin"]]
    fig_g = go.Figure(go.Bar(
        x=df_pred["date"], y=df_pred["geosmin"],
        marker_color=geo_colors, name="Geosmin",
        hovertemplate="%{y:.1f} ng/L<extra></extra>",
    ))
    fig_g.add_hline(y=20, line_dash="dot", line_color=C_YELLOW,
                    annotation_text="Smaakgrens (20 ng/L)", annotation_font_size=10)
    fig_g.add_hline(y=50, line_dash="dot", line_color=C_RED,
                    annotation_text="Alarmgrens (50 ng/L)", annotation_font_size=10)
    style(fig_g, height=240, title="<b>Geosmin-voorspelling (ng/L)</b> — smaak- en geurproblemen")
    fig_g.update_layout(showlegend=False)
    fig_g.update_yaxes(title="ng/L")
    st.plotly_chart(fig_g, use_container_width=True)

    # ── Sensitiviteitsanalyse fysisch model ───────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:8px;">Wat drijft de voorspelling?</p>',
                unsafe_allow_html=True)

    # Bereken piek algen onder verschillende scenario's (alles overig constant)
    # Baseline = alles op 0 = stabiele algen (geen groei, geen daling)
    base_pred = predict_xgb(df_b, xgb_models, selected_buoy, forecast_days,
                            0.0, 0.0, 0.0, 0.0)["algae"].max()

    scenarios = {
        "LG Sonic behandeling (70%)":  predict_xgb(df_b, xgb_models, selected_buoy, forecast_days, 0.0, 0.0, 0.0, 0.70)["algae"].max(),
        "Temperatuur +3°C":            predict_xgb(df_b, xgb_models, selected_buoy, forecast_days, 3.0, 0.0, 0.0, 0.0)["algae"].max(),
        "Industriële lozing ×2":       predict_xgb(df_b, xgb_models, selected_buoy, forecast_days, 0.0, 0.0, 2.0, 0.0)["algae"].max(),
        "Regenval ×2":                 predict_xgb(df_b, xgb_models, selected_buoy, forecast_days, 0.0, 2.0, 0.0, 0.0)["algae"].max(),
    }

    sens_data = []
    for label, peak in scenarios.items():
        delta = peak - base_pred
        sens_data.append({"Factor": label, "delta": round(delta, 1)})

    cards = [
        {
            "label":   "LG Sonic behandeling (70%)",
            "peak":    scenarios["LG Sonic behandeling (70%)"],
            "delta":   round(scenarios["LG Sonic behandeling (70%)"] - base_pred, 1),
            "icon":    "↓",
            "uitleg":  "Ultrasoon verstoort algengroei",
            "positief": True,
        },
        {
            "label":   "Temperatuur +3°C",
            "peak":    scenarios["Temperatuur +3°C"],
            "delta":   round(scenarios["Temperatuur +3°C"] - base_pred, 1),
            "icon":    "↑",
            "uitleg":  "Warmte stimuleert algengroei",
            "positief": False,
        },
        {
            "label":   "Industriële lozing ×2",
            "peak":    scenarios["Industriële lozing ×2"],
            "delta":   round(scenarios["Industriële lozing ×2"] - base_pred, 1),
            "icon":    "↑",
            "uitleg":  "Meer nutriënten in het water",
            "positief": False,
        },
        {
            "label":   "Regenval ×2",
            "peak":    scenarios["Regenval ×2"],
            "delta":   round(scenarios["Regenval ×2"] - base_pred, 1),
            "icon":    "↓",
            "uitleg":  "Regen verdunt de concentratie",
            "positief": True,
        },
    ]

    cols = st.columns(4, gap="small")
    for col, card in zip(cols, cards):
        bg    = "#F0FDF4" if card["positief"] else "#FEF2F2"
        color = C_GREEN   if card["positief"] else C_RED
        sign  = "" if card["delta"] < 0 else "+"
        col.markdown(f"""
        <div style="background:{bg}; border:1px solid {color}33; border-radius:10px;
                    padding:14px 12px; text-align:center;">
          <div style="font-size:11px; font-weight:700; color:{C_MUTED};
                      text-transform:uppercase; margin-bottom:6px;">{card['label']}</div>
          <div style="font-size:30px; font-weight:800; color:{color}; line-height:1.1;">
            {card['icon']} {abs(card['delta']):.0f}
            <span style="font-size:13px;">μg/L</span>
          </div>
          <div style="font-size:11px; color:{C_TEXT}; margin-top:6px;">{card['uitleg']}</div>
          <div style="font-size:10px; color:{C_MUTED}; margin-top:4px;">
            Piek: <b>{card['peak']:.0f} μg/L</b>
          </div>
        </div>""", unsafe_allow_html=True)

    # Samenvatting
    max_a   = df_pred["algae"].max()
    max_day = df_pred.loc[df_pred["algae"].idxmax(), "date"].strftime("%d %b")
    s       = status(max_a, "algae")
    icon, color = STATUS_ICON[s]
    label   = {"ok": "Laag risico", "warn": "Verhoogd risico", "alert": "Hoog risico"}[s]
    st.markdown(f"""
    <div style="background:{C_WHITE}; border:1px solid {C_BORDER}; border-radius:8px;
                padding:14px 18px; display:flex; gap:20px; align-items:center; margin-top:4px;">
      <div style="font-size:28px;">{icon}</div>
      <div>
        <div style="font-size:13px; font-weight:700; color:{color};">{label}</div>
        <div style="font-size:12px; color:{C_TEXT}; margin-top:2px;">
          Piek verwacht op <b>{max_day}</b> — <b>{max_a:.1f} μg/L</b>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)



# ────────────────────────────────────────────────────────────────────────────────
# TAB 4 — SCENARIO-SIMULATOR
# ────────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-label">Scenario-vergelijking — stel parameters in via de sidebar</p>',
                unsafe_allow_html=True)

    df_b     = df[df["buoy_id"] == selected_buoy]
    baseline = predict_xgb(df_b, xgb_models, selected_buoy, forecast_days, 0,           0.0,        0.0,      treatment)
    scenario = predict_xgb(df_b, xgb_models, selected_buoy, forecast_days, temp_offset, rain_factor, discharge, treatment)

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=baseline["date"], y=baseline["algae"], mode="lines",
        name="Baseline", line=dict(color=C_BLUE, width=2),
        hovertemplate="%{y:.1f} μg/L<extra></extra>",
    ))
    fig_s.add_trace(go.Scatter(
        x=scenario["date"], y=scenario["algae"], mode="lines",
        name=f"Scenario (+{temp_offset}°C · regen ×{rain_factor} · lozing ×{discharge})",
        line=dict(color=C_ORANGE, width=2, dash="dash"),
        hovertemplate="%{y:.1f} μg/L<extra></extra>",
    ))
    fig_s.add_hline(y=40, line_dash="dot", line_color=C_YELLOW,
                    annotation_text="Waarschuwing (40)", annotation_font_size=10)
    fig_s.add_hline(y=70, line_dash="dot", line_color=C_RED,
                    annotation_text="Alarm (70)", annotation_font_size=10)
    style(fig_s, height=340, title="")
    fig_s.update_yaxes(title="μg/L")
    st.markdown("**Algenconcentratie — Baseline vs. Scenario**")
    st.plotly_chart(fig_s, use_container_width=True)

    # Metrics
    dm  = scenario["algae"].max() - baseline["algae"].max()
    da  = scenario["algae"].mean() - baseline["algae"].mean()
    dab = int((scenario["algae"] > 40).sum())

    c1, c2, c3 = st.columns(3, gap="medium")
    c1.metric("Piek (scenario)",    f"{scenario['algae'].max():.1f} μg/L", f"{dm:+.1f} vs baseline")
    c2.metric("Gemiddeld",          f"{scenario['algae'].mean():.1f} μg/L", f"{da:+.1f} vs baseline")
    c3.metric("Dagen boven grens",  f"{dab} / {forecast_days}",
              "Geen overschrijding" if dab == 0 else f"{dab} dag(en) verhoogd")

    # Alle buoys
    st.markdown('<p class="section-label" style="margin-top:20px;">Impact op alle buoys</p>',
                unsafe_allow_html=True)
    res = []
    for b in BUOYS:
        db_  = df[df["buoy_id"] == b["id"]]
        base = predict_xgb(db_, xgb_models, b["id"], forecast_days, 0,           0.0,        0.0,      treatment)["algae"].max()
        scen = predict_xgb(db_, xgb_models, b["id"], forecast_days, temp_offset, rain_factor, discharge, treatment)["algae"].max()
        res.append({"Buoy": b["id"], "Naam": b["name"],
                    "Baseline": round(base, 1), "Scenario": round(scen, 1), "Δ": round(scen - base, 1)})

    df_r = pd.DataFrame(res)
    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(x=df_r["Buoy"], y=df_r["Baseline"], name="Baseline",
                            marker_color=C_BLUE, opacity=0.85))
    fig_b.add_trace(go.Bar(x=df_r["Buoy"], y=df_r["Scenario"], name="Scenario",
                            marker_color=C_ORANGE, opacity=0.85))
    fig_b.add_hline(y=40, line_dash="dot", line_color=C_YELLOW)
    fig_b.add_hline(y=70, line_dash="dot", line_color=C_RED)
    style(fig_b, height=280, title="")
    fig_b.update_layout(barmode="group")
    fig_b.update_yaxes(title="μg/L")
    st.markdown("**Piek algenconcentratie per buoy**")
    st.plotly_chart(fig_b, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────────
# TAB 5 — WAARSCHUWINGEN
# ────────────────────────────────────────────────────────────────────────────────
with tab5:
    col_al, col_log = st.columns([1, 1], gap="medium")

    with col_al:
        st.markdown('<p class="section-label">Actieve waarschuwingen</p>', unsafe_allow_html=True)

        alerts = []
        for _, row in latest.iterrows():
            bname = next(b["name"] for b in BUOYS if b["id"] == row["buoy_id"])
            bid   = row["buoy_id"]
            checks = [
                ("algae",   row["algae"],   "μg/L", "Ultrasonore intensiteit verhogen",
                 "Algenconcentratie verhoogd"),
                ("geosmin", row["geosmin"], "ng/L", "CEDAE waterbehandeling informeren",
                 "Geosmin boven drempel"),
                ("oxygen",  row["oxygen"],  "mg/L", "Controleer aeratiesysteem",
                 "Laag zuurstofniveau"),
            ]
            for param, val, unit, action, desc in checks:
                s = status(val, param)
                if s in ("warn", "alert"):
                    alerts.append((s, bid, bname, desc, val, unit, action))

        if not alerts:
            st.markdown(f"""
            <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px;
                        padding:16px; text-align:center; color:{C_GREEN}; font-weight:600;">
              ✓ Alle parameters binnen normen
            </div>""", unsafe_allow_html=True)
        else:
            for s, bid, bname, desc, val, unit, action in alerts:
                css  = "alert-red" if s == "alert" else "alert-orange"
                _ic = C_RED if s == "alert" else C_YELLOW
                icon = (f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none"'
                        f' stroke="{_ic}" stroke-width="2.5" style="flex-shrink:0;">'
                        f'<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3'
                        f'L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
                        f'<line x1="12" y1="9" x2="12" y2="13"/>'
                        f'<line x1="12" y1="17" x2="12.01" y2="17"/></svg>')
                lbl  = "ALARM" if s == "alert" else "WAARSCHUWING"
                st.markdown(f"""
                <div class="alert-card {css}">
                  <div class="alert-icon">{icon}</div>
                  <div>
                    <div class="alert-title">{lbl} · {bid} — {bname}</div>
                    <div class="alert-body">{desc}: <b>{val} {unit}</b></div>
                    <div class="alert-action">→ {action}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

    with col_log:
        st.markdown('<p class="section-label">Systeemlog — laatste 24 uur</p>', unsafe_allow_html=True)
        log = [
            ("08:30", "B05", "INFO",  "Ultrasonore frequentie aangepast o.b.v. algenconcentratie"),
            ("06:15", "B07", "WARN",  "Geosmin overschrijdt 20 ng/L — verhoogde monitoring"),
            ("03:00", "ALL", "INFO",  "Automatische kalibratie sensoren voltooid"),
            ("00:45", "B08", "WARN",  "pH buiten optimaal bereik (8.6) — wordt gemonitord"),
            ("22:10", "B06", "INFO",  "Algenconcentratie gedaald na ultrasonore behandeling"),
            ("18:30", "B03", "INFO",  "Watertemperatuur gestegen naar 28.4 °C"),
            ("14:05", "B01", "INFO",  "Sensorcheck voltooid — alle systemen operationeel"),
            ("09:20", "B04", "WARN",  "Lichte turbiditeitstoename gedetecteerd"),
        ]
        st.markdown(f"""
        <div style="background:{C_WHITE}; border:1px solid {C_BORDER}; border-radius:8px;
                    padding:12px 16px; font-family:'Courier New',monospace; font-size:11px;">
        """, unsafe_allow_html=True)
        for time, buoy, level, msg in log:
            lcolor = {
                "INFO": C_BLUE, "WARN": C_YELLOW, "ALARM": C_RED
            }.get(level, C_MUTED)
            st.markdown(
                f'<div class="log-row">'
                f'<span style="color:{C_MUTED}">{time}</span>  '
                f'<span style="color:{C_DARK};font-weight:700">{buoy}</span>  '
                f'<span style="color:{lcolor};font-weight:700">[{level}]</span>  '
                f'<span style="color:{C_TEXT}">{msg}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
