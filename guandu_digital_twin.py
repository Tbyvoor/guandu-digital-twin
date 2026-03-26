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
import requests
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
C_TEXT    = "#FFFFFF"
C_MUTED   = "#FFFFFF"
C_WHITE   = "#FFFFFF"
C_BORDER  = "#DDE3E9"

C_CHART_BG = "#111E2D"   # donker achtergrond voor 2D grafieken

CHART = dict(
    plot_bgcolor=C_CHART_BG,
    paper_bgcolor=C_CHART_BG,
    font=dict(family="Inter, sans-serif", color=C_WHITE, size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1E3A5F", borderwidth=1,
                font=dict(size=11, color=C_WHITE), orientation="h", yanchor="bottom", y=1.02),
)

def style(fig, height=280, title="", margin_l=12):
    fig.update_layout(**CHART, height=height,
                      margin=dict(l=margin_l, r=12, t=36, b=12),
                      title=dict(text=title, font=dict(size=13, color="#CBD5E1")))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#1E3A5F",
                     tickfont=dict(size=11, color="#94A3B8"))
    fig.update_yaxes(gridcolor="#1A2E45", zeroline=False, linecolor="#1E3A5F",
                     tickfont=dict(size=11, color="#94A3B8"))
    return fig

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {C_WHITE}; }}

  /* Global text override — p en labels wit, geen span zodat inline kleuren bewaard blijven */
  p, label, li, td, th {{ color: {C_WHITE}; }}

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
  .stApp, .main {{ background: {C_DARK} !important; }}
  .main .block-container {{ background: {C_DARK}; padding-top: 1.5rem; }}

  /* Streamlit native text elements — geen !important op span zodat inline kleuren bewaard blijven */
  .stMarkdown, .stMarkdown p, .stCaption, .stCaption p,
  [data-testid="stMarkdownContainer"] p {{ color: {C_WHITE} !important; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid #1E3A5F; background: {C_DARK}; }}
  .stTabs [data-baseweb="tab"] {{
    font-size: 13px; font-weight: 600; color: {C_WHITE};
    padding: 8px 18px; border-radius: 6px 6px 0 0;
    background: transparent; border: none;
  }}
  .stTabs [aria-selected="true"] {{ color: {C_BLUE} !important; border-bottom: 2px solid {C_BLUE}; }}

  /* KPI cards */
  .kpi-card {{
    background: {C_MID};
    border-radius: 10px;
    padding: 18px 20px;
    border: 1px solid #1E3A5F;
    box-shadow: 0 1px 6px rgba(0,0,0,0.3);
  }}
  .kpi-label {{ font-size: 11px; font-weight: 600; color: {C_WHITE}; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 26px; font-weight: 700; color: {C_WHITE}; line-height: 1.1; }}
  .kpi-sub   {{ font-size: 11px; color: {C_WHITE}; margin-top: 4px; }}
  .kpi-ok    {{ color: {C_GREEN}; }}
  .kpi-warn  {{ color: {C_YELLOW}; }}
  .kpi-alert {{ color: {C_RED}; }}

  /* Buoy status cards */
  .buoy-card {{
    background: {C_MID};
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .buoy-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .buoy-id  {{ font-size: 12px; font-weight: 700; color: {C_WHITE}; min-width: 28px; }}
  .buoy-name{{ font-size: 12px; color: {C_WHITE}; flex: 1; }}
  .buoy-vals{{ font-size: 11px; color: {C_WHITE}; text-align: right; }}

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
  .alert-red    {{ background: rgba(192,57,43,0.15); border-color: rgba(192,57,43,0.4); }}
  .alert-orange {{ background: rgba(243,156,18,0.12); border-color: rgba(243,156,18,0.4); }}
  .alert-icon   {{ line-height: 1; display:flex; align-items:flex-start; padding-top:2px; }}
  .alert-title  {{ font-size: 13px; font-weight: 700; color: {C_WHITE}; }}
  .alert-body   {{ font-size: 12px; color: {C_WHITE}; margin-top: 2px; }}
  .alert-action {{ font-size: 11px; color: {C_WHITE}; margin-top: 6px; font-style: italic; }}

  /* Log */
  .log-row {{ font-family: 'Courier New', monospace; font-size: 11px; padding: 4px 0; border-bottom: 1px solid #1E3A5F; color: {C_WHITE}; }}

  /* Section labels */
  .section-label {{
    font-size: 11px; font-weight: 700; color: {C_WHITE};
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 12px; margin-top: 4px;
  }}

  /* Divider */
  hr {{ border: none; border-top: 1px solid #1E3A5F; margin: 16px 0; }}

  /* Ronde hoeken voor grafieken */
  [data-testid="stPlotlyChart"] > div,
  [data-testid="stPlotlyChart"] iframe {{
    border-radius: 14px !important;
    overflow: hidden;
  }}

  /* Hide streamlit branding */
  #MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── Constanten ─────────────────────────────────────────────────────────────────
BUOYS = [
    {"id": "B01", "name": "Nascente",       "lat": -22.691295, "lon": -43.862656},
    {"id": "B02", "name": "Seropédica",     "lat": -22.683694, "lon": -43.817954},
    {"id": "B03", "name": "Centraal",       "lat": -22.688539, "lon": -43.778841},
    {"id": "B04", "name": "Captação Norte", "lat": -22.660206, "lon": -43.739728},
    {"id": "B05", "name": "CEDAE Inname",   "lat": -22.648723, "lon": -43.698277},
    {"id": "B06", "name": "Station 6",      "lat": -22.655916, "lon": -43.652944},
    {"id": "B07", "name": "Stroomafwaarts", "lat": -22.675860, "lon": -43.648575},
    {"id": "B08", "name": "Uitstroom",      "lat": -22.719731, "lon": -43.638321},
]

# ── Rivierlijn coördinaten (OpenStreetMap / Overpass API) ───────────────────────
# 594 waypoints langs de Rio Guandu — aaneengesloten pad, west → oost.
# Gebied: Paracambi/Pte.Coberta → Japeri (lon -43.86 t/m -43.64).
RIVER_PATH = [
    (-22.6912952,-43.8626561),(-22.6914615,-43.8624061),(-22.6915216,-43.8622519),(-22.6921026,-43.8609231),(-22.6920493,-43.8603396),(-22.6919054,-43.8597272),(-22.6917082,-43.8592188),(-22.6916974,-43.8585207),(-22.6918541,-43.8575581),(-22.6921153,-43.8568361),(-22.6923765,-43.8560575),(-22.6928859,-43.8548966),(-22.6931341,-43.8542595),(-22.6934606,-43.853396),(-22.6936695,-43.8528722),(-22.6938002,-43.8519803),(-22.6938393,-43.8511875),(-22.6935912,-43.8506071),(-22.6934083,-43.849602),(-22.6930557,-43.849064),(-22.6925594,-43.848795),(-22.6919847,-43.8484553),(-22.6915014,-43.8481155),(-22.6910966,-43.8479456),(-22.6903129,-43.8476342),(-22.689908,-43.8476766),(-22.6893464,-43.847705),(-22.6889397,-43.8475851),(-22.6885734,-43.8472632),(-22.6881676,-43.8466517),(-22.6875638,-43.8456968),(-22.6870392,-43.8451067),(-22.686663,-43.8441089),(-22.6866531,-43.8429609),(-22.6869006,-43.8420061),(-22.6871183,-43.8412014),(-22.6872173,-43.8405684),(-22.6871975,-43.8399461),(-22.6869699,-43.8392702),(-22.6866234,-43.8384119),(-22.6863363,-43.8377253),(-22.6860691,-43.8371245),(-22.6858117,-43.8363842),(-22.6856735,-43.8353987),(-22.6857649,-43.8348607),(-22.6860131,-43.834252),(-22.6862613,-43.8339122),(-22.6865356,-43.8334309),(-22.6869013,-43.8329779),(-22.6871756,-43.8323266),(-22.6872017,-43.8319302),(-22.6872017,-43.8314206),(-22.6871625,-43.8308402),(-22.6870711,-43.8302173),(-22.6870058,-43.8297218),(-22.6867968,-43.8289149),(-22.6867445,-43.8283061),(-22.6867811,-43.8266934),(-22.6866106,-43.8251283),(-22.6865037,-43.8243043),(-22.6862935,-43.8222623),(-22.6861778,-43.8220298),(-22.6854872,-43.8211891),(-22.685323,-43.8210006),(-22.6849851,-43.8205154),(-22.6848348,-43.8202308),(-22.6847104,-43.8198438),(-22.6841078,-43.8186284),(-22.6836942,-43.8179545),(-22.6835036,-43.8177369),(-22.6829893,-43.8173152),(-22.6825437,-43.8168362),(-22.6822329,-43.8164377),(-22.6820807,-43.8161124),(-22.6818236,-43.8158205),(-22.6817591,-43.8156912),(-22.6817106,-43.8154716),(-22.6816589,-43.8152592),(-22.6816604,-43.8150103),(-22.681668,-43.8148931),(-22.6816814,-43.8146891),(-22.6817135,-43.8145085),(-22.6817729,-43.8143962),(-22.6818694,-43.8143196),(-22.6819793,-43.8142609),(-22.6820928,-43.8142167),(-22.6821798,-43.8142159),(-22.6823976,-43.8142645),(-22.6826189,-43.8143348),(-22.6830259,-43.8144813),(-22.6832205,-43.8145554),(-22.6834453,-43.8146544),(-22.6836937,-43.8147569),(-22.6838952,-43.8148634),(-22.6840477,-43.8148872),(-22.6843158,-43.8149498),(-22.6845403,-43.8149912),(-22.6847384,-43.8150832),(-22.6849198,-43.8151791),(-22.6850845,-43.8152895),(-22.6852423,-43.8153639),(-22.6854102,-43.8154418),(-22.685728,-43.8155839),(-22.6859494,-43.8156686),(-22.6862206,-43.8156951),(-22.6864646,-43.8156606),(-22.6867553,-43.8156148),(-22.6869822,-43.8155371),(-22.6870788,-43.8154749),(-22.6871649,-43.8153479),(-22.6871972,-43.8151926),(-22.6872054,-43.8149472),(-22.6871539,-43.8147709),(-22.6870418,-43.8145446),(-22.6868495,-43.8143298),(-22.6866873,-43.8141148),(-22.6865957,-43.8139424),(-22.6865574,-43.8137336),(-22.6865817,-43.8134051),(-22.6867389,-43.8128318),(-22.6868601,-43.8124808),(-22.6870273,-43.8120177),(-22.687281,-43.8114888),(-22.6874571,-43.8111827),(-22.6876288,-43.810871),(-22.687741,-43.8106681),(-22.6877966,-43.8104908),(-22.6878655,-43.8103171),(-22.6879341,-43.8100929),(-22.6880292,-43.8098359),(-22.6881515,-43.8096329),(-22.6882644,-43.8095165),(-22.6884179,-43.8094538),(-22.6886318,-43.8094339),(-22.6889359,-43.8095385),(-22.6891908,-43.8096192),(-22.6893349,-43.8096469),(-22.6894587,-43.8096566),(-22.6896767,-43.8095953),(-22.689779,-43.809542),(-22.6899509,-43.8093561),(-22.6900492,-43.8091686),(-22.6900771,-43.8091153),(-22.6900619,-43.8088831),(-22.6899859,-43.8086223),(-22.6898632,-43.8084345),(-22.6897537,-43.8081958),(-22.6896775,-43.8079132),(-22.6896416,-43.8076158),(-22.6896387,-43.8072236),(-22.6896691,-43.8067803),(-22.6897927,-43.8062055),(-22.6899514,-43.8057975),(-22.6900944,-43.8055089),(-22.6902802,-43.805193),(-22.6904992,-43.8048608),(-22.6907111,-43.8045765),(-22.6908828,-43.8043402),(-22.6910142,-43.8041521),(-22.6911224,-43.8038049),(-22.6911946,-43.8034918),(-22.6912189,-43.8033863),(-22.6912723,-43.8030791),(-22.6912966,-43.8027669),(-22.6913527,-43.8023805),(-22.6913764,-43.8020858),(-22.6913554,-43.8017518),(-22.6912974,-43.8014021),(-22.691232,-43.8010313),(-22.6911591,-43.8006539),(-22.6911017,-43.8003918),(-22.691018,-43.800091),(-22.6908915,-43.7999122),(-22.6907349,-43.7996987),(-22.6905147,-43.7994408),(-22.6903696,-43.7992789),(-22.6902354,-43.799109),(-22.690131,-43.7989587),(-22.6900112,-43.7987329),(-22.6899026,-43.798523),(-22.689846,-43.7983563),(-22.6897923,-43.7980942),(-22.6897334,-43.7976673),(-22.6896631,-43.7971546),(-22.6896189,-43.7966776),(-22.6895926,-43.7961287),(-22.6895741,-43.7956235),(-22.6895662,-43.7951165),(-22.689626,-43.7947141),(-22.689679,-43.7944033),(-22.6897025,-43.7940927),(-22.6897152,-43.7938181),(-22.6897579,-43.7936148),(-22.6898777,-43.7933551),(-22.6900344,-43.7930832),(-22.6901628,-43.7929249),(-22.6902769,-43.7928801),(-22.6904245,-43.7928868),(-22.6906023,-43.7929648),(-22.6907059,-43.7930077),(-22.6908498,-43.7930024),(-22.6910378,-43.7929729),(-22.6912216,-43.7928758),(-22.6913973,-43.7926793),(-22.6916,-43.792402),(-22.6917169,-43.7922418),(-22.6917751,-43.7921259),(-22.6917964,-43.7920183),(-22.6917768,-43.7918712),(-22.6917201,-43.7916847),(-22.6915818,-43.7914512),(-22.6914691,-43.7911935),(-22.691353,-43.7909677),(-22.6912186,-43.79077),(-22.6911106,-43.7906356),(-22.6910174,-43.7905051),(-22.6909646,-43.7903663),(-22.6909781,-43.7901911),(-22.6910279,-43.789948),(-22.6911762,-43.7895528),(-22.6912958,-43.7892629),(-22.6914555,-43.7889073),(-22.6915533,-43.7886638),(-22.6916073,-43.7884842),(-22.6916392,-43.7883089),(-22.6916887,-43.7880339),(-22.6917275,-43.7877948),(-22.6917783,-43.7875819),(-22.6918253,-43.7874581),(-22.6918872,-43.7873382),(-22.6919445,-43.7871109),(-22.6919797,-43.7868838),(-22.6920156,-43.7867561),(-22.6921321,-43.7865442),(-22.6922856,-43.7863479),(-22.6924166,-43.7861),(-22.6924962,-43.7858964),(-22.692563,-43.7856931),(-22.6925828,-43.785633),(-22.6926694,-43.7853696),(-22.6926857,-43.785091),(-22.6926458,-43.7847306),(-22.6924406,-43.7842261),(-22.692357,-43.7841527),(-22.6918808,-43.7834501),(-22.6917858,-43.7828751),(-22.6914691,-43.7819309),(-22.690598,-43.7811584),(-22.6898061,-43.7797851),(-22.6885391,-43.778841),(-22.6886975,-43.7779827),(-22.6883807,-43.7771244),(-22.6878264,-43.7754936),(-22.686797,-43.774292),(-22.6871137,-43.7729187),(-22.6861634,-43.7713737),(-22.6854507,-43.7712021),(-22.684738,-43.7710304),(-22.6844212,-43.7703438),(-22.6837877,-43.7688847),(-22.682679,-43.7662239),(-22.6819663,-43.7656231),(-22.6807784,-43.7655373),(-22.6786402,-43.7644215),(-22.6773731,-43.7640781),(-22.674918,-43.7624473),(-22.6736508,-43.7598724),(-22.6738884,-43.7584133),(-22.6740468,-43.7576408),(-22.6734924,-43.7568684),(-22.6718297,-43.7552557),(-22.6716498,-43.7550173),(-22.6713131,-43.7545718),(-22.6712392,-43.7544739),(-22.6705401,-43.7536084),(-22.6702236,-43.7526979),(-22.6698683,-43.7512135),(-22.6695325,-43.7504311),(-22.6679962,-43.7488784),(-22.6669041,-43.7475006),(-22.6657811,-43.7460227),(-22.6648893,-43.7451567),(-22.6640676,-43.7445946),(-22.6636753,-43.7442207),(-22.662794,-43.7428335),(-22.6614344,-43.7409289),(-22.6602061,-43.7397275),(-22.6588917,-43.7388322),(-22.6585787,-43.738261),(-22.6581447,-43.7379626),(-22.6575739,-43.7378495),(-22.656788,-43.7372551),(-22.6563055,-43.7366981),(-22.6550925,-43.7352104),(-22.6545199,-43.7342954),(-22.6510698,-43.7295999),(-22.6492699,-43.7269934),(-22.6474815,-43.7246818),(-22.6459864,-43.72319),(-22.6447349,-43.7222777),(-22.6437107,-43.7211901),(-22.6433653,-43.7202294),(-22.6429822,-43.7194978),(-22.6425428,-43.7176151),(-22.6416831,-43.7145367),(-22.641249,-43.7131968),(-22.6468875,-43.6996654),(-22.647541,-43.6996654),(-22.6484004,-43.6991504),(-22.6487228,-43.6982767),(-22.6491354,-43.6937118),(-22.6495417,-43.6923799),(-22.649859,-43.6914762),(-22.6502443,-43.6901524),(-22.6503252,-43.6897024),(-22.6503804,-43.6892877),(-22.6503703,-43.6888526),(-22.6503591,-43.6882701),(-22.6501608,-43.687907),(-22.6495373,-43.6874333),(-22.6493936,-43.6870691),(-22.6493734,-43.6867644),(-22.6495476,-43.6864389),(-22.6498296,-43.685767),(-22.6499604,-43.6853949),(-22.6499968,-43.684584),(-22.6498746,-43.6839675),(-22.6496874,-43.6833656),(-22.6492274,-43.6824826),(-22.6489184,-43.6816668),(-22.6484243,-43.6810197),(-22.6481028,-43.6806227),(-22.6477438,-43.6799783),(-22.6473499,-43.6793915),(-22.6473342,-43.6786789),(-22.6474588,-43.6780039),(-22.6476524,-43.6774086),(-22.6478395,-43.676814),(-22.6479932,-43.6765107),(-22.6484137,-43.6762119),(-22.6491952,-43.6757748),(-22.6500149,-43.6749387),(-22.6506213,-43.6742941),(-22.6516623,-43.6730329),(-22.6522475,-43.672185),(-22.6524658,-43.6714737),(-22.6526403,-43.6711863),(-22.6528033,-43.6709302),(-22.652912,-43.6707825),(-22.6533641,-43.6703553),(-22.6536566,-43.6700974),(-22.6540174,-43.6698401),(-22.6544523,-43.66972),(-22.6547725,-43.6696113),(-22.6551946,-43.6693489),(-22.6554085,-43.6690595),(-22.6555657,-43.6687198),(-22.6555783,-43.6683928),(-22.6554888,-43.6677443),(-22.6554094,-43.6672864),(-22.6554022,-43.6669649),(-22.6554462,-43.6666881),(-22.6556848,-43.6662253),(-22.6560572,-43.665545),(-22.6563015,-43.6650987),(-22.6567008,-43.6646581),(-22.6571193,-43.6641772),(-22.6572308,-43.663949),(-22.657224,-43.6636668),(-22.6570485,-43.6632952),(-22.6570854,-43.6626502),(-22.6572446,-43.6623175),(-22.6575277,-43.6622264),(-22.6578281,-43.6622345),(-22.6582153,-43.662111),(-22.6587439,-43.6618394),(-22.6593336,-43.6613603),(-22.6597917,-43.6608306),(-22.6599963,-43.6603318),(-22.6600245,-43.6600574),(-22.65995,-43.6595364),(-22.6595988,-43.6584594),(-22.6593408,-43.657526),(-22.6589005,-43.6562696),(-22.6584234,-43.6551505),(-22.6578167,-43.6542757),(-22.65721,-43.6535843),(-22.6565522,-43.6532338),(-22.6559163,-43.6529445),(-22.6550399,-43.6523305),(-22.6540448,-43.6514828),(-22.6534425,-43.6506634),(-22.6532292,-43.6499357),(-22.6531436,-43.6494329),(-22.6527511,-43.6480389),(-22.6521778,-43.6466642),(-22.6520089,-43.6455642),(-22.6518331,-43.6448143),(-22.6515961,-43.6441342),(-22.6511472,-43.6433945),(-22.6508932,-43.6426039),(-22.6505921,-43.6416074),(-22.6505872,-43.6410319),(-22.6508437,-43.6405802),(-22.6512371,-43.6401551),(-22.6518118,-43.639623),(-22.6521125,-43.6393529),(-22.6527455,-43.6387841),(-22.6535718,-43.6379828),(-22.654417,-43.6371251),(-22.654951,-43.6366969),(-22.6558638,-43.6361263),(-22.6574268,-43.6350079),(-22.6580736,-43.6345312),(-22.6592048,-43.6338525),(-22.6601976,-43.6335128),(-22.6618915,-43.6332292),(-22.662705,-43.6332212),(-22.6634545,-43.6333401),(-22.6644963,-43.6337092),(-22.6649536,-43.6339153),(-22.6652788,-43.6341761),(-22.6655762,-43.6344146),(-22.6663877,-43.6351803),(-22.6667347,-43.6354225),(-22.6670936,-43.6355243),(-22.6674511,-43.6354786),(-22.6679892,-43.6352346),(-22.6683342,-43.6352453),(-22.6686348,-43.6353827),(-22.6689548,-43.6355725),(-22.6694632,-43.6358741),(-22.6699418,-43.6362836),(-22.6703108,-43.6368135),(-22.6706034,-43.6375547),(-22.6708885,-43.6386441),(-22.6710569,-43.6393164),(-22.6711238,-43.6402986),(-22.671152,-43.6413233),(-22.6711744,-43.6424252),(-22.6711466,-43.6429871),(-22.6710252,-43.6440865),(-22.6709863,-43.6448661),(-22.6710248,-43.6455748),(-22.6711139,-43.6461074),(-22.6713504,-43.6463649),(-22.6719176,-43.6467258),(-22.6722331,-43.6470358),(-22.672817,-43.6476367),(-22.6732024,-43.6478084),(-22.6735734,-43.6478118),(-22.6744782,-43.647824),(-22.6754822,-43.6482781),(-22.6758605,-43.6485755),(-22.6765755,-43.6490327),(-22.6770764,-43.64924),(-22.6779785,-43.6497016),(-22.6788676,-43.650701),(-22.6793091,-43.651488),(-22.6798849,-43.6521213),(-22.6808676,-43.652415),(-22.6820347,-43.6519474),(-22.6826784,-43.6517504),(-22.6836056,-43.6513664),(-22.6841739,-43.6510688),(-22.6845433,-43.6508556),(-22.6850391,-43.6505536),(-22.6853363,-43.6503556),(-22.6858787,-43.6501835),(-22.6861854,-43.6498793),(-22.6866521,-43.6496275),(-22.6871554,-43.6494844),(-22.687831,-43.6489736),(-22.6883876,-43.6484746),(-22.6891937,-43.6480139),(-22.6898271,-43.6481291),(-22.6904221,-43.6482251),(-22.6908732,-43.6480811),(-22.6912333,-43.6477883),(-22.6915385,-43.647281),(-22.6923223,-43.6467472),(-22.693193,-43.646199),(-22.6937042,-43.6458681),(-22.6943605,-43.6454377),(-22.6948193,-43.6453439),(-22.6953769,-43.6453093),(-22.6961993,-43.645634),(-22.6969287,-43.6456915),(-22.6974853,-43.6451733),(-22.6982241,-43.6450581),(-22.6988797,-43.6452903),(-22.6992632,-43.6449627),(-22.6997423,-43.6448149),(-22.7002455,-43.644703),(-22.7007371,-43.6447581),(-22.7016758,-43.6445209),(-22.7028211,-43.6440793),(-22.7038383,-43.6439065),(-22.705051,-43.6436792),(-22.7057938,-43.6435122),(-22.7062822,-43.6431997),(-22.7073751,-43.642518),(-22.7078092,-43.6421012),(-22.7083497,-43.6417289),(-22.7087766,-43.641362),(-22.7091786,-43.6407378),(-22.7095161,-43.6403538),(-22.7101338,-43.6398375),(-22.7107672,-43.6393961),(-22.7116501,-43.6391466),(-22.7123194,-43.6390884),(-22.7138319,-43.6383116),(-22.7150665,-43.6375151),(-22.7161413,-43.6369393),(-22.7168707,-43.6368242),(-22.7174273,-43.6368242),(-22.7178328,-43.6371314),(-22.7186749,-43.6378222),(-22.7197305,-43.6383213),
]

def _nearest_path_idx(lat: float, lon: float) -> int:
    """Geeft de index van het dichtstbijzijnde punt op RIVER_PATH."""
    return min(range(len(RIVER_PATH)),
               key=lambda i: (RIVER_PATH[i][0]-lat)**2 + (RIVER_PATH[i][1]-lon)**2)

# Pre-reken de padindex voor elke boei (werkt ook voor noord-zuidsegmenten)
_BUOY_PATH_IDX = [_nearest_path_idx(b["lat"], b["lon"]) for b in BUOYS]

def get_river_segment(buoy_i: int, buoy_j: int):
    """Geeft de rivierpunten tussen buoy i en buoy j via padindexen."""
    a, b = _BUOY_PATH_IDX[buoy_i], _BUOY_PATH_IDX[buoy_j]
    if a <= b:
        return RIVER_PATH[a:b+1]
    else:
        return RIVER_PATH[b:a+1][::-1]


THR = {
    "algae":   {"warn": 40,  "alert": 70},
    "geosmin": {"warn": 20,  "alert": 50},
    "ph":      {"low": 6.5,  "high": 8.5},
    "oxygen":  {"warn": 5.0, "alert": 3.0},
}


def seasonal_solar(doy):
    return 18.0 + 6.0 * np.sin(2 * np.pi * (doy - 355) / 365)


# ── Live temperatuurdata via Open-Meteo API ────────────────────────────────────
@st.cache_data(ttl=3600)   # ververs elk uur
def fetch_temperature_data():
    """
    Haalt dagelijkse luchttemperatuur op voor Guandu-coördinaten via Open-Meteo.
    Watertemperatuur ≈ luchttemperatuur − 1.5°C (rivier thermische buffer).
    Geeft dict: date → watertemperatuur (historisch + 16-daagse forecast).
    """
    lat, lon = -22.858, -43.686
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min"
        f"&timezone=America%2FSao_Paulo"
        f"&past_days=60&forecast_days=16"
    )
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()["daily"]
        temp_map = {}
        for d, tmax, tmin in zip(data["time"], data["temperature_2m_max"], data["temperature_2m_min"]):
            air_temp = (tmax + tmin) / 2
            water_temp = air_temp - 1.5   # rivier koeler dan lucht
            temp_map[d] = round(water_temp, 1)
        return temp_map
    except Exception:
        return {}   # bij fout: leeg → fallback naar seizoenswaarden


def get_water_temp(date, base_temp, temp_map):
    """Geeft watertemperatuur: live data indien beschikbaar, anders seizoensmodel."""
    key = date.strftime("%Y-%m-%d")
    if key in temp_map:
        return temp_map[key]
    doy = date.timetuple().tm_yday
    return base_temp + 2.5 * np.sin(2 * np.pi * (doy - 355) / 365)


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

        temp_map_hist = fetch_temperature_data()
        for i, d in enumerate(dates):
            doy  = d.timetuple().tm_yday
            temp = get_water_temp(d, base_temp, temp_map_hist) + np.random.normal(0, 0.4)
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


# ── Wetenschappelijk fysisch voorspellingsmodel (met parameterresponse) ────────
def predict_xgb(df_b, models_dict, buoy_id, days=21,
                dt=0.0, rf=0.0, disch=0.0, treatment=0.0):
    """
    Wetenschappelijk logistisch groeimodel met cardinaal temperatuurmodel.
    Parameters modificeren de natuurlijke algengroeidynamiek:
      dt    = temperatuurstijging (°C, gem. over periode)
      rf    = extra regenval (factor, 0=geen, 2=dubbel normaal)
      disch = lozingsintensiteit (factor, 0=geen, 3=hoge lozing)
      treatment = LG Sonic vermogen (0=uit, 1=vol)
    Gebaseerd op: Bernard & Rémond (2012), Paerl & Huisman (2008),
                  LG Sonic veldstudies (87% reductie in 3 weken).
    """
    hist = df_b.sort_values("date").reset_index(drop=True)
    N_start   = float(hist["algae"].iloc[-1])
    base_temp = float(hist["temp"].iloc[-1])

    K     = 140.0   # draagkracht μg/L (gemeten piek Guandu 135, 1998)
    r_max = 0.08    # max groeisnelheid /dag (veldwaarde Microcystis, Paerl 2008)
    T_opt = 33.0    # optimum temperatuur °C (tropische Microcystis, Brazilië)
    T_min = 15.0    # minimum temperatuur °C
    T_max = 40.0    # maximum temperatuur °C
    m_base = 0.008  # basale sterfte /dag (lyse + sedimentatie)
    N_floor = max(2.0, N_start * 0.10)  # LG Sonic doodt nooit alles (~10% residu)

    # Deterministisch seed op basis van omgevingsparameters (ZONDER treatment).
    # Zo hebben de no-treat en behandelde lijn dezelfde omgevingsruis,
    # en verandert de no-treat lijn NIET als de treatment slider beweegt.
    seed = hash((buoy_id, days, round(dt, 2), round(rf, 2), round(disch, 2))) % (2 ** 31)
    rng  = np.random.default_rng(seed)

    N        = N_start
    now      = datetime.now()
    temp_map = fetch_temperature_data()
    preds    = []

    for i in range(1, days + 1):
        future_date = now + timedelta(days=i)
        doy = future_date.timetuple().tm_yday

        # Live watertemperatuur (Open-Meteo) + gebruikers delta
        T = get_water_temp(future_date, base_temp, temp_map) + dt + rng.normal(0, 0.3)

        # Cardinaal temperatuurmodel (Bernard & Rémond 2012)
        if T <= T_min or T >= T_max:
            f_T = 0.0
        elif T <= T_opt:
            f_T = (T - T_min) / (T_opt - T_min)
        else:
            f_T = (T_max - T) / (T_max - T_opt)

        # Lichtfactor — seizoenscyclus Rio (meer licht in zomer)
        f_L = 0.65 + 0.35 * np.sin(2 * np.pi * (doy - 355) / 365)

        # Nutriëntenboost door lozing — vertraagd effect (3-5 dagen absorptie)
        # Nutriënten zijn pas beschikbaar voor algen na opname in het ecosysteem
        nutrient_ramp = min(1.0, max(0.0, (i - 1) / 4.0))  # volledig na dag 5
        f_N = 1.0 + nutrient_ramp * (disch / 3.0) * 0.5

        # Logistische groei: temperatuur × licht × nutriënten × draagkracht
        growth = r_max * f_T * f_L * f_N * (1 - N / K)

        # Seizoensgebonden sterfte (hoger bij suboptimale temperatuur)
        mortality = m_base + 0.012 * (1 - f_T)

        # Regen — first flush effect (Guandu, tropisch agrarisch stroomgebied):
        # Dag 1: nutriëntenpiek door runoff (suikerriet, citrus, riool) → meer groei
        # Dag 2-3: flush neemt af, verdunning begint te domineren
        # Dag 5+: netto verdunning volledig actief (max ~20% netto reductie bij rf=2)
        flush_decay    = np.exp(-(i - 1) / 2.0)                 # halveert elke 2 dagen
        rain_flush     = rf * 0.015 * flush_decay * (1 - N / K) # nutriëntenpiek dag 1
        rain_dilution  = rf * 0.010                              # ~1%/dag bij rf=1, max ~2%/dag bij rf=2
        net_rain       = rain_dilution - rain_flush
        # Max netto effect bij rf=2 over 21 dagen ≈ −18% (wetenschappelijk max 20%)

        # LG Sonic kill rate — opbouw over 5 dagen
        # Variatie koppelt aan algengroei zonder behandeling:
        # Bij hogere groeisnelheid (bloei) → meer dagelijkse fluctuatie in effectiviteit
        # Biologisch: snellere celdeling = meer variatie in gasblaasjesdichtheid
        net_growth = max(0.0, growth - mortality)           # groei zonder behandeling
        noise_scale = min(0.15, 0.05 + net_growth * 1.5)   # schaal met groeisnelheid
        treatment_ramp = min(1.0, i / 5.0) * treatment
        daily_eff = rng.normal(1.0, noise_scale)
        if rng.random() < 0.08:                             # 8% kans op verminderde dag
            daily_eff *= rng.uniform(0.3, 0.6)
        kill_rate = max(0.0, treatment_ramp * 0.10 * daily_eff)

        # Netto populatiedynamiek
        N_next = N * (1 + growth - mortality - net_rain - kill_rate)
        # Kleinere dagelijkse schommeling zodat parametereffecten zichtbaar blijven
        daily_noise = N * (0.03 + net_growth * 0.4)
        N = max(N_floor, min(K, N_next + rng.normal(0, daily_noise)))

        geo = max(0.0, N * 0.5 + rng.normal(0, 1.5))
        preds.append({
            "date":      future_date,
            "algae":     round(N, 1),
            "geosmin":   round(geo, 1),
            "treatment": treatment,
        })

    return pd.DataFrame(preds)


# ── Wetenschappelijke algengroei voorspelling (geen parameters) ────────────────
def predict_scientific(df_b, buoy_id, days=90):
    """
    Wetenschappelijke 90-daagse algengroei voorspelling op basis van:
    - Logistisch groeimodel (Verhulst)
    - Cardinaal temperatuurmodel (Bernard & Rémond 2012)
    - Seizoenscyclus Rio de Janeiro (belichting + temperatuur)
    - Historische variabiliteit als ruis
    Geen gebruikersparameters — puur wetenschappelijk baseline.
    """

    hist = df_b.sort_values("date").reset_index(drop=True)
    N_start   = float(hist["algae"].iloc[-1])
    base_temp = float(hist["temp"].iloc[-1])
    hist_std  = max(1.0, hist["algae"].tail(14).std())

    K     = 140.0   # draagkracht (gemeten piek Guandu 135 μg/L, 1998)
    r_max = 0.08    # max groeisnelheid cyanobacteriën (/dag) — veldwaarde Microcystis
    T_opt = 33.0    # optimum temperatuur (°C) — tropische Microcystis Brazilië
    T_min = 15.0    # minimum temperatuur (°C)
    T_max = 40.0    # maximum temperatuur (°C)
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

# ── Session state voor klikbare kaart ─────────────────────────────────────────
if "selected_buoy" not in st.session_state:
    st.session_state.selected_buoy = "B01"

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
    buoy_ids = [b["id"] for b in BUOYS]
    selected_buoy = st.selectbox(
        "Buoy", label_visibility="collapsed",
        options=buoy_ids,
        format_func=lambda x: f"{x}  —  {next(b['name'] for b in BUOYS if b['id'] == x)}",
        index=buoy_ids.index(st.session_state.selected_buoy),
    )
    st.session_state.selected_buoy = selected_buoy

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

    forecast_days = st.slider("Voorspellingshorizon (dagen)", 3, 90, 21)

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


# ── Terrain 3D — gedeelde initialisatie (gebruikt in tab1, tab3, tab6) ────────
@st.cache_data(ttl=86400, show_spinner="Hoogtedata ophalen via Open-Elevation…")
def _fetch_elevation_grid(lat_min=-22.74, lat_max=-22.62,
                          lon_min=-43.88, lon_max=-43.62, steps=30):
    lats = np.linspace(lat_min, lat_max, steps)
    lons = np.linspace(lon_min, lon_max, steps)
    locations = [{"latitude": float(la), "longitude": float(lo)}
                 for la in lats for lo in lons]
    try:
        resp = requests.post("https://api.open-elevation.com/api/v1/lookup",
                             json={"locations": locations}, timeout=30)
        elev = np.array([r["elevation"] for r in resp.json()["results"]], dtype=float)
        return lats, lons, elev.reshape(steps, steps)
    except Exception:
        LA, LO = np.meshgrid(lats, lons, indexing="ij")
        elev = (80 + 120 * np.exp(-((LA+22.70)**2+(LO+43.80)**2)/0.008)
                   + 60  * np.exp(-((LA+22.65)**2+(LO+43.70)**2)/0.006)
                   + 20  * np.random.rand(steps, steps))
        return lats, lons, elev

_t_lats_g, _t_lons_g, _t_elev_raw = _fetch_elevation_grid()

def _grid_elev(lat, lon, elev_raw=_t_elev_raw, lats=_t_lats_g, lons=_t_lons_g):
    i0 = int(np.clip(np.searchsorted(lats, lat) - 1, 0, len(lats)-2))
    j0 = int(np.clip(np.searchsorted(lons, lon) - 1, 0, len(lons)-2))
    di = float(np.clip((lat - lats[i0]) / (lats[i0+1] - lats[i0] + 1e-9), 0, 1))
    dj = float(np.clip((lon - lons[j0]) / (lons[j0+1] - lons[j0] + 1e-9), 0, 1))
    return float(elev_raw[i0,j0]*(1-di)*(1-dj) + elev_raw[i0+1,j0]*di*(1-dj)
                +elev_raw[i0,j0+1]*(1-di)*dj   + elev_raw[i0+1,j0+1]*di*dj)

# River path sampled from OSM, elevation from grid
_t_sample  = RIVER_PATH[::10]
_t_r_lats  = [float(p[0]) for p in _t_sample]
_t_r_lons  = [float(p[1]) for p in _t_sample]
_t_r_elev  = [_grid_elev(la, lo) + 3.0 for la, lo in zip(_t_r_lats, _t_r_lons)]

# Valley carving
_t_elev_g  = _t_elev_raw.copy()
for _gi in range(len(_t_lats_g)):
    for _gj in range(len(_t_lons_g)):
        _near = min(range(len(_t_r_lats)),
                    key=lambda k: (_t_r_lats[k]-_t_lats_g[_gi])**2
                                 +(_t_r_lons[k]-_t_lons_g[_gj])**2)
        _dist = ((_t_r_lats[_near]-_t_lats_g[_gi])**2
                +(_t_r_lons[_near]-_t_lons_g[_gj])**2)**0.5
        if _dist < 0.020:
            _rz = _t_r_elev[_near] - 3.0
            _w  = max(0.0, 1.0 - _dist/0.020)**1.2
            _t_elev_g[_gi, _gj] = min(_t_elev_g[_gi, _gj],
                                      _t_elev_g[_gi,_gj]*(1-_w) + _rz*_w)

# Buoys snapped to river
def _snap_buoys_to_river():
    out = []
    for b in BUOYS:
        idx = min(range(len(_t_r_lats)),
                  key=lambda i: (_t_r_lats[i]-b["lat"])**2+(_t_r_lons[i]-b["lon"])**2)
        gi  = int(np.clip(np.searchsorted(_t_lats_g, _t_r_lats[idx])-1, 0, len(_t_lats_g)-2))
        gj  = int(np.clip(np.searchsorted(_t_lons_g, _t_r_lons[idx])-1, 0, len(_t_lons_g)-2))
        z   = max(_grid_elev(_t_r_lats[idx], _t_r_lons[idx]),
                  float(_t_elev_g[gi, gj])) + 6
        out.append({"id": b["id"], "name": b["name"],
                    "lat": _t_r_lats[idx], "lon": _t_r_lons[idx], "z": z})
    return out

_t_buoys_3d = _snap_buoys_to_river()

# Depth profile (synthetic, realistisch voor Guandu)
def _make_depth(n):
    np.random.seed(42)
    t = np.linspace(0, 1, n)
    return list(np.clip(3.5+2.0*np.sin(t*np.pi)+1.2*np.sin(t*4*np.pi+0.8)
                        +0.4*np.random.randn(n), 1.0, 8.0))

_t_r_depth = _make_depth(len(_t_r_lats))
_t_r_bed   = [e-d for e, d in zip(_t_r_elev, _t_r_depth)]


def render_terrain_3d(algae_per_buoy: dict, title: str = "", height: int = 580) -> go.Figure:
    """
    Bouw het 3D terrein model.
    algae_per_buoy: dict {buoy_id: algae_value_μg/L}
    """
    # Interpoleer algenwaarden langs rivierpunten
    _indices, _vals = [], []
    for b in _t_buoys_3d:
        idx = min(range(len(_t_r_lats)),
                  key=lambda i: (_t_r_lats[i]-b["lat"])**2+(_t_r_lons[i]-b["lon"])**2)
        _indices.append(idx)
        _vals.append(float(algae_per_buoy.get(b["id"], 20.0)))
    _pairs  = sorted(zip(_indices, _vals))
    _si, _sa = [p[0] for p in _pairs], [p[1] for p in _pairs]
    _si  = [0] + _si  + [len(_t_r_lats)-1]
    _sa  = [_sa[0]] + _sa + [_sa[-1]]
    r_algae = list(np.interp(range(len(_t_r_lats)), _si, _sa))

    fig = go.Figure()

    # Terrein
    fig.add_trace(go.Surface(
        x=_t_lons_g, y=_t_lats_g, z=_t_elev_g,
        colorscale=[[0.0,"#2D6A4F"],[0.35,"#74C69D"],[0.65,"#B7E4C7"],
                    [0.85,"#D4A574"],[1.0,"#A0856B"]],
        opacity=0.85, showscale=True,
        colorbar=dict(title="Hoogte (m)", thickness=12, len=0.55, x=1.02,
                      tickfont=dict(size=9)),
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white")),
        name="Terrein",
        hovertemplate="Hoogte: %{z:.0f} m<extra></extra>",
    ))

    # Wateroppervlak gekleurd op algenconcentratie
    fig.add_trace(go.Scatter3d(
        x=_t_r_lons, y=_t_r_lats, z=_t_r_elev,
        mode="lines",
        line=dict(color=r_algae,
                  colorscale=[[0.0,"#27AE60"],[0.35,"#F1C40F"],
                               [0.65,"#E67E22"],[1.0,"#C0392B"]],
                  cmin=0, cmax=80, width=7,
                  colorbar=dict(title=dict(text="Algen (μg/L)", font=dict(color="#94A3B8",size=10)),
                                thickness=10, len=0.45, x=0.0,
                                tickfont=dict(color="#94A3B8",size=9),
                                tickvals=[0,20,40,60,80])),
        name="Wateroppervlak",
        customdata=[[r_algae[i], _t_r_depth[i]] for i in range(len(r_algae))],
        hovertemplate="Algen: %{customdata[0]:.1f} μg/L<br>Diepte: %{customdata[1]:.1f} m<extra></extra>",
    ))

    # Rivierbodem
    fig.add_trace(go.Scatter3d(
        x=_t_r_lons, y=_t_r_lats, z=_t_r_bed,
        mode="lines",
        line=dict(color="#4A4A6A", width=3, dash="dot"),
        name="Rivierbodem",
        hovertemplate="Rivierbodem: %{z:.0f} m<extra></extra>",
    ))

    # Waterzuilen
    for idx in range(0, len(_t_r_lats), 5):
        fig.add_trace(go.Scatter3d(
            x=[_t_r_lons[idx],_t_r_lons[idx]],
            y=[_t_r_lats[idx],_t_r_lats[idx]],
            z=[_t_r_bed[idx], _t_r_elev[idx]],
            mode="lines", line=dict(color="rgba(0,150,199,0.25)", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    # Buoy markers
    algae_colors_3d = []
    for b in _t_buoys_3d:
        v = algae_per_buoy.get(b["id"], 20.0)
        algae_colors_3d.append(C_GREEN if v < 30 else C_YELLOW if v < 60 else C_RED)

    fig.add_trace(go.Scatter3d(
        x=[b["lon"] for b in _t_buoys_3d],
        y=[b["lat"] for b in _t_buoys_3d],
        z=[b["z"]   for b in _t_buoys_3d],
        mode="markers+text",
        marker=dict(size=9, color=algae_colors_3d,
                    line=dict(color=C_WHITE, width=1)),
        text=[b["id"] for b in _t_buoys_3d],
        textposition="top center",
        textfont=dict(size=10, color=C_WHITE),
        name="MPC-Buoys",
        customdata=[[b["name"], round(algae_per_buoy.get(b["id"], 20.0), 1)]
                    for b in _t_buoys_3d],
        hovertemplate="<b>%{text} — %{customdata[0]}</b><br>Algen: %{customdata[1]} μg/L<extra></extra>",
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor=C_DARK,
        title=dict(text=title, font=dict(color=C_WHITE, size=13), x=0.0, xanchor="left") if title else {},
        scene=dict(
            xaxis=dict(title="Lengtegraad", backgroundcolor=C_DARK,
                       gridcolor="#1E3A5F", tickfont=dict(color="#94A3B8",size=9)),
            yaxis=dict(title="Breedtegraad", backgroundcolor=C_DARK,
                       gridcolor="#1E3A5F", tickfont=dict(color="#94A3B8",size=9)),
            zaxis=dict(title="Hoogte (m)", backgroundcolor=C_DARK,
                       gridcolor="#1E3A5F", tickfont=dict(color="#94A3B8",size=9)),
            bgcolor=C_DARK,
            camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9)),
            aspectmode="manual",
            aspectratio=dict(x=2.2, y=1.2, z=0.4),
        ),
        legend=dict(font=dict(color="#94A3B8",size=11), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
    )
    return fig


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
        st.markdown('<p class="section-label">Live algenconcentratie — 3D kaart</p>', unsafe_allow_html=True)
        _live_algae = {b["id"]: float(latest[latest["buoy_id"]==b["id"]]["algae"].values[0])
                       for b in BUOYS}
        st.plotly_chart(render_terrain_3d(_live_algae, height=620),
                        use_container_width=True)
        st.markdown(f"""
        <div style="display:flex;gap:20px;font-size:11px;color:{C_MUTED};margin-top:4px;">
          <span><span style="color:{C_GREEN};font-weight:700;">●</span> Normaal (&lt;30 μg/L)</span>
          <span><span style="color:{C_YELLOW};font-weight:700;">●</span> Verhoogd (30–60 μg/L)</span>
          <span><span style="color:{C_RED};font-weight:700;">●</span> Alarm (&gt;60 μg/L)</span>
          <span>Rivier: groen→rood = algenconcentratie · draai met muis</span>
        </div>""", unsafe_allow_html=True)

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
              <div style="font-size:11px; color:#475569; margin-bottom:5px;">{buoy_name}</div>
              <div style="display:flex; justify-content:space-between;">
                <div>
                  <div style="font-size:10px; color:#64748B;">Algen</div>
                  <div style="font-size:12px; font-weight:600; color:{C_DARK};">{row['algae']} <span style="font-size:9px; color:#64748B;">μg/L</span></div>
                </div>
                <div>
                  <div style="font-size:10px; color:#64748B;">Geosmin</div>
                  <div style="font-size:12px; font-weight:600; color:{C_DARK};">{row['geosmin']} <span style="font-size:9px; color:#64748B;">ng/L</span></div>
                </div>
                <div>
                  <div style="font-size:10px; color:#64748B;">O₂</div>
                  <div style="font-size:12px; font-weight:600; color:{C_DARK};">{row['oxygen']} <span style="font-size:9px; color:#64748B;">mg/L</span></div>
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
    st.markdown(
        f'<p class="section-label">{selected_buoy} — {buoy_name_sel} · {forecast_days}-daagse voorspelling</p>',
        unsafe_allow_html=True)

    df_b  = df[df["buoy_id"] == selected_buoy].copy()
    hist14 = df_b.tail(14)
    anchor_date = hist14["date"].iloc[-1]
    anchor_val  = hist14["algae"].iloc[-1]
    now_dt = datetime.now()

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

    # ── 3D Algenvoorspelling kaart — bovenaan ─────────────────────────────────
    _forecast_algae = {}
    for _b in BUOYS:
        _df_b_tmp = df[df["buoy_id"] == _b["id"]].copy()
        _pred_tmp = predict_xgb(_df_b_tmp, xgb_models, _b["id"],
                                forecast_days, temp_offset, rain_factor,
                                discharge, treatment)
        _forecast_algae[_b["id"]] = float(_pred_tmp["algae"].iloc[-1])

    st.markdown(f'<p class="section-label" style="margin-top:4px;text-align:left;">3D algenvoorspelling — dag {forecast_days} van de horizon</p>',
                unsafe_allow_html=True)
    st.caption(f"parameters: +{temp_offset}°C · regen ×{rain_factor:.1f} · lozing ×{discharge:.1f} · LG Sonic {int(treatment*100)}%")
    st.plotly_chart(
        render_terrain_3d(
            _forecast_algae,
            title=f"Algenvoorspelling dag {forecast_days} — scenario simulator",
            height=560,
        ),
        use_container_width=True,
    )
    st.markdown(f"""
    <div style="display:flex;gap:20px;font-size:11px;margin-top:4px;margin-bottom:20px;">
      <span><span style="color:{C_GREEN};font-weight:700;">●</span> Normaal (&lt;30 μg/L)</span>
      <span><span style="color:{C_YELLOW};font-weight:700;">●</span> Verhoogd (30–60 μg/L)</span>
      <span><span style="color:{C_RED};font-weight:700;">●</span> Alarm (&gt;60 μg/L)</span>
    </div>""", unsafe_allow_html=True)

    fig_f = go.Figure()
    # Onbehandeld (grijs) — referentie
    fig_f.add_trace(go.Scatter(
        x=dates_fwd, y=notr_vals, mode="lines",
        name="Zonder behandeling", line=dict(color="#B0BEC5", width=1.5, dash="dot"),
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
    # Voorspelling met behandeling
    fig_f.add_trace(go.Scatter(
        x=dates_fwd, y=pred_vals, mode="lines+markers",
        name=f"Voorspelling (behandeling {int(treatment*100)}%)",
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

    style(fig_f, height=380, title="")
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


if False:
    # tab6 verwijderd
    st.markdown('<p class="section-label">3D terreinvisualisatie — Guandu rivier</p>', unsafe_allow_html=True)

    _live_algae_t6 = {b["id"]: float(latest[latest["buoy_id"]==b["id"]]["algae"].values[0])
                      for b in BUOYS}
    st.plotly_chart(render_terrain_3d(_live_algae_t6,
                                      title="3D Terrein — Guandu rivier & MPC-Buoys",
                                      height=640),
                    use_container_width=True)

    st.markdown(f"""
    <div style="display:flex;gap:24px;justify-content:center;margin-top:8px;
                font-size:12px;color:{C_MUTED};">
      <span><span style="color:{C_GREEN};font-weight:700;">●</span> Normaal (&lt;30 μg/L)</span>
      <span><span style="color:{C_YELLOW};font-weight:700;">●</span> Verhoogd (30–60 μg/L)</span>
      <span><span style="color:{C_RED};font-weight:700;">●</span> Alarm (&gt;60 μg/L)</span>
      <span><span style="color:{C_GREEN};font-weight:700;">—</span>/<span style="color:{C_RED};font-weight:700;">—</span> Rivier kleur = algen</span>
      <span><span style="color:#4A4A6A;font-weight:700;">- -</span> Rivierbodem</span>
    </div>""", unsafe_allow_html=True)

    # ── Diepteprofiel chart ──────────────────────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:20px;">Diepteprofiel langs de rivier</p>',
                unsafe_allow_html=True)

    from math import radians, cos, sin, asin, sqrt as _sqrt
    def _haversine_m(la1, lo1, la2, lo2):
        R = 6371000
        a = sin(radians(la2-la1)/2)**2 + cos(radians(la1))*cos(radians(la2))*sin(radians(lo2-lo1)/2)**2
        return 2*R*asin(_sqrt(a))

    _dist_km = [0.0]
    for _i in range(1, len(_t_r_lats)):
        _dist_km.append(_dist_km[-1] + _haversine_m(_t_r_lats[_i-1], _t_r_lons[_i-1],
                                                     _t_r_lats[_i],   _t_r_lons[_i]) / 1000)

    _buoy_dist_t6, _buoy_depth_mid = [], []
    for _b in _t_buoys_3d:
        _idx = min(range(len(_t_r_lats)),
                   key=lambda i: (_t_r_lats[i]-_b["lat"])**2+(_t_r_lons[i]-_b["lon"])**2)
        _buoy_dist_t6.append(_dist_km[_idx])
        _buoy_depth_mid.append(-_t_r_depth[_idx] / 2)

    _algae_color_t6 = [C_GREEN if _live_algae_t6.get(_b["id"],20)<30
                       else C_YELLOW if _live_algae_t6.get(_b["id"],20)<60
                       else C_RED for _b in _t_buoys_3d]

    fig_depth = go.Figure()
    fig_depth.add_trace(go.Scatter(x=_dist_km, y=[0.0]*len(_dist_km), mode="lines",
                                   line=dict(color=C_BLUE, width=2), name="Wateroppervlak"))
    fig_depth.add_trace(go.Scatter(x=_dist_km, y=[-d for d in _t_r_depth], mode="lines",
                                   line=dict(color="#023E8A", width=1.5),
                                   fill="tonexty", fillcolor="rgba(0,150,199,0.35)",
                                   name="Rivierbodem"))
    fig_depth.add_trace(go.Scatter(x=_buoy_dist_t6, y=_buoy_depth_mid,
                                   mode="markers+text",
                                   marker=dict(size=10, color=_algae_color_t6,
                                               line=dict(color=C_WHITE, width=1)),
                                   text=[_b["id"] for _b in _t_buoys_3d],
                                   textposition="top center",
                                   textfont=dict(size=9, color=C_TEXT),
                                   customdata=[[_b["name"]] for _b in _t_buoys_3d],
                                   hovertemplate="<b>%{text}</b> — %{customdata[0]}<br>%{x:.1f} km<extra></extra>",
                                   name="Buoys"))
    fig_depth.update_layout(**CHART, height=200, margin=dict(l=12,r=12,t=20,b=40),
                            xaxis=dict(title="Afstand (km)", showgrid=False, linecolor=C_BORDER),
                            yaxis=dict(title="Diepte (m)", tickvals=[0,-2,-4,-6,-8],
                                       ticktext=["0","2","4","6","8"],
                                       gridcolor="#EEF2F6", zeroline=True,
                                       zerolinecolor=C_BLUE, zerolinewidth=1.5),
                            showlegend=False)
    st.plotly_chart(fig_depth, use_container_width=True)
    st.caption("Dieptedata: synthetisch model (1–8 m) · echte bathymetrische data koppelen zodra beschikbaar.")
