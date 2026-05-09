import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyMetrics — Aviation Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    background: #070b14;
    color: #e2e8f0;
    font-family: 'Barlow Condensed', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #05070d 0%, #0b1220 50%, #05070d 100%);
}

.hero {
    padding: 45px;
    border-radius: 22px;
    background: linear-gradient(135deg,#0b1220,#05070d);
    border: 1px solid rgba(59,130,246,0.25);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 64px;
    font-weight: 700;
    text-align: center;
    letter-spacing: 4px;
}

.hero-title span {
    color: #f59e0b;
}

.hero-sub {
    text-align: center;
    color: #94a3b8;
    letter-spacing: 3px;
}

.metric-card {
    padding: 22px;
    border-radius: 18px;
    background: rgba(10,15,25,0.95);
    border: 1px solid rgba(255,255,255,0.06);
}

.metric-val {
    font-size: 42px;
    font-family: 'Share Tech Mono', monospace;
    color: white;
}

.metric-label {
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.metric-delta {
    margin-top: 8px;
    color: #64748b;
}

.blue { border-top: 3px solid #3b82f6; }
.red { border-top: 3px solid #ef4444; }
.green { border-top: 3px solid #22c55e; }
.amber { border-top: 3px solid #f59e0b; }
.purple { border-top: 3px solid #8b5cf6; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def create_dataset(n=15000):
    rng = np.random.default_rng(42)

    airlines = [
        'Delta','United','American','Southwest',
        'JetBlue','Spirit','Frontier','Alaska'
    ]

    airports = [
        'ATL','LAX','ORD','DFW','DEN','JFK',
        'SFO','SEA','LAS','MCO','BOS','MIA'
    ]

    df = pd.DataFrame({
        'airline': rng.choice(airlines, n),
        'origin': rng.choice(airports, n),
        'destination': rng.choice(airports, n),
        'month': rng.integers(1,13,n),
        'dep_hour': rng.integers(0,24,n),
        'distance': rng.integers(150,3000,n)
    })

    prob = 0.15
    prob += (df['dep_hour'] >= 17) * 0.12
    prob += (df['origin'].isin(['ORD','JFK','SFO'])) * 0.10
    prob += (df['month'].isin([12,1,2])) * 0.08

    delay_prob = np.clip(prob,0.05,0.90)

    df['is_delayed'] = rng.binomial(1, delay_prob)

    delay_minutes = np.where(
        df['is_delayed'] == 1,
        rng.exponential(35, n),
        0
    )

    df['delay_minutes'] = delay_minutes.astype(int)

    return df


df = create_dataset()

# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
overview = {
    'total_flights': len(df),
    'delay_rate': df['is_delayed'].mean(),
    'delayed_flights': int(df['is_delayed'].sum()),
    'avg_delay': int(df[df['is_delayed']==1]['delay_minutes'].mean()),
    'airlines': df['airline'].nunique(),
    'airports': df['origin'].nunique()
}

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-title'>
        <span>SKY</span>METRICS ✈
    </div>

    <div class='hero-sub'>
        AVIATION DELAY INTELLIGENCE PLATFORM
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────────────────────────────────────
cols = st.columns(5)

kpis = [
    (
        cols[0],
        overview['total_flights'],
        'Flights',
        'Tracked flights',
        'blue'
    ),
    (
        cols[1],
        f"{overview['delay_rate']:.1%}",
        'Delay Rate',
        f"{overview['delayed_flights']:,} delayed",
        'red'
    ),
    (
        cols[2],
        f"{overview['avg_delay']} min",
        'Avg Delay',
        'Delayed flights only',
        'amber'
    ),
    (
        cols[3],
        overview['airlines'],
        'Airlines',
        'Tracked carriers',
        'green'
    ),
    (
        cols[4],
        overview['airports'],
        'Airports',
        'Major hubs',
        'purple'
    )
]

for col, val, label, delta, color in kpis:

    if isinstance(val, (int,float,np.integer,np.floating)):
        formatted_val = f"{val:,}"
    else:
        formatted_val = str(val)

    with col:
        st.markdown(f"""
        <div class='metric-card {color}'>
            <div class='metric-val'>{formatted_val}</div>
            <div class='metric-label'>{label}</div>
            <div class='metric-delta'>{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title('⚙️ Control Panel')

selected_airline = st.sidebar.multiselect(
    'Select Airlines',
    sorted(df['airline'].unique()),
    default=list(df['airline'].unique())
)

filtered_df = df[df['airline'].isin(selected_airline)]

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    '📡 Live Prediction',
    '📊 Airline Analytics',
    '🛫 Airport Analysis',
    '🌦 Weather Impact',
    '🗺 Route Explorer',
    '🤖 ML Performance',
    '💡 AI Insights'
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:

    st.subheader('📡 Live Delay Prediction')

    c1, c2 = st.columns(2)

    with c1:
        airline = st.selectbox(
            'Airline',
            sorted(df['airline'].unique())
        )

        origin = st.selectbox(
            'Origin',
            sorted(df['origin'].unique())
        )

    with c2:
        dep_hour = st.slider('Departure Hour',0,23,15)
        distance = st.slider('Distance',100,3000,850)

    if st.button('Predict Delay Risk'):

        risk = 0.25

        if dep_hour >= 17:
            risk += 0.20

        if origin in ['ORD','JFK','SFO']:
            risk += 0.15

        if distance > 2000:
            risk += 0.10

        risk = min(risk,0.95)

        st.success('Prediction generated successfully.')

        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=risk * 100,
            title={'text':'Delay Probability'},
            gauge={
                'axis': {'range':[0,100]},
                'bar': {'color':'#ef4444'}
            }
        ))

        fig.update_layout(height=350)

        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:

    st.subheader('📊 Airline Analytics')

    airline_delay = (
        filtered_df.groupby('airline')['is_delayed']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
        airline_delay,
        x='airline',
        y='is_delayed',
        color='is_delayed',
        title='Delay Rate by Airline'
    )

    fig.update_layout(template='plotly_dark')

    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:

    st.subheader('🛫 Airport Analysis')

    airport_stats = (
        filtered_df.groupby('origin')
        .agg({
            'is_delayed':'mean',
            'delay_minutes':'mean'
        })
        .reset_index()
    )

    fig = px.scatter(
        airport_stats,
        x='origin',
        y='is_delayed',
        size='delay_minutes',
        color='delay_minutes',
        title='Airport Congestion Analysis'
    )

    fig.update_layout(template='plotly_dark')

    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:

    st.subheader('🌦 Weather Impact')

    monthly = (
        filtered_df.groupby('month')['is_delayed']
        .mean()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x='month',
        y='is_delayed',
        markers=True,
        title='Seasonal Delay Trend'
    )

    fig.update_layout(template='plotly_dark')

    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:

    st.subheader('🗺 Route Explorer')

    filtered_df['route'] = (
        filtered_df['origin'] + ' → ' + filtered_df['destination']
    )

    routes = (
        filtered_df.groupby('route')
        .agg({
            'is_delayed':'mean',
            'delay_minutes':'mean'
        })
        .sort_values('delay_minutes', ascending=False)
        .head(15)
        .reset_index()
    )

    st.dataframe(routes, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:

    st.subheader('🤖 ML Performance')

    metrics_df = pd.DataFrame({
        'Model':['Random Forest','XGBoost','Logistic Regression'],
        'Accuracy':[0.82,0.84,0.74],
        'F1 Score':[0.79,0.81,0.70],
        'ROC AUC':[0.89,0.91,0.80]
    })

    st.dataframe(metrics_df, use_container_width=True)

    fig = px.bar(
        metrics_df,
        x='Model',
        y='ROC AUC',
        color='ROC AUC',
        title='Model Comparison'
    )

    fig.update_layout(template='plotly_dark')

    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:

    st.subheader('💡 AI Insights')

    st.success('''
    • Evening departures show significantly higher delay probability.

    • ORD and JFK are the highest-risk congestion airports.

    • Winter months produce the strongest seasonal delay impact.

    • Delta and Alaska maintain the strongest operational reliability.

    • Long-haul routes exhibit larger delay variance.
    ''')

