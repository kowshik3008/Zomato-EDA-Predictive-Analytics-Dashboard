import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Step 1: Load & Clean Data
# -------------------------------
df = pd.read_csv("Zomato data .csv")

# Clean 'rate'
df['rate'] = df['rate'].str.replace('/5', '').str.strip()
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

# Clean 'approx_cost(for two people)'
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '')
df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')

# Drop duplicates & dropna mapping
df = df.drop_duplicates().dropna(subset=['rate', 'approx_cost(for two people)', 'votes'])

# Encode Yes/No for general Use
df['online_order_encoded'] = df['online_order'].map({'Yes': 1, 'No': 0})
df['book_table_encoded'] = df['book_table'].map({'Yes': 1, 'No': 0})

# -------------------------------
# Step 2: Machine Learning Setup
# -------------------------------
# A. Clustering
features_for_clustering = ['rate', 'votes', 'approx_cost(for two people)']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features_for_clustering])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)
cluster_mapping = {0: 'Budget-Friendly', 1: 'High-Volume/Popular', 2: 'High-End Dining'} 
df['Cluster'] = df['Cluster'].map(cluster_mapping)

# B. Predictive Model (Random Forest)
features_for_prediction = ['approx_cost(for two people)', 'votes', 'online_order_encoded', 'book_table_encoded']
X = df[features_for_prediction]
y = df['rate']
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# -------------------------------
# Step 3: Initialize Dash App with Bootstrap Theme
# -------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# -------------------------------
# Step 4: Layout Definition
# -------------------------------

sidebar = html.Div(
    [
        html.H2("Dashboard Controls", className="display-6"),
        html.Hr(),
        html.P("Filter the dashboard data below:", className="lead"),
        
        html.Label("Restaurant Type"),
        dcc.Dropdown(
            id='type-dropdown',
            options=[{'label': t, 'value': t} for t in df['listed_in(type)'].unique()],
            value=list(df['listed_in(type)'].unique()),
            multi=True,
            style={'color': '#000'}
        ),
        html.Br(),
        
        html.Label("Online Order availability"),
        dbc.RadioItems(
            options=[
                {"label": "All", "value": "All"},
                {"label": "Yes", "value": "Yes"},
                {"label": "No", "value": "No"},
            ],
            value="All",
            id="online-radio",
            inline=True,
        ),
        html.Br(),
        
        html.Label("Table Booking availability"),
        dbc.RadioItems(
            options=[
                {"label": "All", "value": "All"},
                {"label": "Yes", "value": "Yes"},
                {"label": "No", "value": "No"},
            ],
            value="All",
            id="table-radio",
            inline=True,
        ),
        
        html.Hr(),
        html.H4("Machine Learning: Rating Predictor", className="mt-4"),
        html.P("Predict rating for a new restaurant:"),
        
        html.Label("Est. Cost for Two:"),
        dcc.Slider(
            id='sim-cost', min=100, max=3000, step=100, value=800,
            marks={100: '100', 1000: '1k', 2000:'2k', 3000:'3k'},
            tooltip={"placement": "bottom", "always_visible": False}
        ),
        html.Br(),
        
        html.Label("Est. Votes:"),
        dcc.Slider(
            id='sim-votes', min=0, max=5000, step=100, value=500,
            marks={0: '0', 2500: '2.5k', 5000: '5k'},
            tooltip={"placement": "bottom", "always_visible": False}
        ),
        html.Br(),
        
        html.Label("Online Order?"),
        dbc.RadioItems(
            options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
            value=1, id="sim-online", inline=True,
        ),
        html.Br(),
        
        html.Label("Book Table?"),
        dbc.RadioItems(
            options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
            value=0, id="sim-table", inline=True,
        ),
        
        html.Div(id="prediction-output", className="mt-3 p-3 bg-primary text-white text-center rounded fw-bold fs-4")

    ],
    className="p-3 bg-dark text-white h-100"
)

content = html.Div(
    [
        html.H1("Zomato EDA & Predictive Analytics Portfolio", className="text-center mt-3 mb-4 fw-bold"),
        
        # KPI Cards row
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Total Restaurants", className="card-title"), html.H2(id="kpi-total")])], color="info", outline=True)),
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Average Cost", className="card-title"), html.H2(id="kpi-cost")])], color="success", outline=True)),
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Average Rating", className="card-title"), html.H2(id="kpi-rating")])], color="warning", outline=True)),
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Total Votes", className="card-title"), html.H2(id="kpi-votes")])], color="danger", outline=True)),
        ], className="mb-4"),
        
        # Plotly Charts
        dbc.Row([
            dbc.Col(dcc.Graph(id='rating-dist'), md=6),
            dbc.Col(dcc.Graph(id='cluster-scatter'), md=6),
        ]),
        
        dbc.Row([
            dbc.Col(dcc.Graph(id='pie-charts'), md=6),
            dbc.Col(dcc.Graph(id='cost-boxplot'), md=6),
        ])
    ],
    className="p-4"
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(sidebar, md=3, className="p-0 border-end border-secondary shadow-lg"),
        dbc.Col(content, md=9)
    ], className="g-0")
], fluid=True)

# -------------------------------
# Step 5: Callbacks
# -------------------------------
@app.callback(
    [Output('kpi-total', 'children'),
     Output('kpi-cost', 'children'),
     Output('kpi-rating', 'children'),
     Output('kpi-votes', 'children'),
     Output('rating-dist', 'figure'),
     Output('cluster-scatter', 'figure'),
     Output('pie-charts', 'figure'),
     Output('cost-boxplot', 'figure')],
    [Input('type-dropdown', 'value'),
     Input('online-radio', 'value'),
     Input('table-radio', 'value')]
)
def update_dashboard(selected_types, online_val, table_val):
    dff = df.copy()
    
    # Safely handle empty selection in dropdown
    if not selected_types:
        dff = pd.DataFrame(columns=df.columns)
    else:
        dff = dff[dff['listed_in(type)'].isin(selected_types)]
        
    if online_val != "All":
        dff = dff[dff['online_order'] == online_val]
    if table_val != "All":
        dff = dff[dff['book_table'] == table_val]
        
    # KPIs
    kpi1 = len(dff)
    kpi2 = f"₹{dff['approx_cost(for two people)'].mean():.0f}" if len(dff) > 0 else "0"
    kpi3 = f"{dff['rate'].mean():.1f} ⭐" if len(dff) > 0 else "0"
    kpi4 = f"{dff['votes'].sum():,}"
    
    # If dataframe is empty, return empty figures
    if len(dff) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark", title="No Data Available for Selection")
        return kpi1, kpi2, kpi3, kpi4, empty_fig, empty_fig, empty_fig, empty_fig

    # 1. Top 10 Restaurants by Rating (Bar Chart)
    top10 = dff.sort_values(by='rate', ascending=False).head(10)
    fig1 = px.bar(top10, x='rate', y='name', orientation='h', 
                  title="Top 10 Rated Restaurants", color='rate',
                  hover_data=['votes', 'approx_cost(for two people)', 'listed_in(type)'],
                  color_continuous_scale="Viridis", template="plotly_dark")
    fig1.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=40, b=0))

    # 2. 3D Cluster Scatter Plot
    fig2 = px.scatter_3d(dff, x='votes', y='approx_cost(for two people)', z='rate', 
                         color='Cluster', title="Restaurant Segments (K-Means)", 
                         hover_name='name', hover_data=['listed_in(type)', 'online_order', 'book_table'],
                         template="plotly_dark", size_max=10, opacity=0.8)
    fig2.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    # 3. Pie Charts for Online Order & Table Booking
    from plotly.subplots import make_subplots
    fig3 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                         subplot_titles=['Online Order', 'Table Booking'])
    
    online_counts = dff['online_order'].value_counts()
    book_counts = dff['book_table'].value_counts()
    
    if not online_counts.empty:
        fig3.add_trace(go.Pie(labels=online_counts.index, values=online_counts.values, marker=dict(colors=['#00cc96', '#ef553b'])), 1, 1)
    if not book_counts.empty:
        fig3.add_trace(go.Pie(labels=book_counts.index, values=book_counts.values, marker=dict(colors=['#ab63fa', '#ffa15a'])), 1, 2)
        
    fig3.update_layout(title_text="Services Distribution", template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0))

    # 4. Boxplot
    fig4 = px.box(dff, x='listed_in(type)', y='approx_cost(for two people)', color='listed_in(type)',
                  hover_data=['name', 'rate', 'votes'],
                  title="Cost Distribution by Type", template="plotly_dark")
    fig4.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    return kpi1, kpi2, kpi3, kpi4, fig1, fig2, fig3, fig4

@app.callback(
    Output('prediction-output', 'children'),
    [Input('sim-cost', 'value'),
     Input('sim-votes', 'value'),
     Input('sim-online', 'value'),
     Input('sim-table', 'value')]
)
def predict_rating(cost, votes, online, table):
    # Predict rating using the trained Random Forest model
    input_data = pd.DataFrame([[cost, votes, online, table]], 
                              columns=['approx_cost(for two people)', 'votes', 'online_order_encoded', 'book_table_encoded'])
    
    predicted_rating = rf_model.predict(input_data)[0]
    return f"Predicted Rating: {predicted_rating:.1f} ⭐ / 5.0"

# -------------------------------
# Step 6: Run App
# -------------------------------
if __name__ == '__main__':
    # Launch on a different port (8051) to avoid conflicts if the previous process is still bound to 8050
    app.run(debug=True, port=8051)