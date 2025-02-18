#app.py
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv('Fraud_Data_Merged.csv')

# Ensure the fraud label column is correctly named and is integer type
fraud_column = 'class'  # Replace with the correct column name if different
df[fraud_column] = df[fraud_column].astype(int)

# Convert purchase_time to datetime (if not already)
if 'purchase_time' in df.columns:
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['purchase_date'] = df['purchase_time'].dt.date  # Extract date for trend analysis

# Initialize the Dash app
app = Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),

    # Summary Boxes
    html.Div([
        html.Div([
            html.H3("Total Transactions"),
            html.P(len(df))
        ], className="summary-box"),
        html.Div([
            html.H3("Total Fraud Cases"),
            html.P(df[fraud_column].sum())
        ], className="summary-box"),
        html.Div([
            html.H3("Fraud Percentage"),
            html.P(f"{df[fraud_column].mean() * 100:.2f}%")
        ], className="summary-box")
    ], className="summary-row"),

    # Line Chart: Fraud Cases Over Time
    dcc.Graph(id='fraud-trend'),

    # Geographical Map: Fraud by Country
    dcc.Graph(id='fraud-map'),

    # Bar Charts: Fraud by Device, Source, and Browser
    html.Div([
        dcc.Graph(id='fraud-by-device'),
        dcc.Graph(id='fraud-by-source'),
        dcc.Graph(id='fraud-by-browser')
    ], className="bar-charts")
])

# Callback for Fraud Trend Line Chart
@app.callback(
    Output('fraud-trend', 'figure'),
    Input('fraud-trend', 'id')
)
def update_fraud_trend(_):
    if 'purchase_date' in df.columns:
        trend_data = df.groupby('purchase_date')[fraud_column].sum().reset_index()
        return px.line(trend_data, x='purchase_date', y=fraud_column, title="Fraud Cases Over Time")
    else:
        return px.line(title="No Date Data Available")

# Callback for Fraud Map
@app.callback(
    Output('fraud-map', 'figure'),
    Input('fraud-map', 'id')
)
def update_fraud_map(_):
    if 'country' in df.columns:
        map_data = df.groupby('country')[fraud_column].sum().reset_index()
        return px.choropleth(map_data, locations='country', locationmode='country names',
                             color=fraud_column, title="Fraud Cases by Country")
    else:
        return px.choropleth(title="No Country Data Available")

# Callback for Fraud by Device
@app.callback(
    Output('fraud-by-device', 'figure'),
    Input('fraud-by-device', 'id')
)
def update_fraud_by_device(_):
    if 'device_id' in df.columns:
        device_data = df.groupby('device_id')[fraud_column].sum().reset_index()
        return px.bar(device_data, x='device_id', y=fraud_column, title="Fraud Cases by Device")
    else:
        return px.bar(title="No Device Data Available")

# Callback for Fraud by Source
@app.callback(
    Output('fraud-by-source', 'figure'),
    Input('fraud-by-source', 'id')
)
def update_fraud_by_source(_):
    if 'source' in df.columns:
        source_data = df.groupby('source')[fraud_column].sum().reset_index()
        return px.bar(source_data, x='source', y=fraud_column, title="Fraud Cases by Source")
    else:
        return px.bar(title="No Source Data Available")

# Callback for Fraud by Browser
@app.callback(
    Output('fraud-by-browser', 'figure'),
    Input('fraud-by-browser', 'id')
)
def update_fraud_by_browser(_):
    if 'browser' in df.columns:
        browser_data = df.groupby('browser')[fraud_column].sum().reset_index()
        return px.bar(browser_data, x='browser', y=fraud_column, title="Fraud Cases by Browser")
    else:
        return px.bar(title="No Browser Data Available")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)