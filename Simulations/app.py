import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import stock  # Import our stock.py module

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # For potential deployment

app.layout = html.Div([
    html.H1("Stock Forecast Simulator"),
    html.Div([
        html.Label("Stock Ticker:"),
        dcc.Input(id='ticker-input', type='text', value='AAPL', style={'marginRight': '20px'}),
        
        html.Label("Historical Forecast Days:"),
        dcc.Input(id='forecast-days', type='number', value=100, style={'marginRight': '20px'}),
        
        html.Label("Number of Simulations:"),
        dcc.Input(id='num-simulations', type='number', value=100, style={'marginRight': '20px'}),
        
        html.Label("Future Forecast Days:"),
        dcc.Input(id='future-forecast-days', type='number', value=30, style={'marginRight': '20px'}),
        
        html.Button("Run Simulation", id='run-button', n_clicks=0)
    ], style={'marginBottom': '20px'}),
    
    html.Div(id='error-message', style={'color': 'red', 'marginBottom': '20px'}),
    
    # Graphs for historical and historical forecast, side by side
    html.Div([
        dcc.Graph(id='history-graph', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='forecast-graph', style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ]),
    
    # Graph for future forecast below
    html.Div([
        dcc.Graph(id='future-forecast-graph')
    ], style={'marginTop': '30px'}),
    
    # A box below the historical plot that outputs mu and sigma
    html.Div(id='metrics-output', style={'marginTop': '20px', 'fontWeight': 'bold'})
])

@app.callback(
    [Output('history-graph', 'figure'),
     Output('forecast-graph', 'figure'),
     Output('future-forecast-graph', 'figure'),
     Output('error-message', 'children'),
     Output('metrics-output', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('forecast-days', 'value'),
     State('num-simulations', 'value'),
     State('future-forecast-days', 'value')]
)
def run_simulation(n_clicks, ticker, forecast_days, num_simulations, future_forecast_days):
    if n_clicks == 0:
        return {}, {}, {}, "", ""
    
    # Validate inputs
    try:
        forecast_days = int(forecast_days)
        num_simulations = int(num_simulations)
        future_forecast_days = int(future_forecast_days)
    except ValueError:
        return {}, {}, {}, "All forecast days and simulation numbers must be integers.", ""
    
    # Fetch historical data
    try:
        prices, df = stock.get_data(ticker)
    except Exception as e:
        return {}, {}, {}, f"Error fetching data for {ticker}: {e}", ""
    
    # Build historical plot
    history_fig = go.Figure()
    history_fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Close'],
        mode='lines',
        name=f"{ticker.upper()} Historical"
    ))
    history_fig.update_layout(
        title=f"{ticker.upper()} Historical Closing Prices",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    
    # Compute drift and volatility (assuming daily data: delta_t = 1)
    mu, sigma = stock.compute_drift_and_diffusion(prices, delta_t=1)
    
    # --- Historical Forecast: Compare simulation with true historical values ---
    dt = 1  # 1 day per step
    start_index = -forecast_days
    initial_hist = prices[start_index]
    
    simulations_hist = np.zeros((num_simulations, forecast_days))
    simulations_hist[:, 0] = initial_hist
    np.random.seed(42)
    for i in range(num_simulations):
        dW = np.random.normal(0, np.sqrt(dt), size=forecast_days)
        for t in range(1, forecast_days):
            simulations_hist[i, t] = simulations_hist[i, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t-1])
    avg_simulation_hist = np.mean(simulations_hist, axis=0)
    true_values = prices[start_index:]
    time_axis_hist = np.arange(forecast_days)
    
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(
        x=time_axis_hist,
        y=true_values,
        mode='lines+markers',
        name="True Historical (Last N Days)"
    ))
    forecast_fig.add_trace(go.Scatter(
        x=time_axis_hist,
        y=avg_simulation_hist,
        mode='lines+markers',
        name=f"Average GBM (N={num_simulations})"
    ))
    forecast_fig.update_layout(
        title=f"Historical Forecast vs. True (Last {forecast_days} Days)",
        xaxis_title="Forecast Steps (Days)",
        yaxis_title="Price"
    )
    
    # --- Future Forecast: Forecast into the future from the last data point ---
    initial_future = prices[-1]
    simulations_future = np.zeros((num_simulations, future_forecast_days))
    simulations_future[:, 0] = initial_future
    np.random.seed(42)
    for i in range(num_simulations):
        dW = np.random.normal(0, np.sqrt(dt), size=future_forecast_days)
        for t in range(1, future_forecast_days):
            simulations_future[i, t] = simulations_future[i, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t-1])
    avg_simulation_future = np.mean(simulations_future, axis=0)
    time_axis_future = np.arange(future_forecast_days)
    
    future_forecast_fig = go.Figure()
    future_forecast_fig.add_trace(go.Scatter(
        x=time_axis_future,
        y=avg_simulation_future,
        mode='lines+markers',
        name=f"Future Average GBM (N={num_simulations})"
    ))
    future_forecast_fig.update_layout(
        title=f"Future Forecast for Next {future_forecast_days} Days (from Last Data Point)",
        xaxis_title="Forecast Steps (Days)",
        yaxis_title="Price"
    )
    
    # Metrics text for mu and sigma
    metrics_text = f"Calculated μ: {mu:.5f} | Calculated σ: {sigma:.5f}"
    
    return history_fig, forecast_fig, future_forecast_fig, "", metrics_text

if __name__ == '__main__':
    app.run_server(debug=True)
