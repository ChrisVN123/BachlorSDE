import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import stock  # Import our stock.py module

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # For potential deployment

app.layout = html.Div([
    html.H1("Stock Forecast Simulator"),
    html.Div([
        html.Label("Stock Ticker:"),
        dcc.Input(id='ticker-input', type='text', value='AAPL', style={'marginRight': '20px'}),
        
        html.Label("Historical Forecast Days (N):"),
        dcc.Input(id='forecast-days', type='number', value=100, style={'marginRight': '20px'}),
        
        html.Label("Number of Simulations:"),
        dcc.Input(id='num-simulations', type='number', value=100, style={'marginRight': '20px'}),
        
        html.Label("Future Forecast Days:"),
        dcc.Input(id='future-forecast-days', type='number', value=30, style={'marginRight': '20px'}),
        
        html.Button("Run Simulation", id='run-button', n_clicks=0)
    ], style={'marginBottom': '20px'}),
    
    html.Div(id='error-message', style={'color': 'red', 'marginBottom': '20px'}),
    
    # Graph for historical forecast vs true (last N days)
    dcc.Graph(id='historical-forecast-graph', style={'width': '100%', 'marginBottom': '30px'}),
    
    # Graph for combined historical data and future forecast continuation
    dcc.Graph(id='combined-graph', style={'width': '100%'}),
    
    # Box for outputting μ and σ
    html.Div(id='metrics-output', style={'marginTop': '20px', 'fontWeight': 'bold'})
])

@app.callback(
    [Output('historical-forecast-graph', 'figure'),
     Output('combined-graph', 'figure'),
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
        return {}, {}, "", ""
    
    # Validate inputs
    try:
        forecast_days = int(forecast_days)
        num_simulations = int(num_simulations)
        future_forecast_days = int(future_forecast_days)
    except ValueError:
        return {}, {}, "All forecast days and simulation numbers must be integers.", ""
    
    # Fetch historical data
    try:
        prices, df = stock.get_data(ticker)
    except Exception as e:
        return {}, {}, f"Error fetching data for {ticker}: {e}", ""
    
    # Compute drift and volatility (assuming daily data, so delta_t = 1)
    mu, sigma = stock.compute_drift_and_diffusion(prices, delta_t=1)
    
    # -----------------------------------------
    # 1. Historical Forecast Plot (Last N Days)
    # -----------------------------------------
    dt = 1  # 1 day per step
    start_index = -forecast_days
    initial_hist = prices[start_index]
    
    # Run simulations starting from forecast_days back
    simulations_hist = np.zeros((num_simulations, forecast_days))
    simulations_hist[:, 0] = initial_hist
    np.random.seed(42)
    for i in range(num_simulations):
        dW = np.random.normal(0, np.sqrt(dt), size=forecast_days)
        for t in range(1, forecast_days):
            simulations_hist[i, t] = simulations_hist[i, t-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma * dW[t-1])
    avg_simulation_hist = np.mean(simulations_hist, axis=0)
    
    # True historical values for the last N days
    true_values = prices[start_index:]
    time_axis_hist = np.arange(forecast_days)
    
    historical_forecast_fig = go.Figure()
    historical_forecast_fig.add_trace(go.Scatter(
        x=time_axis_hist,
        y=true_values,
        mode='lines+markers',
        name="True Historical (Last N Days)"
    ))
    historical_forecast_fig.add_trace(go.Scatter(
        x=time_axis_hist,
        y=avg_simulation_hist,
        mode='lines+markers',
        name=f"Average GBM (N={num_simulations})"
    ))
    historical_forecast_fig.update_layout(
        title=f"Historical Forecast vs. True (Last {forecast_days} Days)",
        xaxis_title="Forecast Steps (Days)",
        yaxis_title="Price"
    )
    
    # -----------------------------------------
    # 2. Combined Plot: Historical Data + Future Forecast
    # -----------------------------------------
    combined_fig = go.Figure()
    # Plot full historical data
    combined_fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name="Historical Data"
    ))
    
    # Future forecast: starting from last historical data point
    initial_future = prices[-1]
    simulations_future = np.zeros((num_simulations, future_forecast_days))
    simulations_future[:, 0] = initial_future
    np.random.seed(42)
    for i in range(num_simulations):
        dW = np.random.normal(0, np.sqrt(dt), size=future_forecast_days)
        for t in range(1, future_forecast_days):
            simulations_future[i, t] = simulations_future[i, t-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma * dW[t-1])
    avg_simulation_future = np.mean(simulations_future, axis=0)
    
    # Create future dates continuing from the last historical date
    last_date = pd.to_datetime(df.index[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=future_forecast_days, freq='D')
    
    # Append the forecast as a new trace (dotted line, new color)
    combined_fig.add_trace(go.Scatter(
        x=future_dates,
        y=avg_simulation_future,
        mode='lines+markers',
        name="Future Forecast",
        line=dict(color='red', dash='dot')
    ))
    
    combined_fig.update_layout(
        title=f"{ticker.upper()} Combined Historical Data and Future Forecast (Next {future_forecast_days} Days)",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    
    # Metrics text for μ and σ
    metrics_text = f"Calculated μ: {mu:.5f} | Calculated σ: {sigma:.5f}"
    
    return historical_forecast_fig, combined_fig, "", metrics_text

if __name__ == '__main__':
    app.run_server(debug=True)
