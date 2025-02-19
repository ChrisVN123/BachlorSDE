import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import stock  # This is your stock.py file

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # For deployment if needed

app.layout = html.Div([
    html.H1("Stock Forecast Simulator"),
    html.Div([
        html.Label("Stock Ticker:"),
        dcc.Input(id='ticker-input', type='text', value='AAPL', style={'marginRight': '20px'}),
        
        html.Label("Forecast Days:"),
        dcc.Input(id='forecast-days', type='number', value=100, style={'marginRight': '20px'}),
        
        html.Label("Number of Simulations:"),
        dcc.Input(id='num-simulations', type='number', value=100, style={'marginRight': '20px'}),
        
        html.Button("Run Simulation", id='run-button', n_clicks=0)
    ], style={'marginBottom': '20px'}),
    
    html.Div(id='error-message', style={'color': 'red', 'marginBottom': '20px'}),
    
    html.Div([
        dcc.Graph(id='history-graph', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='forecast-graph', style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ])
])

@app.callback(
    [Output('history-graph', 'figure'),
     Output('forecast-graph', 'figure'),
     Output('error-message', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('forecast-days', 'value'),
     State('num-simulations', 'value')]
)
def run_simulation(n_clicks, ticker, forecast_days, num_simulations):
    # Only run when the button is clicked
    if n_clicks == 0:
        return {}, {}, ""
    
    # Validate inputs
    try:
        forecast_days = int(forecast_days)
        num_simulations = int(num_simulations)
    except ValueError:
        return {}, {}, "Forecast days and number of simulations must be integers."
    
    # Fetch historical data from stock.py
    try:
        prices, df = stock.get_data(ticker)
    except Exception as e:
        return {}, {}, f"Error fetching data for {ticker}: {e}"
    
    # Build historical plot using Plotly
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
    
    # Compute drift and volatility from the entire price series (assuming daily data, delta_t=1)
    mu, sigma = stock.compute_drift_and_diffusion(prices, delta_t=1)
    
    # Run N GBM simulations for the forecast period.
    # Use the last 'forecast_days' points from the historical data as the starting point.
    dt = 1  # 1 day per step
    start_index = -forecast_days
    initial = prices[start_index]
    
    simulations = np.zeros((num_simulations, forecast_days))
    simulations[:, 0] = initial
    np.random.seed(42)  # For reproducibility
    for i in range(num_simulations):
        dW = np.random.normal(0, np.sqrt(dt), size=forecast_days)
        for t in range(1, forecast_days):
            simulations[i, t] = simulations[i, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t-1])
    avg_simulation = np.mean(simulations, axis=0)
    true_values = prices[start_index:]
    time_axis = np.arange(forecast_days)
    
    # Build forecast plot using Plotly
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(
        x=time_axis,
        y=true_values,
        mode='lines+markers',
        name="True Historical"
    ))
    forecast_fig.add_trace(go.Scatter(
        x=time_axis,
        y=avg_simulation,
        mode='lines+markers',
        name=f"Average GBM (N={num_simulations})"
    ))
    forecast_fig.update_layout(
        title="Forecast vs. True Historical (Last {} Days)".format(forecast_days),
        xaxis_title="Forecast Steps (Days)",
        yaxis_title="Price"
    )
    
    return history_fig, forecast_fig, ""

if __name__ == '__main__':
    app.run_server(debug=True)
