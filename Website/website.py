# app.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pandas as pd

import stock
import Ornstein_prices

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ------------------------------------------------
# 1. Define the two page layouts
# ------------------------------------------------

# Page 1: Stock Forecast
stock_layout = html.Div([
    html.H2("Stock Forecast Simulator"),

    html.Div([
        html.Label("Stock Ticker:"),
        dcc.Input(id='ticker-input', type='text', value='AAPL', style={'marginRight': '20px'}),

        html.Label("Historical Forecast Days (N):"),
        dcc.Input(id='forecast-days', type='number', value=100, style={'marginRight': '20px'}),

        html.Label("Number of Simulations:"),
        dcc.Input(id='num-simulations', type='number', value=10, style={'marginRight': '20px'}),

        html.Label("Future Forecast Days:"),
        dcc.Input(id='future-forecast-days', type='number', value=200, style={'marginRight': '20px'}),

        html.Button("Run Simulation", id='run-button', n_clicks=0)
    ], style={'marginBottom': '20px'}),

    # Error message
    html.Div(id='error-message', style={'color': 'red', 'marginBottom': '20px'}),

    # Graphs
    dcc.Graph(id='historical-forecast-graph', style={'width': '100%', 'marginBottom': '30px'}),
    dcc.Graph(id='combined-graph', style={'width': '100%'}),

    # Metrics (μ and σ)
    html.Div(id='metrics-output', style={'marginTop': '20px', 'fontWeight': 'bold'})
])

# Page 2: Electricity Price Estimation
electricity_layout = html.Div([
    html.H2("Electricity Price Estimation (OU)"),
    html.Div([
        html.P("Estimate Ornstein–Uhlenbeck parameters for electricity prices."),
        html.Button("Run OU Estimation", id="ou-run-button", n_clicks=0)
    ], style={'marginBottom': '20px'}),

    dcc.Graph(id="ou-graph", style={'width': '100%'}),
    html.Div(id='ou-params', style={'marginTop': '20px', 'fontWeight': 'bold'}),
    
    # New plots
    html.H4("Residuals Histogram"),
    dcc.Graph(id="residuals-histogram", style={'width': '100%', 'marginBottom': '20px'}),
    
    html.H4("ACF Comparison"),
    dcc.Graph(id="acf-plot", style={'width': '100%'})
])


# ------------------------------------------------
# 2. Main layout with top navbar + page content
# ------------------------------------------------
# Main layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    # -------------------------
    # "Nicer" Navbar Section
    # -------------------------
    html.Nav([
        # A brand title or app name at the left
        html.Div("Bachelor Project Dashboard", 
                 style={"fontWeight": "bold", "fontSize": "1.4rem"}),

        # Navigation links in an <ul> list
        html.Ul([
            # Each <li> is a menu item
            html.Li(
                html.A("Stock Forecast", href="/stock",
                       style={"textDecoration": "none", 
                              "color": "#007bff", 
                              "padding": "8px",
                              "fontWeight": "bold"})
            ),
            html.Li(
                html.A("Electricity Price", href="/electricity",
                       style={"textDecoration": "none", 
                              "color": "#007bff", 
                              "padding": "8px",
                              "fontWeight": "bold"})
            )
        ], style={
            "display": "flex",
            "listStyleType": "none",
            "margin": 0,
            "padding": 0,
            "gap": "15px"  # space between links
        })
    ], style={
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "backgroundColor": "#f8f9fa",
        "padding": "10px 20px",
        "borderBottom": "1px solid #ccc"
    }),

    # Where each page’s layout is rendered
    html.Div(id='page-content', style={"padding": "20px"})
])

# Callback to choose which layout to display
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == "/stock" or pathname == "/":
        return stock_layout
    elif pathname == "/electricity":
        return electricity_layout
    else:
        return html.Div([
            html.H3("404: Page Not Found"),
            html.P(f"The pathname {pathname} is invalid.")
        ])

# ------------------------------------------------
# 4. Callbacks for the Stock page
# ------------------------------------------------
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
        prices, df = stock.get_data(ticker,5)
    except Exception as e:
        return {}, {}, f"Error fetching data for {ticker}: {e}", ""

    # Compute drift & volatility
    mu, sigma = stock.compute_drift_and_diffusion(prices, delta_t=1)

    # ------------------------------------------------
    # 1) Historical forecast plot (last N days)
    # ------------------------------------------------
    dt = 1
    start_index = -forecast_days if forecast_days < len(prices) else 0
    initial_hist = prices[start_index]

    simulations_hist = np.zeros((num_simulations, forecast_days))
    simulations_hist[:, 0] = initial_hist
    np.random.seed(42)
    for i in range(num_simulations):
        dW = np.random.normal(0, np.sqrt(dt), size=forecast_days)
        for t in range(1, forecast_days):
            simulations_hist[i, t] = (simulations_hist[i, t-1] *
                                      np.exp((mu - 0.5 * sigma**2)*dt + sigma * dW[t-1]))
    avg_simulation_hist = np.mean(simulations_hist, axis=0)

    # True historical values for last N days
    true_values = prices[start_index:]
    time_axis_hist = np.arange(len(true_values))

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
        xaxis_title="Forecast Steps",
        yaxis_title="Price"
    )

    # ------------------------------------------------
    # 2) Combined Plot: Historical + Future Forecast
    # ------------------------------------------------
    combined_fig = go.Figure()
    # Plot full historical data
    combined_fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name="Historical Data"
    ))

    # Future forecast
    if len(prices) == 0:
        return {}, {}, "No price data found", ""

    initial_future = prices[-1]
    simulations_future = np.zeros((num_simulations, future_forecast_days))
    simulations_future[:, 0] = initial_future
    np.random.seed(42)
    for i in range(num_simulations):
        dW = np.random.normal(0, np.sqrt(dt), size=future_forecast_days)
        for t in range(1, future_forecast_days):
            simulations_future[i, t] = (simulations_future[i, t-1] *
                                        np.exp((mu - 0.5*sigma**2)*dt + sigma*dW[t-1]))
    avg_simulation_future = np.mean(simulations_future, axis=0)

    # Future dates
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=future_forecast_days, freq='D')

    combined_fig.add_trace(go.Scatter(
        x=future_dates,
        y=avg_simulation_future,
        mode='lines+markers',
        name="Future Forecast",
        line=dict(dash='dot')  # dashed line
    ))
    combined_fig.update_layout(
        title=f"{ticker.upper()} Combined Historical Data + Future Forecast (Next {future_forecast_days} Days)",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    metrics_text = f"Calculated μ: {mu:.5f} | Calculated σ: {sigma:.5f}"
    return historical_forecast_fig, combined_fig, "", metrics_text

# ------------------------------------------------
# 5. Callback for the Electricity (OU) page
# ------------------------------------------------
@app.callback(
    [Output('ou-graph', 'figure'),
     Output('ou-params', 'children'),
     Output('residuals-histogram', 'figure'),
     Output('acf-plot', 'figure')],
    Input('ou-run-button', 'n_clicks')
)
def run_ou_estimation(n_clicks):
    if n_clicks == 0:
        return go.Figure(), "", go.Figure(), go.Figure()
    hours, prices, sim_path, mu_hat, sigma_hat, theta_hat = Ornstein_prices.simple_ornstein("priceData.csv")

    fig, param_text = Ornstein_prices.run_ou_estimation(prices, hours, sim_path, mu_hat, sigma_hat, theta_hat)

    hist_plot, acf_plot = Ornstein_prices.plot_residuals_and_acf(prices=prices, simulated_prices=sim_path, lags=30)
    
    return fig, param_text, hist_plot, acf_plot


# ------------------------------------------------
# 6. Run the server
# ------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
