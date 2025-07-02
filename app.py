import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sqlalchemy
import os
from dotenv import load_dotenv
load_dotenv()

# --- Database Setup ---
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set. Make sure you have a .env file.")

# Fix for deprecated postgres:// URL format
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

engine = sqlalchemy.create_engine(DATABASE_URL)

# --- Data Loading Function ---
def load_tickers():
    with engine.connect() as connection:
        query = "SELECT DISTINCT symbol FROM stock_data WHERE symbol != 'QQQ'"
        tickers = pd.read_sql(query, connection)['symbol'].tolist()
        return tickers

def load_data_for_ticker(ticker):
    with engine.connect() as connection:
        query = sqlalchemy.text("SELECT * FROM stock_data WHERE symbol = :ticker OR symbol = 'QQQ'")
        df = pd.read_sql(query, connection, params={'ticker': ticker}, index_col='id')
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        
        stock_data = df[df['symbol'] == ticker].copy()
        benchmark_data = df[df['symbol'] == 'QQQ'].copy()
        return stock_data, benchmark_data

# --- UI Enhancements ---
# Color palette for tickers
TICKER_COLORS = [
    '#E6F3FF', '#F0FFF0', '#FFF5E6', '#F5F5F5', '#E6E6FA', 
    '#FFF0F5', '#F0F8FF', '#FAEBD7', '#F5FFFA', '#FFFACD'
]

unique_tickers = load_tickers()
if not unique_tickers:
    raise ValueError("No tickers found in the database. Please run tech_data.py to populate it.")
ticker_color_map = {ticker: TICKER_COLORS[i % len(TICKER_COLORS)] for i, ticker in enumerate(unique_tickers)}

# App initialization
app = dash.Dash(__name__)
server = app.server  # This is crucial for Vercel deployment

app.layout = html.Div([
    # Main Content Area
    html.Div([
        html.H1('Financial Analysis Tool'),
        
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[{'label': ticker, 'value': ticker} for ticker in unique_tickers],
            value=unique_tickers[0]
        ),
        
        html.Div([
            html.H4('Technical Indicators', style={'margin-top': '20px', 'margin-bottom': '10px'}),
            dcc.Checklist(
                id='indicator-checklist',
                options=[
                    {'label': 'Bollinger Bands', 'value': 'bb'},
                    {'label': 'RSI', 'value': 'rsi'},
                    {'label': 'MACD', 'value': 'macd'},
                    {'label': 'Ichimoku Cloud', 'value': 'ichimoku'},
                    {'label': 'ADX', 'value': 'adx'},
                    {'label': 'Parabolic SAR', 'value': 'psar'},
                    {'label': 'Donchian Channels', 'value': 'donchian'},
                    {'label': 'Rate of Change', 'value': 'roc'},
                    {'label': 'Elliott Wave Oscillator', 'value': 'ewo'},
                ],
                value=[],
                labelStyle={'display': 'inline-block', 'margin-right': '15px', 'cursor': 'pointer'},
                style={'padding-bottom': '15px', 'border-bottom': '1px solid #ddd'}
            ),
        ], style={'textAlign': 'center'}),

        dcc.Graph(id='stock-graph'),
        
        html.Div([
            html.Button('Buy Signal', id='buy-button', n_clicks=0, style={'margin-right': '10px'}),
            html.Button('Sell Signal', id='sell-button', n_clicks=0, style={'margin-right': '10px'}),
            html.Button('Remove Last Signal', id='remove-last-button', n_clicks=0, style={'margin-right': '10px'}),
        ], style={'textAlign': 'center', 'padding': '20px', 'margin-top': '50px'}),
        
        html.Div(id='selected-point-info', style={'margin-top': '10px', 'textAlign': 'center'}),
        html.Div(id='save-status', style={'margin-top': '10px', 'textAlign': 'center'}),
        html.Div(id='trade-profitability-status', style={'margin-top': '10px', 'textAlign': 'center', 'fontWeight': 'bold'}),

        html.Div([
            html.Div([
                html.H3('Selected Signals', style={'textAlign': 'center'}),
                dash_table.DataTable(
                    id='signals-table',
                    columns=[
                        {'name': 'Date', 'id': 'Date'},
                        {'name': 'Ticker', 'id': 'ticker'},
                        {'name': 'Close', 'id': 'Close', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                        {'name': 'Signal', 'id': 'signal'},
                    ],
                    data=[],
                    row_deletable=True,
                    style_table={
                        'height': '300px',
                        'overflowY': 'auto',
                        'width': '100%',
                        'borderRadius': '18px',
                        'boxShadow': '0 2px 12px rgba(0,0,0,0.07)'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'fontWeight': 'bold',
                        'border': 'none',
                        'fontSize': '16px',
                        'padding': '10px 0',
                    },
                    style_header={
                        'backgroundColor': '#fff',
                        'fontWeight': 'bold',
                        'fontSize': '20px',
                        'textAlign': 'center',
                        'border': 'none',
                    },
                    style_data_conditional=[],
                    fixed_rows={'headers': True}
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
            
            html.Div([
                html.H3('Profitable Trades', style={'textAlign': 'center'}),
                dash_table.DataTable(
                    id='profitable-trades-table',
                    columns=[
                        {'name': 'Ticker', 'id': 'Ticker'},
                        {'name': 'Buy Date', 'id': 'buy_date'},
                        {'name': 'Buy Price', 'id': 'price_at_buy', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                        {'name': 'Sell Date', 'id': 'sell_date'},
                        {'name': 'Sell Price', 'id': 'price_at_sell', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                        {'name': 'Return %', 'id': 'return_pct', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Benchmark Return %', 'id': 'NSDAQ100etf_return_pct', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    ],
                    data=[],
                    style_table={
                        'height': '300px',
                        'overflowY': 'auto',
                        'width': '100%',
                        'borderRadius': '18px',
                        'boxShadow': '0 2px 12px rgba(0,0,0,0.07)'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'fontWeight': 'bold',
                        'border': 'none',
                        'fontSize': '16px',
                        'padding': '10px 0',
                    },
                    style_header={
                        'backgroundColor': '#fff',
                        'fontWeight': 'bold',
                        'fontSize': '20px',
                        'textAlign': 'center',
                        'border': 'none',
                    },
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'width': '100%'}),

        dcc.Store(id='signals-storage', data=[]),
        dcc.Store(id='profitable-trades-storage', data=[]),

    ], className='main-content')
], className='app-container')

@app.callback(
    Output('stock-graph', 'figure'),
    Input('ticker-dropdown', 'value'),
    Input('signals-storage', 'data'),
    Input('indicator-checklist', 'value')
)
def update_graph(selected_ticker, signals, selected_indicators):
    # --- DATA IS NOW LOADED HERE ---
    df, _ = load_data_for_ticker(selected_ticker)
    df.sort_values('Date', inplace=True)

    # --- TradingView Style Implementation ---

    # 1. Define TradingView colors
    INCREASING_COLOR = '#26a69a'
    DECREASING_COLOR = '#ef5350'
    GRID_COLOR = '#EAEAEA'

    # 2. Determine number of rows needed for indicators
    indicator_rows = [ind for ind in ['rsi', 'macd', 'adx', 'roc', 'ewo'] if ind in selected_indicators]
    num_rows = 2 + len(indicator_rows)
    row_heights = [0.7] + [0.15] * (len(indicator_rows) + 1) # Main chart, volume, then indicators

    specs = [[{"secondary_y": False}], [{"secondary_y": False}]] # Price and Volume panes
    for _ in indicator_rows:
        specs.append([{"secondary_y": False}]) # Indicator panes

    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True,
                          vertical_spacing=0.02,
                          row_heights=row_heights,
                          specs=specs)

    # 3. Add Candlestick Trace (Price Pane - Row 1)
    fig.add_trace(go.Candlestick(x=df['Date'],
                                   open=df['Open'],
                                   high=df['High'],
                                   low=df['Low'],
                                   close=df['Close'],
                                   name='Price',
                                   increasing_fillcolor=INCREASING_COLOR,
                                   increasing_line_color=INCREASING_COLOR,
                                   decreasing_fillcolor=DECREASING_COLOR,
                                   decreasing_line_color=DECREASING_COLOR),
                  row=1, col=1)

    # Add price-based indicators to Price Pane
    if 'bb' in selected_indicators and all(c in df.columns for c in ['bb_upper', 'bb_lower', 'bb_middle']):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['bb_upper'], mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['bb_lower'], mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    if 'ichimoku' in selected_indicators and all(c in df.columns for c in ['ichimoku_senkou_a', 'ichimoku_senkou_b']):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['ichimoku_senkou_a'], mode='lines', name='Ichimoku A', line=dict(color='rgba(0, 255, 0, 0.2)')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['ichimoku_senkou_b'], mode='lines', name='Ichimoku B', line=dict(color='rgba(255, 0, 0, 0.2)'), fill='tonexty'), row=1, col=1)
    if 'psar' in selected_indicators and 'psar' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['psar'], mode='markers', name='Parabolic SAR', marker=dict(color='purple', size=4)), row=1, col=1)
    if 'donchian' in selected_indicators and all(c in df.columns for c in ['donchian_upper', 'donchian_lower']):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['donchian_upper'], mode='lines', name='Donchian Upper', line=dict(color='cyan', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['donchian_lower'], mode='lines', name='Donchian Lower', line=dict(color='cyan', width=1, dash='dash')), row=1, col=1)

    # 4. Add Volume Bar Trace (Volume Pane - Row 2)
    volume_colors = [INCREASING_COLOR if row['Close'] >= row['Open'] else DECREASING_COLOR for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'],
                         marker_color=volume_colors,
                         name='Volume'),
                  row=2, col=1)

    # 5. Add Indicator Traces (Rows 3+)
    current_row = 3
    for indicator in indicator_rows:
        if indicator == 'rsi' and 'rsi' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['rsi'], mode='lines', name='RSI'), row=current_row, col=1)
            fig.update_yaxes(title_text="RSI", side='right', gridcolor=GRID_COLOR, row=current_row, col=1)
        elif indicator == 'macd' and all(c in df.columns for c in ['macd_line', 'macd_signal', 'macd_histogram']):
            fig.add_trace(go.Scatter(x=df['Date'], y=df['macd_line'], mode='lines', name='MACD Line'), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['macd_signal'], mode='lines', name='MACD Signal'), row=current_row, col=1)
            fig.add_trace(go.Bar(x=df['Date'], y=df['macd_histogram'], name='MACD Hist'), row=current_row, col=1)
            fig.update_yaxes(title_text="MACD", side='right', gridcolor=GRID_COLOR, row=current_row, col=1)
        elif indicator == 'adx' and 'adx' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['adx'], mode='lines', name='ADX'), row=current_row, col=1)
            fig.update_yaxes(title_text="ADX", side='right', gridcolor=GRID_COLOR, row=current_row, col=1)
        elif indicator == 'roc' and 'roc' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['roc'], mode='lines', name='ROC'), row=current_row, col=1)
            fig.update_yaxes(title_text="ROC", side='right', gridcolor=GRID_COLOR, row=current_row, col=1)
        elif indicator == 'ewo' and 'elliott_wave_oscillator' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['elliott_wave_oscillator'], mode='lines', name='EWO'), row=current_row, col=1)
            fig.update_yaxes(title_text="EWO", side='right', gridcolor=GRID_COLOR, row=current_row, col=1)
        current_row += 1

    # 6. Update the overall layout to match TradingView
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=10, r=50, t=10, b=10)
    )

    # 7. Update Y-Axes styles
    fig.update_yaxes(
        side='right',
        tickfont=dict(size=12, color='#333'),
        gridcolor=GRID_COLOR,
        row=1, col=1
    )
    fig.update_yaxes(
        showticklabels=False, # Hide volume axis labels
        gridcolor=GRID_COLOR,
        row=2, col=1
    )

    # 8. Update X-Axis style for all panes
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        tickfont=dict(size=12, color='#787878'),
        showticklabels=True # Ensure x-axis labels are visible on the bottom pane
    )
    
    # Hide x-axis labels on all but the bottom chart
    for i in range(1, num_rows):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

    # Add annotations for signals
    annotations = []
    for signal in signals:
        if signal['ticker'] == selected_ticker:
            signal_date = pd.to_datetime(signal['Date'])
            signal_df = df[df['Date'] == signal_date]
            if not signal_df.empty:
                if signal['signal'] == 'buy':
                    annotations.append(dict(x=signal_date, 
                                            y=signal_df.iloc[0]['Low'], 
                                            text="B", showarrow=True, arrowhead=2, 
                                            ax=0, ay=20, bgcolor="#26a69a"))
                elif signal['signal'] == 'sell':
                    annotations.append(dict(x=signal_date, 
                                            y=signal_df.iloc[0]['High'], 
                                            text="S", showarrow=True, arrowhead=2, 
                                            ax=0, ay=-20, bgcolor="#ef5350"))
    fig.update_layout(annotations=annotations)

    return fig

@app.callback(
    Output('selected-point-info', 'children'),
    Input('stock-graph', 'clickData')
)
def display_click_data(clickData):
    if clickData is None:
        return "Click on a point in the graph to select it."
    else:
        point = clickData['points'][0]
        date = point['x']
        return f"Selected Date: {date}"

@app.callback(
    Output('signals-storage', 'data'),
    Output('trade-profitability-status', 'children'),
    Output('profitable-trades-storage', 'data'),
    Input('buy-button', 'n_clicks'),
    Input('sell-button', 'n_clicks'),
    Input('remove-last-button', 'n_clicks'),
    Input('signals-table', 'data_previous'),
    State('stock-graph', 'clickData'),
    State('ticker-dropdown', 'value'),
    State('signals-storage', 'data'),
    State('profitable-trades-storage', 'data')
)
def update_signals(buy_clicks, sell_clicks, remove_clicks, table_data_previous, clickData, selected_ticker, existing_signals, profitable_trades):
    # --- DATA IS NOW LOADED HERE ---
    stock_data, benchmark_data = load_data_for_ticker(selected_ticker)
    
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    profitability_status = ""
    
    if 'remove-last-button' in changed_id:
        if existing_signals:
            # Find the last signal for the current ticker and remove it
            for i in range(len(existing_signals) - 1, -1, -1):
                if existing_signals[i]['ticker'] == selected_ticker:
                    existing_signals.pop(i)
                    break
        return existing_signals, profitability_status, profitable_trades

    if table_data_previous is not None and len(existing_signals) > len(table_data_previous):
        # A row was deleted from the table by the user
        # The new data is the current state of the table
        current_dates = {row['Date'] for row in table_data_previous}
        existing_signals = [s for s in existing_signals if s['Date'] in current_dates]

    if not clickData or ('buy-button' not in changed_id and 'sell-button' not in changed_id):
        return existing_signals, profitability_status, profitable_trades

    point = clickData['points'][0]
    date = point['x']
    
    # Check for duplicates and enforce trading logic
    last_signal_for_ticker = None
    for s in reversed(existing_signals):
        if s['ticker'] == selected_ticker:
            last_signal_for_ticker = s
            break

    signal_type = None
    if 'buy-button' in changed_id:
        if last_signal_for_ticker is None or last_signal_for_ticker['signal'] == 'sell':
            signal_type = 'buy'
    elif 'sell-button' in changed_id:
        if last_signal_for_ticker and last_signal_for_ticker['signal'] == 'buy':
            signal_type = 'sell'

    if signal_type:
        signal_data = stock_data[(stock_data['ticker'] == selected_ticker) & (stock_data['Date'] == date)].to_dict('records')[0]
        signal_data['signal'] = signal_type
        existing_signals.append(signal_data)

        if signal_type == 'sell':
            # Check for profitability
            buy_signal = last_signal_for_ticker
            sell_signal = signal_data

            buy_date = pd.to_datetime(buy_signal['Date'])
            sell_date = pd.to_datetime(sell_signal['Date'])
            price_at_buy = buy_signal['Close']
            price_at_sell = sell_signal['Close']

            # Stock return
            return_value = price_at_sell - price_at_buy
            return_pct = (return_value / price_at_buy) * 100

            # Benchmark return
            benchmark_buy_price = benchmark_data[benchmark_data['Date'] == buy_date]['Close'].iloc[0]
            benchmark_sell_price = benchmark_data[benchmark_data['Date'] == sell_date]['Close'].iloc[0]
            benchmark_return_value = benchmark_sell_price - benchmark_buy_price
            benchmark_return_pct = (benchmark_return_value / benchmark_buy_price) * 100

            if return_pct > benchmark_return_pct:
                profitability_status = f"Profitable Trade! Return: {return_pct:.2f}% vs Benchmark: {benchmark_return_pct:.2f}%"
                profitable_trades.append({
                    'Ticker': selected_ticker,
                    'buy_date': buy_date.strftime('%Y-%m-%d'),
                    'price_at_buy': price_at_buy,
                    'sell_date': sell_date.strftime('%Y-%m-%d'),
                    'price_at_sell': price_at_sell,
                    'return_value': return_value,
                    'return_pct': return_pct,
                    'NSDAQ100etf_buy_date': buy_date.strftime('%Y-%m-%d'),
                    'NSDAQ100etf_price_at_buy': benchmark_buy_price,
                    'NSDAQ100etf_sell_date': sell_date.strftime('%Y-%m-%d'),
                    'NSDAQ100etf_price_at_sell': benchmark_sell_price,
                    'NSDAQ100etf_return_value': benchmark_return_value,
                    'NSDAQ100etf_return_pct': benchmark_return_pct
                })
            else:
                profitability_status = f"Not a Profitable Trade. Return: {return_pct:.2f}% vs Benchmark: {benchmark_return_pct:.2f}%"
            
            return existing_signals, profitability_status, profitable_trades

    return existing_signals, profitability_status, profitable_trades

@app.callback(
    Output('signals-table', 'data'),
    Output('signals-table', 'style_data_conditional'),
    Input('signals-storage', 'data')
)
def update_signals_table(signals):
    if not signals:
        return [], []

    df = pd.DataFrame(signals)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    table_data = df[['Date', 'ticker', 'Close', 'signal']].to_dict('records')

    styles = [
        {
            'if': {'filter_query': '{signal} = "buy"', 'column_id': 'signal'},
            'color': '#28a745',
            'fontWeight': 'bold',
        },
        {
            'if': {'filter_query': '{signal} = "sell"', 'column_id': 'signal'},
            'color': '#dc3545',
            'fontWeight': 'bold',
        }
    ]

    return table_data, styles

@app.callback(
    Output('profitable-trades-table', 'data'),
    Input('profitable-trades-storage', 'data')
)
def update_profitable_trades_table(trades):
    return trades

# This is required for Vercel
if __name__ == '__main__':
    app.run_server(debug=False)