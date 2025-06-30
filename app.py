import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from data_utils import load_stock_data, load_benchmark_data

# Load data
stock_data = load_stock_data('stock_data')
benchmark_data = load_benchmark_data('stock_data')

# --- UI Enhancements ---
# Color palette for tickers
TICKER_COLORS = [
    '#E6F3FF', '#F0FFF0', '#FFF5E6', '#F5F5F5', '#E6E6FA', 
    '#FFF0F5', '#F0F8FF', '#FAEBD7', '#F5FFFA', '#FFFACD'
]

unique_tickers = stock_data['ticker'].unique()
ticker_color_map = {ticker: TICKER_COLORS[i % len(TICKER_COLORS)] for i, ticker in enumerate(unique_tickers)}

# App initialization
app = dash.Dash(__name__, external_stylesheets=['/assets/styles.css'])

app.layout = html.Div([
    # Left-side Toolbar
    html.Div([
        html.Button('ƒ', id='indicator-button', className='toolbar-button')
    ], className='toolbar'),

    # Main Content Area
    html.Div([
        html.H1('Financial Analysis Tool'),
        
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[{'label': ticker, 'value': ticker} for ticker in stock_data['ticker'].unique()],
            value=stock_data['ticker'].unique()[0]
        ),
        
        dcc.Graph(id='stock-graph'),
        
        html.Div([
            html.Button('Buy Signal', id='buy-button', n_clicks=0, style={'margin-right': '10px'}),
            html.Button('Sell Signal', id='sell-button', n_clicks=0, style={'margin-right': '10px'}),
            html.Button('Save Signals', id='save-button', n_clicks=0, className='button-primary'),
        ], style={'textAlign': 'center', 'padding': '20px'}),
        
        html.Div(id='selected-point-info', style={'margin-top': '10px', 'textAlign': 'center'}),
        html.Div(id='save-status', style={'margin-top': '10px', 'textAlign': 'center'}),

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
            style_table={
                'height': '300px',
                'overflowY': 'auto',
                'width': '920px',
                'margin': '0 auto',
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

        dcc.Store(id='signals-storage', data=[])
    ], className='main-content')
], className='app-container')

@app.callback(
    Output('stock-graph', 'figure'),
    Input('ticker-dropdown', 'value'),
    Input('signals-storage', 'data')
)
def update_graph(selected_ticker, signals):
    df = stock_data[stock_data['ticker'] == selected_ticker].copy()
    df.sort_values('Date', inplace=True)

    # Calculate 50-day SMA
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.03, 
                          row_heights=[0.7, 0.3])

    # Price Pane
    fig.add_trace(go.Candlestick(x=df['Date'],
                                   open=df['Open'],
                                   high=df['High'],
                                   low=df['Low'],
                                   close=df['Close'],
                                   increasing_line_color='#00b0b9', 
                                   decreasing_line_color='#ef5350',
                                   name='Price'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], 
                             mode='lines', name='SMA 50', 
                             line=dict(color='blue', width=1)), row=1, col=1)

    # Volume Pane
    colors = ['#00b0b9' if row['Close'] >= row['Open'] else '#ef5350' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], 
                         marker_color=colors, name='Volume'), row=2, col=1)

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
                                            ax=0, ay=20, bgcolor="#00b0b9"))
                elif signal['signal'] == 'sell':
                    annotations.append(dict(x=signal_date, 
                                            y=signal_df.iloc[0]['High'], 
                                            text="S", showarrow=True, arrowhead=2, 
                                            ax=0, ay=-20, bgcolor="#ef5350"))
    fig.update_layout(annotations=annotations)

    fig.update_layout(
        title=f'{selected_ticker} Price and Volume',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Volume", row=2, col=1)

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
    Input('buy-button', 'n_clicks'),
    Input('sell-button', 'n_clicks'),
    State('stock-graph', 'clickData'),
    State('ticker-dropdown', 'value'),
    State('signals-storage', 'data')
)
def store_signal(buy_clicks, sell_clicks, clickData, selected_ticker, existing_signals):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if not clickData:
        return existing_signals

    point = clickData['points'][0]
    date = point['x']
    
    # Check for duplicates and enforce trading logic
    last_signal_for_ticker = None
    for s in reversed(existing_signals):
        if s['ticker'] == selected_ticker:
            last_signal_for_ticker = s['signal']
            break

    signal_type = None
    if 'buy-button' in changed_id:
        if last_signal_for_ticker is None or last_signal_for_ticker == 'sell':
            signal_type = 'buy'
    elif 'sell-button' in changed_id:
        if last_signal_for_ticker == 'buy':
            signal_type = 'sell'

    if signal_type:
        signal_data = stock_data[(stock_data['ticker'] == selected_ticker) & (stock_data['Date'] == date)].to_dict('records')[0]
        signal_data['signal'] = signal_type
        existing_signals.append(signal_data)

    return existing_signals

@app.callback(
    Output('signals-table', 'data'),
    Output('signals-table', 'style_data_conditional'),
    Input('signals-storage', 'data')
)
def update_signals_table(signals):
    if not signals:
        return [], []

    df = pd.DataFrame(signals)
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
    Output('save-status', 'children'),
    Output('signals-storage', 'data', allow_duplicate=True),
    Input('save-button', 'n_clicks'),
    State('signals-storage', 'data'),
    prevent_initial_call=True
)
def save_signals(n_clicks, signals):
    if n_clicks > 0 and signals:
        # 1. Save raw signals
        df_raw = pd.DataFrame(signals)
        file_exists_raw = os.path.exists('annotated_signals.csv')
        df_raw.to_csv('annotated_signals.csv', mode='a', header=not file_exists_raw, index=False)

        # 2. Process for profitable trades
        profitable_trades = []
        signals_df = pd.DataFrame(signals).sort_values(by=['ticker', 'Date'])
        
        for ticker, ticker_signals in signals_df.groupby('ticker'):
            last_buy_signal = None
            for index, row in ticker_signals.iterrows():
                if row['signal'] == 'buy':
                    last_buy_signal = row
                elif row['signal'] == 'sell' and last_buy_signal is not None:
                    # Found a trade pair
                    buy_date = pd.to_datetime(last_buy_signal['Date'])
                    sell_date = pd.to_datetime(row['Date'])
                    price_at_buy = last_buy_signal['Close']
                    price_at_sell = row['Close']

                    # Stock return
                    return_value = price_at_sell - price_at_buy
                    return_pct = (return_value / price_at_buy) * 100

                    # Benchmark return
                    benchmark_buy_price = benchmark_data[benchmark_data['Date'] == buy_date]['Close'].iloc[0]
                    benchmark_sell_price = benchmark_data[benchmark_data['Date'] == sell_date]['Close'].iloc[0]
                    benchmark_return_value = benchmark_sell_price - benchmark_buy_price
                    benchmark_return_pct = (benchmark_return_value / benchmark_buy_price) * 100

                    if return_pct > benchmark_return_pct:
                        profitable_trades.append({
                            'Ticker': ticker,
                            'buy_date': buy_date,
                            'price_at_buy': price_at_buy,
                            'sell_date': sell_date,
                            'price_at_sell': price_at_sell,
                            'return_value': return_value,
                            'return_pct': return_pct,
                            'NSDAQ100etf_buy_date': buy_date,
                            'NSDAQ100etf_price_at_buy': benchmark_buy_price,
                            'NSDAQ100etf_sell_date': sell_date,
                            'NSDAQ100etf_price_at_sell': benchmark_sell_price,
                            'NSDAQ100etf_return_value': benchmark_return_value,
                            'NSDAQ100etf_return_pct': benchmark_return_pct
                        })
                    
                    last_buy_signal = None # Reset for next trade

        if profitable_trades:
            df_profitable = pd.DataFrame(profitable_trades)
            file_exists_profitable = os.path.exists('profitable_trades.csv')
            df_profitable.to_csv('profitable_trades.csv', mode='a', header=not file_exists_profitable, index=False)

        return "Signals saved and profitable trades analyzed.", []
    return "", dash.no_update

if __name__ == '__main__':
    app.run(debug=True)
