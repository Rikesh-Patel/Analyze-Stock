import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import yfinance as yf
from datetime import datetime as dt
import requests


st.set_page_config(page_title='Comparative Stock Analysis')
st.set_option('deprecation.showPyplotGlobalUse', False)
mpl.rcParams['lines.color'] = 'r'
st.markdown("""
        <h1 style='text-align: center; color: #FFFFFF; margin-bottom: -30px;'>
      Comparative Stock Analyzer
        </h1>
    """, unsafe_allow_html=True
    )
st.caption("""
        <p style='text-align: center; color: #FFFFFF;'>
        by <a href='https://www.rikeshpatel.io/'>Rikesh Patel</a>
        </p>
    """, unsafe_allow_html=True
    )
st.markdown("""This app retrieves price data from multiple databases and visualizes it in real time, 
so you don't have to!""")


# Sidebar Controls
st.sidebar.write('User Input Features')
start_time = st.sidebar.date_input('Start Date', dt(2020,1,1))
end_time = st.sidebar.date_input('End Date', dt.now())
Interval = st.sidebar.selectbox('Interval', ["1d", "5d", "1wk", "1mo", "3mo", "1y"])


# Web scrape S&P500 data and cache it for efficiency
@st.cache
def load_data():
    #Load list of all S&P 500 stock symbols, sorted by market cap
    df_sp500_raw = pd.read_html('https://stockmarketmba.com/stocksinthesp500.php')[0]
    df_sp500 = df_sp500_raw.copy()
    df_sp500['Market cap'] = df_sp500['Market cap'].replace('[\$,]', '', regex=True).astype(int)
    df_sp500 = df_sp500.sort_values('Market cap', ascending=False)[1:]['Symbol']
    #Load list of top 100 ETF ticker symbols
    df_etf_raw = pd.read_html('https://etfdb.com/compare/market-cap/')[0]
    df_etf = df_etf_raw['Symbol']
    df_raw = pd.concat([df_sp500_raw, df_etf_raw])
    df = df_sp500.append(df_etf)
    return df, df_sp500, df_etf, df_raw

df, df_sp500, df_etf, df_raw = load_data()

# Sidebar+ - Stock selection
selected_stock= st.sidebar.multiselect('Select Stock(s)', df_sp500.unique(), None)
selected_index = st.sidebar.multiselect('Select Indices', df_etf.unique(), None)
try:
    selected_unique = st.sidebar.text_input('Type Ticker Symbols (Ex: AAPL, MSFT, SPY)', max_chars=100).strip(", ").upper().split(",")
    selected_unique = [s.strip(' ') for s in selected_unique]
    selected = selected_stock + selected_index + selected_unique
except:
    selected = selected_stock + selected_index

# Filtering data
df_selected = df_raw[df_raw['Symbol'].isin(selected)]






import time
from plotly import graph_objs as go

def f_plot_selected(df, symbols, start_index, end_index):
    # plot dataframe's columns by index in the range given
    f_plot_data(df.loc[start_index:end_index], symbols)        

def f_get_data(symbols, start_index, end_index, interval):
    # Query stock data from YFinance
    dates = pd.date_range(start_index, end_index)
    df = pd.DataFrame(index=dates)
    
    if 'SPY' not in symbols:  # add SPY for comparative reference, if absent
        symbols.append('SPY')

    for symbol in symbols:
        connected = False
        while not connected:
            try:
                ticker_df = yf.download(symbol, start=start_index, end=end_index, interval=interval)
                connected = True
            except Exception as e:
                print("Error: " + str(e))
                time.sleep( 10 )
                pass   
        
        # clean data and reconstruct in desired format to plot
        ticker_df = ticker_df.reset_index()
        ticker_df.set_index('Date', inplace=True, drop=False) 
        df_temp = ticker_df[['Date','Adj Close']]
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp[symbol])
        
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
            
    return df

def f_get_all_data(symbols, start_index, end_index):
    # Query stock data
    dates = pd.date_range(start_index, end_index)
    df = pd.DataFrame(index=dates) 

    connected = False
    while not connected:
        try:
            ticker_df = yf.download(symbols, start=start_index, end=end_index)
            if not ticker_df:
                ticker_df = yf.download('SPY', start=start_index, end=end_index)
            connected = True
        except Exception as e:
            print("Error: " + str(e))
            time.sleep( 10 )
            pass   
    
    ticker_df = ticker_df.reset_index()
    ticker_df.set_index('Date', inplace=True, drop=False) 
    df = ticker_df.dropna()
    return df

def f_normalize_data(df):
    # normalizes stock data in respect to price in day 1
    return df/df.iloc[0,:]    
    
def f_plot_data(df, symbols):
    # plot details
    fig = go.Figure()
    fig.layout.update(
    xaxis_rangeslider_visible=True,
    title = "ㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤRelative Stock Price Growth",
    xaxis_title="Time",
    yaxis_title="Price",
    yaxis_tickformat = '$.2f',
    xaxis={
        "range": [df.index.min(), df.index.max()],
        "rangeslider": {"visible": True},
        "rangeselector": 
        dict(
            buttons= 
            list([
                dict(count=1,
                     label="1M",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6M",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1Y",
                     step="year",
                     stepmode="backward"),
                dict(label="All",
                     step="all")
            ]), 
            font = dict(color='dimgrey')), 
            "type" : "date"
        }, 
    hovermode="x unified")
    for symbol in symbols:
        fig.add_scatter(x=df.index, y=df[symbol], name=symbol)
    
    st.plotly_chart(fig)

    

def f_run():
    # Combine all the steps from retrieving data to plotting        
    symbols = list(df_selected.Symbol)
    df = f_get_data(symbols, start_time, end_time, Interval) #start_time
    df = f_normalize_data(df)
    f_plot_selected(df, list(symbols), start_time, end_time)
    return df

df = f_run()





from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import plotly.express as px







##Prediction

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 
	df = DataFrame(data)
	cols, names = list(), list()

	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# combine
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop NaN value rows
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
 
# Train and test sets for supervised learning
def prepare_data(data, n_test, n_lag, n_seq):
	# transform data to be stationary
	diff_values = data.diff().values
	diff_values = diff_values.reshape(len(diff_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	# split into train and test sets
	train, test = supervised.values[:-n_test], supervised.values[-n_test:]
	return scaler, train, test
 
# fit LSTM neural network to training data
def fit_lstm(train, n_lag, n_batch, nb_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
	X, y = train[:, 0:n_lag], train[:, n_lag:]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	
	# build LSTM network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
		model.reset_states()
	return model
 
# make a forecast with network,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]
 
# evaluate the LSTM model
def make_forecasts(model, n_batch, train, test, n_lag):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts
 
# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted
 
# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = np.array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted
 
# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		st.write('t+%d RMSE: %f' % ((i+1), rmse))
 
# final plot
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset
    fig = px.line(series, x=series.index, y=series).update_layout(
    xaxis_title="Time",
    yaxis_title="Price",
    yaxis_tickformat = '$.2f',
    xaxis={
        "range": [series.index.min(), series.index.max()],
        # add range slider and buttons for zooming
        "rangeslider": {"visible": True},
        "rangeselector": dict(
            buttons= list([
                dict(count=1,
                     label="1M",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6M",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1Y",
                     step="year",
                     stepmode="backward"),
                dict(label="All",
                     step="all")
            ]), 
            font = dict(color='dimgrey')), 
            "type" : "date"}, 
    hovermode="x unified")
    fig.update_traces(hovertemplate=selected_ticker+" : %{y}")
    
    # plot moving average line in orange
    ma100 = series.rolling(100).mean() 
    fig.add_trace(go.Line(x=series.index, y=ma100, name='MA100')) 

    # plot the forecasts in red
    for i in range(len(forecasts)):
            #pl= st.empty()
            off_s = len(series) - n_test + i - 1
            off_e = off_s + len(forecasts[i]) + 1
            xaxis = [x for x in range(off_s, off_e)]
            xaxis1 = series[xaxis].index
            yaxis = [series.values[off_s]] + forecasts[i]
            fig.add_trace(go.Scatter(x= xaxis1, y=yaxis, line_color="#ff0000", mode='lines', showlegend=False, hoverinfo='skip'))
            if i == len(forecasts)-1:
                 fig.add_trace(go.Scatter(x= xaxis1, y=yaxis, line_color="#ff0000", mode='lines', name='Forecast', hoverinfo='skip'))

	# show the plot
    st.plotly_chart(fig)








# LSTM network configuration for model
n_lag = 2
n_seq = 3
n_test = 50 #12
n_epochs = 1#10 #10
n_batch = 1
n_neurons = 1

# Grab ticker from user
ticker_input = st.text_input(label= 'Input a Stock Ticker', max_chars=10, value='SPY').rstrip(", ").upper().split(", ")[0]

# Default ticker SPY
if not ticker_input:
    selected_ticker = 'SPY'
else:
    selected_ticker = ticker_input


# API call for stock description    
@st.cache
def news():
	api = 'Im0OD1nRQC58yzafY7pLwop5717DssjJ'
	today = str(dt.now())
	today = today[:today.find(" ")]
	ticker = selected_ticker

	api_url = f'https://api.polygon.io/v3/reference/tickers/{ticker}?date={today}&apiKey={api}'
	news_raw = requests.get(api_url).json()
	try:
	    news = news_raw['results'].get('description') 
	    if news == 'None':
	        pass
	    else:       
	    	return news
	except:
	    pass
	
    if not news:
    else: 
	st.markdown(news())

if st.button('Predict'):
        
    # prepare data
    prediction_data = f_get_data([selected_ticker], "2019-01-01", today, "1d")[selected_ticker]
    scaler, train, test = prepare_data(prediction_data, n_test, n_lag, n_seq)
    # fit model
    model = fit_lstm(train, n_lag, n_batch, n_epochs, n_neurons)
    # make forecasts
    forecasts = make_forecasts(model, n_batch, train, test, n_lag)
    # inverse transform forecasts and test
    forecasts = inverse_transform(prediction_data, forecasts, scaler, n_test+2)
    actual = [row[n_lag:] for row in test]
    actual = inverse_transform(prediction_data, actual, scaler, n_test+2)
    # evaluate forecasts

    print(time.perf_counter())
    # plot forecasts
    st.markdown("<h1 style='text-align: center; font-size:18px;'>" + selected_ticker + " LSTM Model </h1>", unsafe_allow_html=True)
    plot_forecasts(prediction_data, forecasts, n_test+2)
    evaluate_forecasts(actual, forecasts, n_lag, n_seq)

    st.markdown("""
	     Author's Note: <br>
	     The Long Short-Term Memory neural network has an ongoing list of  pros and cons. For instance, they can be fine
	     tuned along parameters, but this process is time-intensive and hyper-focused. In terms of efficiency, we can conclude that the LSTM network's
	     performance fails to meet expectations when compared to other relevant models, like AR and ARIMA that better fit modeling stock price
	     time series data. Ergo, LSTM is inferior in terms of extrapolating ability and requiring large and consistent data.""")

st.markdown("""* **Data source :** [Stock Market MBA](https://stockmarketmba.com/stocksinthesp500.php), [Polygon API](https://api.polygon.io/), [ETF Database](https://etfdb.com/compare/market-cap/)
""")


# Downloadable CSV data
def filedownload(df):
    csv = df_selected.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="Security_Details.csv">Download CSV File</a>'
    return href

st.sidebar.markdown(filedownload(df_selected), unsafe_allow_html=True)
