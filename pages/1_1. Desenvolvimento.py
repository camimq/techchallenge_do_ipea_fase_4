import streamlit as st
import pandas as pd
import numpy as np
import sys

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import streamlit as st


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model

def get_data():
    start_date = '2022-01-01'
    end_date = '2024-12-31'
    df = yf.download('BZ=F', start=start_date, end=end_date)
    df = df[['Close']]  
    df.reset_index(inplace=True)  
    alpha = 0.09  
    df['Smoothed_Close'] = df['Close'].ewm(alpha=alpha, adjust=False).mean()
    return df


def plot_serie_suavizada(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label='Original', color='blue')
    ax.plot(df.index, df['Smoothed_Close'], label=f'Suavizado (alpha=0.9)', color='red')
    ax.set_title('Série Temporal Suavizada usando Suavização Exponencial')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço de Fechamento')
    ax.legend()
    st.pyplot(fig)

def train_model(df):
    st.write("Treinamento do modelo iniciado...")
    #f.drop(columns=['Close'], inplace=True)
    close_data = df['Smoothed_Close'].values
    close_data = close_data.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(close_data)
    close_data = scaler.transform(close_data)
    split_percent = 0.80
    split = int(split_percent*len(close_data))
    close_train = close_data[:split]
    close_test = close_data[split:]
    date_train = df['Date'][:split]
    date_test = df['Date'][split:]
    look_back = 10
    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)
    np.random.seed(7)
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(look_back,1)))
    model.add(Dense(1)),
    model.compile(optimizer="adam", loss="mse", metrics=[MeanSquaredError()])
    num_epochs = 20
    model.fit(train_generator, epochs=num_epochs, verbose=1)

    mse = model.evaluate(test_generator, verbose=1)
    # 1. Fazer previsões usando o conjunto de teste
    test_predictions = model.predict(test_generator)

    # 2. Inverter qualquer transformação aplicada aos dados
    test_predictions_inv = scaler.inverse_transform(test_predictions.reshape(-1, 1))
    test_actuals_inv = scaler.inverse_transform(np.array(close_test).reshape(-1, 1))

    # Ajuste as dimensões
    test_actuals_inv = test_actuals_inv[:len(test_predictions_inv)]

    # Calcular o MAPE
    mape = np.mean(np.abs((test_actuals_inv - test_predictions_inv) / test_actuals_inv)) * 100
    rmse_value = np.sqrt(mse[0])
    return close_data, close_test, close_train, date_test, date_train, mse, mape, rmse_value, train_generator, test_generator, model, scaler

def get_features(df, model):
    #df.drop(columns=['Close'], inplace=True)
    close_data = df['Smoothed_Close'].values
    close_data = close_data.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(close_data)
    close_data = scaler.transform(close_data)
    split_percent = 0.80
    split = int(split_percent*len(close_data))
    close_train = close_data[:split]
    close_test = close_data[split:]
    date_train = df['Date'][:split]
    date_test = df['Date'][split:]
    num_prediction = 14 #definição dos próximos dias
    lock_back = 10
    forecast = predict(num_prediction, model, lock_back) #resultado de novos dias
    forecast_dates = predict_dates(num_prediction)
    return close_data, close_test, close_train, date_test, date_train, forecast_dates, forecast, scaler
   
def exibir_metricas(mse, mape, rmse_value):
    st.write(f"MSE: {mse[0]}")
    st.write(f"MAPE: {mape:.2f}%")
    st.write(f"RMSE: {rmse_value}")

def plot_prediction(date_train, date_test, close_train, close_test, model, test_generator):
    prediction = model.predict(test_generator)
    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))

    trace1 = go.Scatter(
        x = date_train,
        y = close_train,
        mode = 'lines',
        name = 'Data'
    )
    trace2 = go.Scatter(
        x = date_test,
        y = prediction,
        mode = 'lines',
        name = 'Prediction'
    )
    trace3 = go.Scatter(
        x = date_test,
        y = close_test,
        mode='lines',
        name = 'Ground Truth'
    )
    layout = go.Layout(
        title = "Brent Crude Oil Price Prediction",
        xaxis = {'title' : "Data"},
        yaxis = {'title' : "Fechamento"}
    )
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    st.plotly_chart(fig)

def predict(num_prediction, model, look_back):
    prediction_list = close_data[-look_back:]

    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]

    return prediction_list

# Função para gerar as datas dos próximos 'num_prediction' dias
# Assume que o DataFrame 'df' possui uma coluna 'Date' contendo as datas
def predict_dates(num_prediction):
    last_date = df['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

def plot_forecast(date_test,close_test,forecast_dates,forecast):
    close_test = close_test.reshape((-1))
    trace1 = go.Scatter(
    x = date_test,
    y = close_test,
    mode = 'lines',
    name = 'Data'
    )
    trace2 = go.Scatter(
        x = forecast_dates,
        y = forecast,
        mode = 'lines',
        name = 'Prediction'
    )
    layout = go.Layout(
        title = "Forecast Brent Crude Oil Price",
        xaxis = {'title' : "Data"},
        yaxis = {'title' : "Fechamento"}
        )
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    st.plotly_chart(fig)
   
def teste(df, forecast_dates, forecast, scaler):

    forecast = forecast.reshape(-1, 1) #reshape para array
    forecast = scaler.inverse_transform(forecast)
    df = pd.DataFrame(df)
    df_past = df[['Date','Smoothed_Close']]
    df_past['Date'] = pd.to_datetime(df_past['Date'])                          #configurando para datatime
    df_past['Forecast'] = np.nan                                               #Preenchendo com NAs
    df_past['Forecast'].iloc[-1] = df_past['Smoothed_Close'].iloc[-1]
    df_future = pd.DataFrame(columns=['Date', 'Smoothed_Close', 'Forecast'])
    df_future['Date'] = forecast_dates
    df_future['Forecast'] = forecast.flatten()
    df_future['Smoothed_Close'] = np.nan
    frames = [df_past, df_future]
    results = pd.concat(frames, ignore_index=True).set_index('Date')
    last_15_forecasts  = results.drop(columns=['Smoothed_Close'])
    last_15_forecasts  = last_15_forecasts .dropna()

    # Selecionando os últimos 14 valores da coluna 'Forecast'
    last_15_forecasts = results.tail(15)

    # Transpondo o DataFrame
    last_15_forecasts_transposed = last_15_forecasts.T
    st.write("Últimos 14 Valores da Coluna 'Forecast':")
    st.write(last_15_forecasts_transposed)
    return results

def plot_predict(df):
    results2024 =  df.loc['2024-01-01':]
    plot_data = [
        go.Scatter(
            x=results2024.index,
            y=results2024['Smoothed_Close'],
            name='actual'
        ),
        go.Scatter(
            x=results2024.index,
            y=results2024['Forecast'],
            name='prediction'
        )
    ]

    plot_layout = go.Layout(
            title='Forecast Brent '
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

# Configura atributos da página
st.set_page_config(page_title='Desenvolvimento', page_icon='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTADYDWw-Wdk19SHqSlthiAilmWGH6NfW5FVg&s', layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title('1. Desenvolvimento')
st.write('''
          Nesta seção, serão apresentados os detalhes do desenvolvimento do projeto, desde a coleta dos dados, até a implementação do modelo de _Machine Learning_.
          
          **O desenvolvimento do projeto foi dividido em 4 etapas:**
          
          1. **Sobre os Dados**
          2. **Análise Exploratória de Dados**
          3. **Dashboard**
          4. **Modelo de Machine Learning**
          5. **Deploy do Projeto** 
         ''')


#------------------------VARIAVEIS GLOBAIS------------------------------------#


#df = get_data()
#try:
#    model = load_model("meu_modelo.h5")
#except:
#    st.write('não foi possivel carregar o modelo')
#    close_data, close_test, close_train, date_test, date_train, mse, mape, rmse_value, train_generator, test_generator, model = train_model(df)
#"""

look_back = 10

# Configura tabs da página
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Sore os Dados', 'Análise Exploratória de Dados', 'Dashboard', 'Modelo de Machine Learning', 'Deploy do Projeto'])

with tab1:
  st.title('Coleta de Dados')
  col1, col2 = st.columns(2)
  with col1:
    # Inserção da imagem da home
    st.image('img/tabela_base_dados.png', caption=None, width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
  
  with col2:
    st.write('''
              A base utilizada para análise e desenvolvimento do modelo de ML, foram coletadas diretamenta do [_site_ do IPEA](https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis).
            
              Para que seja possível uma análise assertiva e que o mode de _Machine Learning_ seja eficiente, é necessário garantir uma boa quantidade de dados históricos, para que seja possível identificar padrões e tendências e treinar o modelo de forma adequada.
            
              Por essa razão, ficou definido que a base deve ter dados referentes aos últimos 5 anos fechados, ou seja, de 2019 à 2023. 
            
              ## Método de coleta dos dados
              Para a coleta dos dados disponibilizados dentro da base do IPEA, fizemos o _download_ de todos os arquivos (total de 10 planilhas) e realizamos a junção de todas as planilhas em um único _dataset_,
           ''')

  st.write('''
            utilizando **PostgreSQL** e **Knime**, conforme passo a passo abaixo descrito abaixo:
            
            **1. Download dos arquivos:**
            
            Dentro do [_site_ do IPEA](https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis), as bases estão separadas por semestre, desta forma, cada ano, possui dois arquivos; um para o primeiro semestre e outro para o segundo semestre, disponibilizadas em formato `.csv`.
          ''')
  st.image('img/download_base_dados.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  st.write('''
          **2. Concatenação dos arquivos com Knime:**
          
          **3. Importação dos arquivos para o PostgreSQL:**
          
          
          
          
          
          
          ''')


with tab2:
  st.title('Análise Exploratória de Dados')
    
  
with tab3:
  st.title('Dashboard')

with tab4:
  st.title('Modelo de Machine Learning')
  df = get_data()
  st.write('Serie Suavizada')
  plot_serie_suavizada(df)
  st.title('Treinamento do Modelo')

  # Botão para iniciar o treinamento
  if st.button('Iniciar Treinamento'):
    close_data, close_test, close_train, date_test, date_train, mse, mape, rmse_value, train_generator, test_generator, model, scaler = train_model(df)
    exibir_metricas(mse, mape, rmse_value)
    st.title('Gráfico de Previsão do Preço do Petróleo')
    plot_prediction(date_train, date_test, close_train, close_test, model, test_generator)
    #model.save('meu_modelo.h5')
    close_data, close_test, close_train, date_test, date_train, forecast_dates, forecast, scaler  = get_features(df, model)
    #plot_forecast(date_test,close_test,forecast_dates,forecast)
    df_results = teste(df, forecast_dates, forecast, scaler)
    plot_predict(df_results)





with tab5:
  st.title('Deploy do Projeto')
  #plot_prediction(date_train, date_test, close_train, close_test, model, test_generator)
  #model_path = '../meu_modelo.h5'  # Subindo um nível para acessar o arquivo do modelo
  #model = load_model(model_path)
  #try:
  #  close_data, close_test, close_train, date_test, date_train, forecast_dates, forecast  = get_features(df, model)
  #  plot_forecast(date_test,close_data,forecast_dates,forecast)
  #except:
  #  st.title('O modelo ainda não foi treinando, vá até a aba de modelo e faça o treinamento')