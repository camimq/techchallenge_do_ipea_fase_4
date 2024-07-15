# Importante:
# A documentação das funções presentes neste arquivo estão no README.md

# Início das importações de bibliotecas e dependências para rodar o projeto
import streamlit as st
import pandas as pd
import pandas.io.sql as sqlio
import numpy as np
import sys
import seaborn as sns

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from tensorflow.keras.metrics import MeanSquaredError # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.models import load_model # type: ignore
# Fim das importações de bibliotecas e dependências para rodar o projeto

# Início das definições de funções que serão utilizadas no projeto

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

# Fim das definições de funções que serão utilizadas no projeto

### Início da Análise exploratória de dados

# cria o DataFrame
df_anp = pd.read_csv (r'c:/Users/Elitebook/OneDrive/Desktop/web dev/techchallenge_do_ipea_fase_4/bases/df_anp.csv')

df_anp_revenda = pd.read_csv (r'c:\Users\Elitebook\OneDrive\Desktop\web dev\techchallenge_do_ipea_fase_4\bases\df_anp_revenda.csv')

# Valores mínimos, máximos e médios dos produtos por ano
df_anp_valor = df_anp[['ano', 'produto', 'valor_venda']].groupby(['produto', 'ano']).agg(['min', 'max', 'mean']).round(2)

# Valores mínimos, máximos e médios dos produtos por ano - recorte por Estado
df_anp_valor_estado = df_anp[['ano', 'estado', 'produto', 'valor_venda']].groupby(['produto', 'ano', 'estado']).agg(['min', 'max', 'mean']).round(2)

# separando o dataset por tipo de produto
gasolina_aditivada = df_anp[df_anp['produto'] == 'GASOLINA ADITIVADA']
gasolina = df_anp[df_anp['produto'] == 'GASOLINA']
diesel_s10 = df_anp[df_anp['produto'] == 'DIESEL S10']
diesel = df_anp[df_anp['produto'] == 'DIESEL']
etanol = df_anp[df_anp['produto'] == 'ETANOL']
gnv = df_anp[df_anp['produto'] == 'GNV']

# funação para auxiliar na criação de gráficos - Análise exploratória de dados 
def plotar_boxplot_geral(y, dataset):
  fig = plt.figure(figsize=(12, 6))
  ax = sns.boxplot(y=y, data=dataset)
  ax.figure.set_size_inches(4, 4)
  plt.show()

### Fim da Análise Exploratória de Dados

# Configuração dos atributos da página
st.set_page_config(page_title='Desenvolvimento', page_icon='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTADYDWw-Wdk19SHqSlthiAilmWGH6NfW5FVg&s', layout="centered", initial_sidebar_state="auto", menu_items=None)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Início do conteúdo da página
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Sobre os Dados', 'Análise Exploratória de Dados', 'Dashboard', 'Modelo de Machine Learning', 'Deploy do Projeto'])

# Conteúdo da tab Sobre Dados
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
            utilizando **PostgreSQL** e **Knime**, conforme passo a passo descrito abaixo:
            
            **1. Download dos arquivos:**
            
            Dentro do [_site_ do IPEA](https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis), as bases estão separadas por semestre, desta forma, cada ano, possui dois arquivos; um para o primeiro semestre e outro para o segundo semestre, disponibilizadas em formato `.csv`.
          ''')
    st.image('img/download_base_dados.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.write('''
          **2. Concatenação dos arquivos com Knime:**
          
            Os arquivos foram transportados para dentro do Knime, onde foi feita a concatenação dos arquivos, a exclusão da coluna `valor_compra` e a exportação do arquivo final em formato `.csv`.
        ''')
    st.image('img/knime_1.png', caption="Print do schema da concatenação das tabelas no Knime.", width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.write('''
        **3. Importação dos arquivos para o PostgreSQL:**
        
        O arquivo final foi importado para o PostgreSQL, onde foi feita a criação de uma tabela para armazenar os dados. A tabela resultado deste processo, é a que será utilizada para a análise exploratória e treinamento do modelo de _Machine Learning_.
        ''')
    st.image('img/postgres.png', caption="Print da tabela de dados exportada para o Postgres.", width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.write('---')

# Conteúdo da tab Análise Exploratória de Dados         
with tab2:
  st.write('''
           À partir da bases de dados coletada, é possível determinar algumas informações relevantes para a análise exploratória de dados, conforme descrito abaixo:
           
           - **Produtos listados na base:** Diesel, Diesel S10, Etanol, Gasolina, Gasolina Aditivada e GNV.
           - **Período dos dados:** 2019 à 2023.
           ''')
  st.write('## Valores mínimos, máximos e médios dos produtos por ano')
  st.dataframe(df_anp_valor)
  st.write('## Valores mínimos, máximos e médios dos produtos por ano (recorte por Estado)')
  st.dataframe(df_anp_valor_estado)
  st.write('## Estatística básica de cada produto')
  
  col1, col2 = st.columns(2)
  col3, col4 = st.columns(2)
  col5, col6 = st.columns(2)
  col7, col8 = st.columns(2)
  col9, col10 = st.columns(2)
  col11, col12 = st.columns(2)
       
  with col1:
   st.write('### Gasolina Aditivada')
   st.dataframe(gasolina_aditivada.valor_venda.describe().round(2))
  with col2:
   # variável para plotar box 
   fig_boxplot_gasolina_aditivada = px.box(gasolina_aditivada, y='valor_venda', labels={'valor_venda': 'Valor de Venda'})
   st.plotly_chart(fig_boxplot_gasolina_aditivada)
  with col3:
   st.write('### Gasolina')
   st.dataframe(gasolina.valor_venda.describe().round(2))
  with col4:
   # variável para plotar box 
   fig_boxplot_gasolina = px.box(gasolina, y='valor_venda', labels={'valor_venda': 'Valor de Venda'})
   st.plotly_chart(fig_boxplot_gasolina)  
  with col5:
   st.write('### Diesel S10')
   st.dataframe(diesel_s10.valor_venda.describe().round(2))
  with col6:  
   # variável para plotar box 
   fig_boxplot_diesel_s10 = px.box(diesel_s10, y='valor_venda', labels={'valor_venda': 'Valor de Venda'})
   st.plotly_chart(fig_boxplot_diesel_s10)
  with col7:
   st.write('### Diesel')
   st.dataframe(diesel.valor_venda.describe().round(2))
  with col8:
   # variável para plotar box 
   fig_boxplot_diesel = px.box(diesel, y='valor_venda', labels={'valor_venda': 'Valor de Venda'})
   st.plotly_chart(fig_boxplot_diesel)
  with col9:
   st.write('### Etanol')
   st.dataframe(etanol.valor_venda.describe().round(2))
  with col10:
   # variável para plotar box x
   fig_boxplot_etanol = px.box(etanol, y='valor_venda', labels={'valor_venda': 'Valor de Venda'})
   st.plotly_chart(fig_boxplot_etanol) 
  with col11:
   st.write('### GNV')
   st.dataframe(gnv.valor_venda.describe().round(2))
  with col12: 
   # variável para plotar box 
   fig_boxplot_gnv = px.box(gnv, y='valor_venda', labels={'valor_venda': 'Valor de Venda'})
   st.plotly_chart(fig_boxplot_gnv)
  
  st.write('## Desempenho dos produtos por Estado')
  st.write('### Gasolina Aditivada')
  fig_boxplot_gasolina_aditivada_por_estado = px.box(gasolina_aditivada, x='estado', y='valor_venda', labels={'valor_venda': 'Valor de Venda por Estado'}) 
  st.plotly_chart(fig_boxplot_gasolina_aditivada_por_estado)
  
  st.write('### Gasolina')
  fig_boxplot_gasolina_por_estado = px.box(gasolina, x='estado', y='valor_venda', labels={'valor_venda': 'Valor de Venda por Estado'}) 
  st.plotly_chart(fig_boxplot_gasolina_por_estado)
  
  st.write('### Diesel S10')
  fig_boxplot_diesel_s10_por_estado = px.box(diesel_s10, x='estado', y='valor_venda', labels={'valor_venda': 'Valor de Venda por Estado'}) 
  st.plotly_chart(fig_boxplot_diesel_s10_por_estado)
  
  st.write('### Diesel')
  fig_boxplot_diesel_por_estado = px.box(diesel, x='estado', y='valor_venda', labels={'valor_venda': 'Valor de Venda por Estado'}) 
  st.plotly_chart(fig_boxplot_diesel_por_estado)  

  st.write('### Etanol')
  fig_boxplot_etanol_por_estado = px.box(etanol, x='estado', y='valor_venda', labels={'valor_venda': 'Valor de Venda por Estado'}) 
  st.plotly_chart(fig_boxplot_etanol_por_estado) 
  
  st.write('### GNV')
  fig_boxplot_gnv_por_estado = px.box(gnv, x='estado', y='valor_venda', labels={'valor_venda': 'Valor de Venda  por Estado'}) 
  st.plotly_chart(fig_boxplot_gnv_por_estado)

# Conteúdo da tab Dashboard 
with tab3:
  st.write('''
            De acordo com as informações coletadas na análise Exploratória de Dados, definimos 6 _insights_ relevantes que auxiliam uma a tomada de decisão de negócio em um cenário de variação de preços do petróleo.

            ## Marcadores econômicos importantes no período de 2019 à 2023         
           ''')
  
  fig_variacao_precos = px.line(df_anp, x='ano', y='valor_venda', color='produto', title='Variação de Preços de Combustíveis de 2019 à 2023', labels={'valor_venda': 'Valor de Venda (R$)', 'ano': 'Ano', 'produto': 'Produto'})
  st.plotly_chart(fig_variacao_precos)
  
  st.write('''
            De acordo com o gráfico acima, é possível observar que, de maneira geral, os preços dos combustíveis que vinham, entre 2019 e 2020, relativamente estabilizados, à partir da última parte de 2020, começa a subir uma subida vertiginosa até o início de 2023.
            
            No início de 2020, iniciou-se o período de pandemia do COVID-19, que impactou diretamente na economia mundial, e consequentemente, no preço do petróleo. Mas, no contexto da pandemia, a demanda por combustível caiu uma vez que as pessoas viviam as restrições de mobilidade que este período trouxe. A partir de 2021, com a retomada da economia e uma demanda reprimida "liberada" para rodar o mercado,  o preço do petróleo voltou a subir.
            
            Além da demanda pós-demanda, mais um complicador entrou em cena: em fevereiro de 2022, a Russia invade a Ucrania.
            
            [A Russia representa 40% das importações de gás natural da União Europeia](https://www.bbc.com/portuguese/internacional-60673879). Para ajudar, o período da invasão coincidiu com o inverno na Europa; por isso, imediatamente, diversos países europeus passaram a ter problemas imediatos com aquecimento das casas, já que, uma sanção pesada foi imposta à Russia que deixou de exportar o seu gás para diversos países do lado ocidental do mundo, especialmente Europa.
            
            Todo este embrólio, fez com que, em 2023, os combustíveis atingissem o pico de valorização. Um cenário que até os dias atuais não se resolveu por completo.
            
            ## Produtos mais vendidos de 2019 à 2023
            
            No período de 2019 à 2023, o produto com maior demanda é a gasolina. Responsável pelos maiores números de venda em litragem e em valores monetários.
            
            **❗ Importante:** A gasolina é seguida pelo diesel em valores negociados, contudo, o Etanol (em volume) é quem segue na segunda posição; com o Diesel aparecendo em terceiro lugar.
           ''')
  
  col1, col2 = st.columns(2)
  
  with col1:
      produtos_contagem = df_anp['produto'].value_counts()
      fig_produtos_mais_vendidos_litros = px.bar(x=produtos_contagem.values, y=produtos_contagem.index, title = 'Produtos mais vendidos de 2019 à 2023 (litros)', labels={'x': 'Quantidade vendida (litros)', 'y': 'Produto'}, color = produtos_contagem.index, color_continuous_scale=px.colors.sequential.Viridis)
      
      st.plotly_chart(fig_produtos_mais_vendidos_litros)
  with col2: 
   produtos_valor = df_anp.groupby('produto')['valor_venda'].sum().sort_values(ascending=False)
   fig_produtos_mais_vendidos_valor = px.bar(x=produtos_valor.values, y=produtos_valor.index, title = 'Produtos mais vendidos de 2019 à 2023 (R$)', labels={'x': 'Valor vendido (R$)', 'y': 'Produto'}, color = produtos_valor.index, color_continuous_scale=px.colors.sequential.Viridis)
   
   st.plotly_chart(fig_produtos_mais_vendidos_valor)
  
  st.write('''
            ## Volume de negociação
            
            Como é de se esperar, a região sudeste, dada sua alta concentração populacional, região que produz grande parte da riqueza do país e ser a região onde estão localizadas as principais capitais econômicas; é a região que mais consome combustíveis e isso pode ser mostrado, através do volume monetário negociado no período.
            
            ** ❗ O nordeste, embora não seja uma região economicamente tão relevante para o país, em termos de consumo, se coloca em segundo lugar no volume de negociado.**
           ''')
  
  regiao_valor = df_anp.groupby('regiao')['valor_venda'].sum().sort_values(ascending=False)
  
  fig_valores_por_regiao = px.bar(x=regiao_valor.values, y=regiao_valor.index, title = 'Volume negociado por Região de 2019 à 2023 (R$)', labels={'x': 'Valor vendido (R$)', 'y': 'Região'}, color = regiao_valor.index, color_continuous_scale=px.colors.sequential.Viridis)
  
  st.plotly_chart(fig_valores_por_regiao)
  
  st.write('''
            Nas tendências de variações anuais das vendas, é possível observar diferentes períodos de altas e baixas, com destaque principal para a gasolina, como o produto mais comercionalizado e **o etanol em sexto lugar**, como o produto menos vendido no período de 2019 à 2023, em termos de valores.
            
            ** ❗ Embora o Etanol tenha bom desempenho em vendas, é o produto que indica trazer menor margem de receita**. 
           ''')
  venda_historica = df_anp.pivot_table(values='valor_venda', index='produto', columns='ano', aggfunc='sum', fill_value=0)
  
  fig_venda_historica_por_produto = px.imshow(venda_historica, x = ['2019', '2020', '2021', '2022', '2023'], y =  ['DIESEL', 'DIESEL S10', 'ETANOL', 'GASOLINA', 'GASOLINA ADITIVADA', 'GNV'], labels = dict(x='Ano', y='Produto', color='Valor de Venda (R$)'), color_continuous_scale=px.colors.sequential.Viridis, aspect = 'auto', text_auto='True', title = 'Venda Histórica por Produto')
  
  st.write('''
            ## Top 10 Revendas 2019 à 2023
            
            Os TOP 10 nacional de revendedora, tiveram foco em estratégia de mercado, negócio e análise de desempenho durante os anos monitorados para que pudessem ser líderes em seus mercados.
            
            **❗ A prova de que esse tipo de foco é importante, está no líder do _chart_; a [Rede Sim](https://www.simrede.com.br/nossos-negocios), é uma empresa que existe a 38 anos, fundada em Flores da Cunha, no Rio Grande do Sul. Hoje, é a maior rede de postos de combustíveis do país, ainda que, de acordo com os dados, a região Sul do país, ocupe o terceiro lugar em volume monetário negociado.**           
           ''')

# Conteúdo da tab Modelo de Machine Learning
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

# Conteúdo da tab Deploy do Projeto
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

# Fim do conteúdo da página