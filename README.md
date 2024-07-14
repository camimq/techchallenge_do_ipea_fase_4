# Tech Challenge Ipea | Fase 4 - Curso de Pós-Gradução FIAP em Data Analytics

Projeto desenvolvido para conclusão do módulo 4 do curso de Pós-Graduação em Data Analytics da FIAP. O projeto consiste em desenvolver um modelo de Machine Learning para prever o preço do barril de petróleo Brent, utilizando dados históricos de preços de fechamento do petróleo.

Além disso, foi criado um dashboard para visualização dos principais dados levantados na análise exploratória, contendo _insights_ relevantes para tomadas de decisões, no cenário de variação do preço do petróleo, dentro do período analisado (2019 à 2023).

## :key: Como Rodar o Projeto

Clone este repositório, acesse a pasta do projeto e siga os passos abaixo:

> No Terminal | 1. Inicia o ambiente virtual

`.\venv/Scripts/activate`

> No terminal | 2. Executa o arquivo principal com Streamlit

`streamlit run Home.py`

## 📄 Documentação do código
Dentro do projeto, há diversas funções que foram criadas para alcançarmos os objetivos do desafio. Muitas dessas funções são extensas, por isso, ficou definido que a documentação desses códigos deveria ser colocada em um documento apartado, afim de manter o código limpo e organizado, dado que a documentação em questão, por questões pedagógicas é mais detalhada.

### 📢 Arquivo: `Desenvolvimento.py`
Nesta página do projeto, está a parte mais extensa e completa do desenvolvimento do projeto. Nesta página, encontramos os seguintes conteúdos:

1. Sobre os Dados
2. Análise Exploratória de Dados
3. Dashboard
4. Modelo de Machine Learning
5. Deploy do Projeto

#### Funções de Modelo Machine Learning

**Importante:** Documentação do código, foi gerada com auxílio de Inteligência Artificial.

##### Função def `get_data()`
Uma implementação da função _download_ que faz o _download_ de dados de ações da **Yahoo Finance**. A função possui vários parâmetros que permitem personalizar o _download_, como o período de tempo, o intervalo dos dados, a lista de ações a serem baixadas, entre outros. Serve para baixar dados de ações da **Yahoo Finance** de forma programática, permitindo a personalização do período de tempo, o intervalo dos dados e outras opções de _download_.

```
def get_data():
    start_date = '2022-01-01'
    end_date = '2024-12-31'
    df = yf.download('BZ=F', start=start_date, end=end_date)
    df = df[['Close']]  
    df.reset_index(inplace=True)  
    alpha = 0.09  
    df['Smoothed_Close'] = df['Close'].ewm(alpha=alpha, adjust=False).mean()
    return df
```

#### Função `def plot_serie_suaviada(df)`

Esta função personalizada, plota uma temporal suavizada usando suavização exponencial.
**A função plota duas linhas nos eixos:**
1. A primeira linha representa a série temporal original e é plotada usando os valores do índice do DataFrame `df` no eixo x e os valores da coluna `'Close'` no eixo `y`. A linha é colorida de azul e é rotulada como `'Original'`.

2. A segunda linha representa a série temporal suavizada e é plotada usando os mesmos valores do índice do DataFrame `df` no eixo `x` e os valores da coluna `'Smoothed_Close'` no eixo `y`. A linha é colorida de vermelho e é rotulada como `'Suavizado (alpha=0.9)'`.

```
def plot_serie_suavizada(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label='Original', color='blue')
    ax.plot(df.index, df['Smoothed_Close'], label=f'Suavizado (alpha=0.9)', color='red')
    ax.set_title('Série Temporal Suavizada usando Suavização Exponencial')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço de Fechamento')
    ax.legend()
    st.pyplot(fig)
```

#### Função `def train_model(df)`
Esta função treina um modelo de aprendizado de máquina usando uma série temporal de dados financeiro.

**1. Início do treinamento:** exibe uma mensagem, indicando que o treinamento do modelo começou.
**2. Preparação dos Dados:**
    - Extrai a coluna `smoothed_Close` do DataFrame e transforma em um array numpy.
    - Redimensiona o array para uma única coluna.
    - Normaliza os dados usando `MinMaxScaler` para que os valores estejam entre 0 e 1.
**3. Divisão dos Dados:**
    - Define uma porcentagem de divisão (80% para treinamento, 20% para teste).
  - Divide os dados em conjuntos de treinamento e teste com base na porcentagem definida.
  - Separa as datas correspondentes aos dados de treinamento e teste.
**4. Preparação para o Treinamento:**
  - Define um `look_back` de 10, que é o número de passos no tempo a serem considerados para prever o próximo valor.
  - Cria geradores de séries temporais para os conjuntos de treinamento e teste, que irão fornecer os dados em lotes para o modelo.
**5. Construção do Modelo:**
  - Inicializa um modelo sequencial.
  - Adiciona uma camada **LSTM** com 100 unidades e ativação `'relu'`.
  - Adiciona uma camada densa para a saída.
  - Compila o modelo com o otimizador `'adam'` e a função de perda `'mse'` (erro quadrático médio), monitorando também o **MSE** como métrica.
**6. Treinamento do Modelo:**
  - Treina o modelo usando o gerador de treinamento por um número definido de épocas (`num_epochs`).
**7. Avaliação do Modelo:**
  - Avalia o modelo usando o gerador de teste para calcular o **MSE**.
**8. Previsões e Transformações:**
  - Faz previsões no conjunto de teste.
  - Inverte a normalização dos dados previstos e dos dados reais de teste para obter valores na escala original.
**9. Cálculo de Métricas:**
  - Ajusta as dimensões dos dados reais para corresponder às previsões.
  - Calcula o **MAPE (Erro Percentual Absoluto Médio)** para avaliar a precisão das previsões.
  - Calcula o **RMSE (Raiz do Erro Quadrático Médio)** a partir do **MSE** obtido na avaliação.
**10. Retorno dos Resultados:**
  - Retorna uma série de objetos, incluindo os dados processados, os conjuntos de treinamento e teste, as métricas de desempenho, os geradores e o modelo treinado, além do objeto scaler para inversão de futuras previsões.

Este código é utilizado para a tarefa de previsão de séries temporais, especialmente no contexto de dados financeiros, onde a precisão das previsões é crucial para a tomada de decisões.

```
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
```

#### Função `get_features (df, model)`:
A função é projetada para processar dados financeiros e prepará-los para modelagem de séries temporais, além de gerar previsões futuras.

**1. Entrada da Função:**
  - **`df`**: DataFrame contendo os dados financeiros.
  - **`model`**: Modelo de aprendizado de máquina treinado para fazer previsões.
**2. Processamento dos Dados:**
  -Extrai a coluna `Smoothed_Close` do DataFrame `df` e a transforma em um array numpy. Essa coluna representa uma versão suavizada dos preços de fechamento, provavelmente para reduzir a volatilidade e destacar tendências de longo prazo.
  - Redimensiona o array para ter uma única coluna, preparando-o para a normalização.
  - Normaliza os dados usando `MinMaxScaler` para que os valores estejam no intervalo de 0 a 1. Isso é uma prática comum em modelagem de séries temporais para facilitar o treinamento do modelo.
**3. Divisão dos Dados:**
  - Define uma porcentagem de divisão (80% para treinamento, 20% para teste).
  - Divide os dados normalizados em conjuntos de treinamento (`close_train`) e teste (`close_test`) com base na porcentagem definida.
  - Separa as datas correspondentes (`date_train` e `date_test`) aos conjuntos de treinamento e teste para uso posterior, possivelmente em visualizações ou para indexar as previsões.
**4. Previsão Futura:**
  - Define o número de dias futuros (`num_prediction`) para os quais as previsões devem ser feitas.
  - Define um `lock_back`, que especifica o número de dias passados a serem considerados para fazer uma previsão.
  - Chama a função *predict*, passando o número de dias para prever, o modelo treinado e o valor de `lock_back`, para gerar previsões futuras (*forecast*).
  - Gera datas para essas previsões futuras chamando `predict_dates`, presumivelmente para criar um índice temporal para as previsões.
**5. Retorno da Função:**
  - Retorna os dados normalizados (`close_data`), os conjuntos de teste e treinamento (`close_test, close_train`), as datas correspondentes a esses conjuntos (d`ate_test, date_train`), as datas das previsões futuras (`forecast_dates`), as previsões futuras (`forecast`) e o objeto *scaler* usado para normalizar os dados. Este objeto *scaler* pode ser usado para inverter a normalização das previsões futuras ou de novos dados de entrada.

Essa função prepara dados para análise de séries temporais, realizar divisões de treinamento/teste e gerar previsões futuras com um modelo existente.

```
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
```

#### Função `def exibir_metricas(mse, mape, rmse_value)`:

A função `exibir_metricas` é definida para receber três parâmetros: `mse`, `mape`, e `rmse_value`. Ela tem como objetivo exibir métricas de erro de modelos de machine learning ou estatísticos, utilizando o *streamlit*.

  - **`mse`**: Espera-se que seja uma lista ou tupla, onde o primeiro elemento (mse[0]) é utilizado. **MSE significa _Mean Squared Error_ (Erro Quadrático Médio)**, uma métrica que mede a qualidade de um modelo, calculando a média dos quadrados dos erros, ou seja, a diferença quadrática média entre os valores estimados e os reais.
  - **`mape`**: Representa o **_Mean Absolute Percentage Error_ (Erro Percentual Absoluto Médio)**, uma métrica que mede a precisão de um modelo em termos percentuais. É exibido formatado com duas casas decimais (`:.2f`) seguido de um símbolo de porcentagem (`%`), indicando que é um valor percentual.
  - **`rmse_value`**: RMSE significa **_Root Mean Squared Error (Raiz do Erro Quadrático Médio)_**, uma métrica que também mede a qualidade de um modelo, sendo a raiz quadrada da média dos quadrados dos erros. Diferentemente do MSE, o RMSE está na mesma unidade que os dados de entrada, o que facilita sua interpretação.

A função utiliza st.write para exibir os valores dessas métricas, provavelmente em um aplicativo web criado com Streamlit. Cada chamada a st.write exibe uma linha de texto no aplicativo, mostrando o nome da métrica seguido de seu valor.

```
def exibir_metricas(mse, mape, rmse_value):
    st.write(f"MSE: {mse[0]}")
    st.write(f"MAPE: {mape:.2f}%")
    st.write(f"RMSE: {rmse_value}")
```

#### Função `def plot_prediction(date_train, date_test, close_train, close_test, model, test_generator)`:

A função é projetada para visualizar previsões de modelos em comparação com dados reais, especificamente para previsões de preços de petróleo bruto Brent. Ela utiliza a biblioteca Plotly para criar gráficos interativos.

**1. Parâmetros da Função:**
  - **`date_train`**: Datas correspondentes aos dados de treinamento.
  - **`date_test`**: Datas correspondentes aos dados de teste.
  - **`close_train`**: Valores reais de fechamento para o conjunto de treinamento.
  - **`close_test`**: Valores reais de fechamento para o conjunto de teste.
  - **`model`**: O modelo de machine learning que será usado para fazer previsões.
  - **`test_generator`**: Um gerador ou estrutura de dados que fornece os dados de teste ao modelo para fazer previsões.
**2. Previsão:**
  - A função começa fazendo previsões usando o model passado como parâmetro, aplicando-o ao t`est_generator`.
**3. Reshape dos Dados:**
  - Os dados de `close_train, close_test`, e as *prediction* são reformatados `(usando .reshape((-1)))` para garantir que tenham a forma correta para plotagem. Isso geralmente transforma os dados em um formato unidimensional.
**4. Criação dos Traços:**
  - Três traços são criados usando go.Scatter da biblioteca Plotly:
      - **`trace1`**: Representa os dados de treinamento (`date_train` vs. `close_train`).
      - **`trace2`**: Representa as previsões do modelo (`date_test` vs. `prediction`).
      - **`trace3`**: Representa os dados reais de teste (`date_test` vs. `close_test`), também conhecidos como **"Ground Truth"**.
**5. Layout do Gráfico:**
  - O layout do gráfico é definido com um título e rótulos para os eixos X e Y.
**6. Criação e Exibição do Gráfico:**
  - Um objeto go.Figure é criado com os traços e o layout definidos anteriormente.
  - O gráfico é exibido usando `st.plotly_chart(fig)`, presumivelmente em um aplicativo web criado com Streamlit, onde st é o alias para a biblioteca Streamlit.

Essa função mostra como as previsões do modelo se alinham com os dados reais, permitindo uma avaliação visual da performance do modelo em prever os preços de fechamento do petróleo bruto Brent.

```
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
```

#### Função `def predict (num_prediction, model, look_back)`:
A função predict é projetada para fazer previsões futuras com base em um modelo de aprendizado de máquina fornecido.

**1. Parâmetros da Função:**
- **`num_prediction`**: O número de previsões futuras que a função deve gerar.
- **`model`**: O modelo de aprendizado de máquina que será usado para fazer as previsões.
- **`look_back`**: O número de pontos de dados anteriores a serem considerados para fazer uma única previsão.
**2. Inicialização da Lista de Previsões:**
- A função começa inicializando `prediction_list` com os últimos `look_back` pontos de dados de `close_data`. `close_data` parece ser uma variável externa à função, presumivelmente contendo dados históricos.
3. Loop de Previsão:
- Para cada previsão que precisa ser feita (determinada por `num_prediction`), a função executa as seguintes etapas:
    - Seleciona os últimos `look_back` pontos de dados de `prediction_list` para usar como entrada para o modelo.
    - Redimensiona a entrada para o formato esperado pelo modelo ((1, `look_back`, 1)), indicando 1 amostra, `look_back` pontos de tempo, e 1 característica por ponto de tempo.
    - Usa o modelo para fazer uma previsão (`model.predict(x)`) e extrai o valor previsto.
    - Anexa o valor previsto a `prediction_list`, para que possa ser usado nas próximas iterações do loop.
**4. Ajuste Final da Lista de Previsões:**
- Após completar todas as previsões, a função ajusta `prediction_list` para remover os dados iniciais usados para a primeira previsão, mantendo apenas as previsões geradas. Isso é feito selecionando os elementos de `prediction_list` a partir de `look_back-1`.
5. Retorno:
- Finalmente, a função retorna `prediction_list`, que agora contém as previsões futuras geradas pelo modelo.

```
def predict(num_prediction, model, look_back):
    prediction_list = close_data[-look_back:]

    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]

    return prediction_list
```

#### Função `def predict_dates(num_prediction)`:
Esta função é projetada para gerar uma lista de datas futuras com base na última data presente em um DataFrame do pandas.

**1. Entrada (`num_prediction`)**: A função recebe um parâmetro `num_prediction`, que especifica o número de datas futuras a serem previstas.
**2. Obtém a última data (`last_date`)**: Dentro da função, `df['Date'].values[-1]` é usado para acessar a última data na coluna 'Date' de um DataFrame df. O método .values converte os dados da coluna em um array do NumPy, e [-1] seleciona o último elemento desse array, ou seja, a última data.
**3. Gera datas futuras (`prediction_dates`)**: Utiliza `pd.date_range()` para gerar um intervalo de datas começando pela última data (`last_date`) e estendendo-se por um número de períodos igual a `num_prediction + 1`. O +1 é necessário porque o pd.`date_range()` inclui a data de início no intervalo gerado. O resultado é uma lista de objetos Timestamp do pandas representando cada data futura.
**4. Retorno:** A função retorna a lista p`rediction_dates` contendo as datas futuras previstas.

Essa função é útil em contextos de análise de séries temporais, onde prever datas futuras com base nos dados existentes é uma tarefa comum, como em previsões financeiras, previsões de demanda, entre outras.

```
def predict_dates(num_prediction):
    last_date = df['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates
```

#### Função `def plot_forecast(date_test,close_test,forecast_dates,forecast)`:
Esta função é projetada para visualizar previsões de séries temporais, especificamente para preços de fechamento de commodities, como o petróleo Brent. 

**1. Parâmetros:**
  - **`date_test`**: Um array ou lista contendo as datas correspondentes aos valores reais de fechamento.
  - **`close_test`**: Um array contendo os valores reais de fechamento. Este array é redimensionado para garantir que seja unidimensional.
  - **`forecast_dates`**: Um array ou lista contendo as datas para as quais as previsões foram feitas.
  - **`forecast`**: Um array contendo os valores previstos de fechamento para as datas em `forecast_dates`.
**2. Redimensionamento de `close_test`:**
  - **`close_test`** = `close_test.reshape((-1))`: Garante que `close_test` seja um array unidimensional, o que é necessário para a plotagem.
**3. Criação de Traces:**
  - **`trace1`**: Representa os dados reais. Utiliza `date_test` como eixo x e `close_test` como eixo y. É configurado para ser exibido como uma linha e recebe o nome `'Data'`.
  - **`trace2`**: Representa as previsões. Utiliza `forecast_dates` como eixo x e forecast como eixo y. Também é configurado para ser exibido como uma linha e recebe o nome `'Prediction'`.
**4. Layout:**
  - Define o título do gráfico como "Forecast Brent Crude Oil Price".
  - Configura os títulos dos eixos x e y como "Data" e "Fechamento", respectivamente.
**5. Criação e Exibição do Gráfico:**
  - **`fig = go.Figure(data=[trace1, trace2], layout=layout)`**: Cria um objeto Figure do Plotly, combinando os traces e o layout definidos anteriormente.
  - **`st.plotly_chart(fig)`**: Utiliza a função `plotly_chart` do Streamlit para exibir o gráfico. Streamlit é uma biblioteca que facilita a criação de aplicativos web para análise de dados em Python.

Essa função é utilizada para visualizar como as previsões de uma série temporal se comparam com os dados reais, permitindo uma análise visual da precisão das previsões.

```
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
```

#### Função `def teste (df, forecast_date, forecast, scaler)`:

Esta função realiza uma série de operações em dados de séries temporais, com o objetivo de preparar e apresentar previsões futuras baseadas em dados passados.

**1. Entradas da Função:**
A função recebe quatro parâmetros:
  - **`df`**: Um DataFrame que contém os dados históricos.
  - **`forecast_dates`**: Um array ou lista contendo as datas para as quais as previsões serão feitas.
  - **`forecast`**: Um array contendo os valores previstos para as datas futuras.
  - scaler: Um objeto de escalonamento (normalmente de uma biblioteca como sklearn) que foi usado para normalizar os dados antes do processo de modelagem.
**2. Desnormalização das Previsões:**
  - As previsões (`forecast`) são primeiro remodeladas para um formato de coluna única e depois desnormalizadas usando o scaler fornecido. Isso é feito para converter os valores previstos de volta à sua escala original.
**3. Preparação dos Dados Históricos:**
  - O DataFrame df é convertido para um novo DataFrame para garantir que está no formato correto.
  - Uma cópia das colunas `'Date'` e `'Smoothed_Close'` é criada (`df_past`), e a coluna `'Date'` é convertida para o tipo datetime.
  - Uma nova coluna `'Forecast'` é adicionada ao `df_past`, inicialmente preenchida com valores NaN, exceto pelo último valor que é preenchido com o último valor de `'Smoothed_Close'`.
**4. Preparação dos Dados Futuros:**
  - Um novo DataFrame `df_future` é criado para armazenar as previsões futuras, contendo colunas para `'Date'`, `'Smoothed_Close'` (inicialmente preenchida com NaN), e `'Forecast'` (preenchida com os valores desnormalizados das previsões).
**5. Combinação dos Dados Históricos e Futuros:**
  - Os DataFrames `df_past` e d`f_future` são combinados em um único DataFrame results, que é então indexado pela coluna `'Date'`.
**6. Seleção e Apresentação das Últimas 15 Previsões:**
  - A função então seleciona as últimas 15 linhas do DataFrame results, que contêm as previsões mais recentes.
  - Essas 15 linhas são transpostas (transformando linhas em colunas e vice-versa) para facilitar a visualização.
  - Finalmente, essas previsões transpostas são exibidas usando `st.write()`, presumivelmente uma função de uma biblioteca como Streamlit, indicando que esta função pode ser parte de um aplicativo web para visualização de dados.
**7. Retorno:** A função retorna o DataFrame results, que contém tanto os dados históricos quanto as previsões futuras, com as datas como índice.

**Observações:**
- A função parece ter um erro lógico na parte onde pretende selecionar e exibir os últimos 14 valores da coluna `'Forecast'`, mas na verdade seleciona e transpõe as últimas 15 linhas do DataFrame results.
- Há também um potencial problema com a modificação direta de um DataFrame usando `df_past['Forecast'].iloc[-1] = ...` dentro de um contexto pandas sem primeiro copiar o DataFrame para evitar avisos de `SettingWithCopy`.
- A função utiliza pd e np, que são abreviações comuns para as bibliotecas Pandas e NumPy, respectivamente, e `st.write()` sugere o uso da biblioteca Streamlit para exibição de dados em um aplicativo web.

```
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
```

#### Função `def_plot predict(df)`

A função é projetada para visualizar previsões de dados em um gráfico, utilizando a biblioteca Plotly para a criação do gráfico e Streamlit para a exibição.

**1. Entrada:** A função aceita um DataFrame `df` como entrada. Este DataFrame deve conter as colunas `'Smoothed_Close'` e `'Forecast'`, além de um índice de datas que inclui `'2024-01-01'` em diante.
**2. Seleção de Dados:** Dentro da função, é feita uma seleção de dados a partir de `'2024-01-01'` até o final do DataFrame, armazenando este subset em `results2024`. Isso foca a visualização nos dados a partir dessa data.
**3. Preparação dos Dados para o Gráfico:** Dois conjuntos de dados são preparados para plotagem:
  - Um para os valores reais (`'Smoothed_Close'`), representados por uma linha no gráfico.
  - Outro para as previsões (`'Forecast'`), também representado por uma linha no gráfico.
**4. Configuração do Layout do Gráfico:** O layout do gráfico é definido com um título `'Forecast Brent'`.
**5. Criação do Gráfico:** Um objeto Figure do Plotly é criado com os dados e o layout preparados anteriormente.
**6. Exibição do Gráfico:** Por fim, o gráfico é exibido usando `st.plotly_chart(fig)`, que é uma função do Streamlit para renderizar gráficos Plotly.

Resumindo, esta função pega um DataFrame com previsões e valores reais, filtra os dados a partir de `'2024-01-01'`, e utiliza Plotly e Streamlit para criar e exibir um gráfico comparativo entre os valores reais e as previsões.

```
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
```