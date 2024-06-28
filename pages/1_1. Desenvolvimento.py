import streamlit as st
import pandas as pd
import numpy as np

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

with tab5:
  st.title('Deploy do Projeto')