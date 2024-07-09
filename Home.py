import streamlit as st
import pandas as pd
import numpy as np

# Configura atributos da página
st.set_page_config(page_title='Home', page_icon='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTADYDWw-Wdk19SHqSlthiAilmWGH6NfW5FVg&s', layout="centered", initial_sidebar_state="auto", menu_items=None)

# Configura título e subtítulo da página
st.title('Tech Challegenge FIAP - Fase 4')
st.write('Uma análise da base histórica do IPEA, com o objetivo de idenitifcar _insights_ relevantes para tomada de decisões e desenvolver um modelo de Machine Learning para previsão diária do custo do barril de petróleo.')

# Inserção da imagem da home
st.image('img/home_image.jpg', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

# Configura tabs da página
tab1, tab2 = st.tabs(['Sobre o Projeto', 'Descrição do Problema'])
                            
with tab1:
  st.header('Sobre o Projeto')
  col1, col2 = st.columns(2)
  
  with col1:
    st.image('img/fiap.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  
  with col2:
    st.write('''
              Esta é uma aplicação _Streamlit_ desenvolvida pelos alunos do curso de **Pós-Graduação** em _Data Analytics_ da **FIAP**.
              
              Este projeto é proposto para os alunos ao final de cada módulo do curso, com o objetivo de aplicar os conhecimentos adquiridos no módulo em questão, junto com os conhecimentos prévios adquiridos nos módulos anteriores.
              
              O desenvolvimento do projeto é livre, em relação à metodos, ferramentas e tecnologias utilizadas, desde que sejam pertinentes ao escopo do projeto.
            ''')
  st.write('---')

with tab2:
   st.header('Descrição do Problema')
   st.write('''
              À partir da base de dados disponível no [_site_ do IPEA](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view), foi proposto o desenvolvimento de um _dashboard_ interativo que traga _insights_ relevantes para tomada de decisões de um cliente fictício.
              
              Além das análises propostas através do _dashboard_, o projeto também prevê o desenvolvimento de um modelo de _Machine Learning_ para previsão diária do custo do barril de petróleo.
              
              **Desta forma, o objetivo do projeto é:**
              
              - Criar um _dashboard_ interativo com ferramenta de livre escolha do grupo.
              - O _dashboard_ deve fazer parte de um _storytelling_ que traga _insights_ relevantes sobre a variação do preço do petróleo, como situalções geopolíticas, crises econômicas, demanda global por energia etc. **É obrigado apresentar, pelo menos, 4 _insights_**.
              - Criar um modelo de _Machine Learning_ que faça a previsão do preço do petróleo diariamente. Esse modelo deve estar contemplado em seu _storytelling_ e deve conter o código desenvolvido, analisando as performances do modelo.
              - Criar um plano para fazer o _deploy_ em produção do modelo, com as ferramentas que são necessárias.
              - Faça um MVP do modelo desenvolvido em produção, utilizando o **Streamlit**.
            ''')
   st.write('---')