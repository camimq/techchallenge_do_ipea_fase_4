import streamlit as st
import pandas as pd
import numpy as np

# Configura atributos da página
st.set_page_config(page_title='Conclusão', page_icon='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTADYDWw-Wdk19SHqSlthiAilmWGH6NfW5FVg&s', layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title('2. Conclusão')
st.write('''
          Dos objetivos colocados para a entrega deste projeto, entregamos o projeto com os ítens propostos, conforme detalhamento abaixo.
          
          :white_check_mark: **Criar um _dashboard_ interativo com ferramenta de livre escolha do grupo.**
              
          O _dashboard_ deve fazer parte de um _storytelling_ que traga _insights_ relevantes sobre a variação do preço do petróleo, como situalções geopolíticas, crises econômicas, demanda global por energia etc. **É obrigado apresentar, pelo menos, 4 _insights_**.
              
          :white_check_mark: **Criar um modelo de _Machine Learning_ que faça a previsão do preço do petróleo diariamente.**
          
          Esse modelo deve estar contemplado em seu _storytelling_ e deve conter o código desenvolvido, analisando as performances do modelo.
          
          :white_check_mark:**Criar um plano para fazer o _deploy_ em produção do modelo, com as ferramentas que são necessárias.**
          
          *Faça um MVP do modelo desenvolvido em produção, utilizando o **Streamlit**.*
         ''')