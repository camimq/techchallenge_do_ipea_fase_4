import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='Home', page_icon='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTADYDWw-Wdk19SHqSlthiAilmWGH6NfW5FVg&s', layout="centered", initial_sidebar_state="auto", menu_items=None)

st.image('img/home_image.jpg', caption='Foto de Pixabay: https://www.pexels.com/pt-br/foto/por-do-sol-sobre-o-mar-301484/', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.title('Tech Challegenge FIAP - Fase 4')
st.subheader('Uma análise da base histórica do IPEA, com o objetivo de idenitifcar _insights_ relevantes para tomada de decisões e desenvolver um modelo de Machine Learning para previsão diária do custo do barril de petróleo.')

# Configura tabs da página
tab1, tab2, tab3 = st.tabs(['Sobre o Projeto', 'Descrição do Problema', 'Objetivo'])
                            
with tab1:
   st.header('Sobre o Projeto')
   st.image('https://static.streamlit.io/examples/cat.jpg', width=200)

with tab2:
   st.header('Descrição do Problema')
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header('Objetivo')
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


