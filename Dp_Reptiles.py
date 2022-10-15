import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
new_df = pd.read_csv('new_df')

def modelo():
    global modelo_Lr,y_test,X_test
    X = new_df[['Tamanho', 'Fucinho', 'Peso', 'Coloração', 'Alimentação', 'Região']]
    y = new_df[['Nome']].values

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)

    modelo_Lr = LogisticRegression(multi_class='ovr')
    modelo_Lr.fit(X_train, y_train)

def cabecalho():
    titulo = st.title('Classificação dos Répteis :crocodile:')
    subtitulo = st.subheader('Crocodilos x Jacarés')
    escrita = st.write('Máquina preditiva que irá definir o réptil, através de informações das características: ')
    return titulo, subtitulo, escrita

def pagina():
    global peso,regiao,tamanho,coloracao,fucinho,alimentacao,result

    regiao = st.sidebar.selectbox("Em qual Região você viu o animal",[0,1,2,3])
    alimentacao = st.sidebar.selectbox("Qual sua Alimentação", [0, 1, 2, 3, 4, 5, 6])
    coloracao = st.sidebar.selectbox("Qual sua Coloração", [0, 1, 2, 3, 4, 5, 6])
    fucinho = st.sidebar.selectbox("Como era seu Fucinho", [0, 1, 2])
    peso = st.sidebar.selectbox("Quantos Kg aproximadamente ele tinha",[200,270,285,300,320,340,370,400,420,460,470,850,900,1000,1100,1200,1300,1450,1500,1620])
    tamanho = st.sidebar.selectbox("Qual o Tamanho aproximado",[2.0,2.5,2.7,3.0,3.2,3.5,4.0,4.5,4.7,6.0,6.5,7.0,7.5,8.0])



    tab1,tab2,tab3,tab4 = st.tabs(["Região","Alimentação","Coloração","Fucinho"])

    with tab1:
        st.markdown("Região 0 ->>  Norte África")
        st.markdown("Região 1 ->>  América Central")
        st.markdown("Região 2 ->>  América do Sul")
        st.markdown("Região 3 ->>  Estados Unidos")

    with tab2:
        st.markdown("Alimentação 0 ->>  Antílopes")
        st.markdown("Alimentação 1 ->>  Cobras")
        st.markdown("Alimentação 2 ->>  Crustáceos")
        st.markdown("Alimentação 3 ->>  Insetos")
        st.markdown("Alimentação 4 ->>  Moluscos")
        st.markdown("Alimentação 5 ->>  Peixes")
        st.markdown("Alimentação 6 ->>  Zebras")

    with tab3:
        st.markdown("Coloração 0 ->>  Amarelo Escuro")
        st.markdown("Coloração 1 ->>  Bronze Preto")
        st.markdown("Coloração 2 ->>  Cinza")
        st.markdown("Coloração 3 ->>  Preto")
        st.markdown("Coloração 4 ->>  Preto Amarelo")
        st.markdown("Coloração 5 ->>  Verde Escuro")

    with tab4:
        st.markdown("Fucinho 0 ->>  Curto")
        st.markdown("Fucinho 1 ->>  Longo")
        st.markdown("Fucinho 2 ->>  Médio")

    st.markdown('Se estiver com todas opções selecionadas,Clique no botão abaixo:point_down:')

def pegar_dados():
    dados = {'Região': regiao,
             'Peso': peso,
             'Tamanho': tamanho,
             'Coloração': coloracao,
             'Fucinho': fucinho,
             'Alimentação': alimentacao
              }
    features = pd.DataFrame(dados, index=[0])
    return features


def resultado():

    previsao = modelo_Lr.predict(pegar_dados())
    if st.button('Classificar'):
        if previsao == 1:
            st.success('O Réptil foi classificado como: Crocodilo-do-Deserto')
        elif previsao == 2:
            st.success('O Réptil foi classificado como: Crocodilo-Americano')
        elif previsao == 3:
            st.success('O Réptil foi classificado como: Jacaré-do-Pantanal')
        else:
            st.success('O Réptil foi classificado como: Jacaré-Americano')

        st.subheader(f'Nossa máquina possui {accuracy_score(y_test,modelo_Lr.predict(X_test)) * 100}% de precisão')

if __name__ == '__main__':

    print(modelo(), cabecalho(), pagina(), pegar_dados(), resultado())