#carregando as bibliotecas
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model
from PIL import Image

var_model = "model"
var_model_cluster = "cluster.joblib"
var_dataset = "dataset.csv"

#carregando o modelo treinado.
model = load_model(var_model)
model_cluster = joblib.load(var_model_cluster)

#carregando o conjunto de dados.
dataset = pd.read_csv(var_dataset)

#imgem de cabeçalho
image = Image.open('img/head.jpg')
st.image(image, caption='Ticket gate')

# título
st.title("Analise da taxa de rotatividade")

# subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Inteligencia artificial para analise da taxa de rotatividade de funcionários de uma emprresa. "
 "Para visualizar estudo sobre determinado funcionário, insira os dados no meu ao lado.")


# imprime o conjunto de dados usado
#st.dataframe(dataset.drop("turnover",axis=1).head())

# grupos de empregados.
kmeans_colors = ['green' if c == 0 else '#b68900' if c == 1 else 'red' for c in model_cluster.labels_]

st.sidebar.subheader("Defina os atributos do empregado")

# mapeando dados do usuário para cada atributo
satisfaction = st.sidebar.number_input("Grau de satisfação do funcionario", value=dataset["satisfaction"].mean())
evaluation = st.sidebar.number_input("Avaliação do funcionário pela empresa", value=dataset["evaluation"].mean())
averageMonthlyHours = st.sidebar.number_input("Média de horas pr mês", value=dataset["averageMonthlyHours"].mean())
yearsAtCompany = st.sidebar.number_input("Anos de casa", value=dataset["yearsAtCompany"].mean())

# inserindo um botão no menu lateral
btn_predict = st.sidebar.button("Realizar Classificação")

# verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame()
    data_teste["satisfaction"] = [satisfaction]
    data_teste["evaluation"] =	[evaluation]    
    data_teste["averageMonthlyHours"] = [averageMonthlyHours]
    data_teste["yearsAtCompany"] = [yearsAtCompany]
    
    #imprime os dados de teste.
    st.subheader("Predição para os dados informados:")

    #realiza a predição.
    result = predict_model(model, data=data_teste)
    
    #recupera os resultados.
    classe = result["Label"][0]
    prob = result["Score"][0]*100
    
    if classe==1:
        st.write(f"EVASÃO com o valor de probabilidade: {prob:.2f}%")
    else:
        st.write(f"PERMANENCIA com o valor de probabilidade: {prob:.2f}%")

    fig = plt.figure(figsize=(10, 6))
    plt.scatter( x="satisfaction"
                ,y="evaluation"
                ,data=dataset[dataset.turnover==1],
                alpha=0.25,color = kmeans_colors)

    plt.xlabel("Satisfaction")
    plt.ylabel("Evaluation")
    
    plt.scatter( x=[satisfaction]
                ,y=[evaluation]
                ,color="black"
                ,marker="X",s=300)

    plt.title("Grupos de Empregados - Satisfação vs Avaliação.")
    plt.show()
    st.pyplot(fig) 