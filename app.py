import streamlit as st
import cv2
import pytesseract
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
import re

# Configurações iniciais
st.set_page_config(page_title="Reconhecimento de Placas", layout="wide")
st.title("Reconhecimento de Placas Mercosul")

# Inicialização da tabela de dados
if "placas" not in st.session_state:
    st.session_state["placas"] = pd.DataFrame(columns=["Timestamp", "Número da Placa", "Thumbnail", "Informações do Veículo"])

def processar_imagem(imagem):
    """Processa a imagem para extrair texto usando OCR."""
    imagem_cinza = cv2.cvtColor(np.array(imagem), cv2.COLOR_BGR2GRAY)
    texto_extraido = pytesseract.image_to_string(imagem_cinza, config="--psm 8")
    return texto_extraido

def buscar_dados_veiculo(placa):
    """Busca informações do veículo utilizando uma API pública."""
    url_api = f"https://api.consultarplaca.com.br/v2/vehicles/{placa}"
    headers = {
        "Authorization": "Bearer SEU_TOKEN_AQUI"
    }
    try:
        resposta = requests.get(url_api, headers=headers)
        if resposta.status_code == 200:
            return resposta.json()
        else:
            return "Dados não encontrados."
    except Exception as e:
        return f"Erro ao buscar informações: {e}"

def salvar_dados(timestamp, placa, thumbnail, informacoes):
    """Salva os dados na tabela."""
    nova_linha = {
        "Timestamp": timestamp,
        "Número da Placa": placa,
        "Thumbnail": thumbnail,
        "Informações do Veículo": informacoes
    }
    st.session_state["placas"] = pd.concat([st.session_state["placas"], pd.DataFrame([nova_linha])], ignore_index=True)

# Upload de imagem
uploaded_file = st.file_uploader("Envie uma imagem da placa:", type=["jpg", "jpeg", "png"])
if uploaded_file:
    imagem = Image.open(uploaded_file)
    st.image(imagem, caption="Imagem enviada", use_column_width=True)

    # Processamento da imagem
    texto_extraido = processar_imagem(imagem)
    st.write(f"Texto extraído: {texto_extraido}")

    # Verificação de padrão de placa Mercosul
    padrao_placa = r"[A-Z]{3}\d{1}[A-Z]{1}\d{2}"
    if re.match(padrao_placa, texto_extraido.strip()):
        placa = texto_extraido.strip()
        st.success(f"Placa reconhecida: {placa}")

        # Gerar thumbnail
        thumbnail = imagem.resize((100, 50))
        thumbnail_buffer = BytesIO()
        thumbnail.save(thumbnail_buffer, format="JPEG")
        thumbnail_bytes = thumbnail_buffer.getvalue()

        # Buscar informações do veículo
        informacoes = buscar_dados_veiculo(placa)

        # Salvar os dados
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        salvar_dados(timestamp, placa, thumbnail_bytes, informacoes)
    else:
        st.error("Nenhuma placa válida foi encontrada na imagem.")

# Exibir tabela de dados
st.write("### Dados das placas reconhecidas")
if not st.session_state["placas"].empty:
    st.dataframe(st.session_state["placas"].drop(columns=["Thumbnail"]))
