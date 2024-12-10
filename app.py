import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import pandas as pd
from datetime import datetime
import os
from PIL import Image
import io

# Configuração da página Streamlit
st.set_page_config(page_title="Detecção de Placas", layout="wide")

# Configuração do estado da aplicação
if 'detections_data' not in st.session_state:
    st.session_state.detections_data = []

# Inicialização do EasyOCR
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en', 'pt'])  # Adicionando suporte para português

# Inicialização do modelo YOLO
@st.cache_resource
def load_model():
    # Carregando modelo específico para placas - substitua pelo caminho do seu modelo treinado
    return YOLO('best.pt')  # Use seu modelo treinado para placas

# Pré-processamento da imagem para melhorar OCR
def preprocess_plate_image(plate_img):
    # Converte para escala de cinza
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Aplica threshold adaptativo
    binary = cv2.adaptiveThreshold(gray, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Reduz ruído
    denoised = cv2.fastNlMeansDenoising(binary)
    
    return denoised

# Função para processar a placa e extrair o texto
def process_plate(image, ocr_reader):
    try:
        # Pré-processa a imagem
        processed_img = preprocess_plate_image(image)
        
        # Executa OCR
        results = ocr_reader.readtext(processed_img)
        
        if results:
            # Filtra resultados por confiança
            valid_results = [r for r in results if r[2] > 0.5]  # Ajuste o threshold conforme necessário
            
            if valid_results:
                # Pega o texto com maior confiança
                text = max(valid_results, key=lambda x: x[2])[1]
                
                # Remove espaços e caracteres especiais
                plate_text = ''.join(e for e in text if e.isalnum()).upper()
                
                # Valida formato da placa brasileira (ABC1234 ou ABC1D23)
                if len(plate_text) == 7 and plate_text[:3].isalpha():
                    return plate_text
                
        return None
    except Exception as e:
        st.error(f"Erro ao processar placa: {str(e)}")
        return None

# Função para salvar dados em CSV
def save_to_csv(data):
    if data:
        df = pd.DataFrame(data)
        if not os.path.exists('deteccoes.csv'):
            df.to_csv('deteccoes.csv', index=False)
        else:
            df.to_csv('deteccoes.csv', mode='a', header=False, index=False)

def main():
    st.title("Sistema de Detecção de Placas de Veículos")
    
    # Sidebar com configurações
    with st.sidebar:
        st.header("Configurações")
        confidence_threshold = st.slider("Limite de Confiança", 0.0, 1.0, 0.5)
        
    # Carregando modelos com indicador de progresso
    with st.spinner("Carregando modelos..."):
        model = load_model()
        reader = load_ocr()
    
    # Interface para upload de imagem
    uploaded_file = st.file_uploader("Escolha uma imagem", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Convertendo a imagem
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Criando colunas para exibição
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagem Original")
            st.image(image, channels="BGR")
        
        # Detectando objetos na imagem
        with st.spinner("Processando imagem..."):
            results = model(image)
        
        # Processando detecções
        detections = []
        output_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Verifica confiança da detecção
                if box.conf[0] >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Recortando a região da placa
                    plate_region = image[y1:y2, x1:x2]
                    
                    # Processando OCR na região da placa
                    plate_text = process_plate(plate_region, reader)
                    
                    if plate_text:
                        # Salvando thumbnail
                        thumbnail = Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB))
                        thumbnail_path = f"thumbnails/{plate_text}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        os.makedirs('thumbnails', exist_ok=True)
                        thumbnail.save(thumbnail_path)
                        
                        # Preparando dados para salvar
                        detection_data = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'placa': plate_text,
                            'thumbnail_path': thumbnail_path,
                            'confianca': float(box.conf[0])
                        }
                        
                        detections.append(detection_data)
                        st.session_state.detections_data.append(detection_data)
                        
                        # Desenhando retângulo e texto na imagem
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(output_image, plate_text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        with col2:
            st.subheader("Detecções")
            st.image(output_image, channels="BGR")
        
        # Salvando e mostrando detecções
        if detections:
            save_to_csv(detections)
            
            st.subheader("Registro de Detecções")
            df = pd.DataFrame(st.session_state.detections_data)
            st.dataframe(df, use_container_width=True)
            
            # Botão para baixar CSV
            if st.button("Baixar Registros (CSV)"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Clique para baixar",
                    data=csv,
                    file_name="deteccoes_placas.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Nenhuma placa detectada na imagem.")

if __name__ == "__main__":
    main()
