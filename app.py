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
    return easyocr.Reader(['en', 'pt'])

# Inicialização do modelo YOLO
@st.cache_resource
def load_model():
    # Usando modelo base do YOLOv8
    model = YOLO('yolov8n.pt')
    # Configurando para detectar apenas a classe 2 (carros) e 7 (caminhões)
    model.classes = [2, 7]
    return model

def is_plate_format(text):
    """Valida o formato da placa brasileira (antiga e Mercosul)"""
    # Remove espaços e caracteres especiais
    text = ''.join(e for e in text if e.isalnum()).upper()
    
    # Verifica formato antigo (ABC1234)
    if len(text) == 7 and text[:3].isalpha() and text[3:].isdigit():
        return True
    
    # Verifica formato Mercosul (ABC1D23)
    if len(text) == 7 and text[:3].isalpha() and text[3].isdigit() and text[4].isalpha() and text[5:].isdigit():
        return True
    
    return False

def preprocess_plate_image(img):
    """Pré-processa a imagem para melhorar o OCR"""
    try:
        # Converte para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aumenta o contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Reduz ruído
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Binarização adaptativa
        binary = cv2.adaptiveThreshold(denoised, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Operações morfológicas para limpar a imagem
        kernel = np.ones((2,2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return processed
    except Exception as e:
        st.error(f"Erro no pré-processamento: {str(e)}")
        return img

def process_plate(image, ocr_reader):
    try:
        # Pré-processa a imagem
        processed_img = preprocess_plate_image(image)
        
        # Aumenta o tamanho da imagem para melhor OCR
        height, width = processed_img.shape[:2]
        processed_img = cv2.resize(processed_img, (width*2, height*2))
        
        # Executa OCR
        results = ocr_reader.readtext(processed_img)
        
        valid_plates = []
        for (bbox, text, prob) in results:
            # Remove espaços e caracteres especiais
            clean_text = ''.join(e for e in text if e.isalnum()).upper()
            
            if is_plate_format(clean_text) and prob > 0.5:
                valid_plates.append((clean_text, prob))
        
        if valid_plates:
            # Retorna a placa com maior confiança
            return max(valid_plates, key=lambda x: x[1])[0]
            
        return None
    except Exception as e:
        st.error(f"Erro ao processar placa: {str(e)}")
        return None

def main():
    st.title("Sistema de Detecção de Placas de Veículos")
    
    with st.sidebar:
        st.header("Configurações")
        confidence_threshold = st.slider("Limite de Confiança", 0.0, 1.0, 0.25)
    
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
        
        # Detectando veículos na imagem
        with st.spinner("Processando imagem..."):
            results = model(image, conf=confidence_threshold)
        
        # Processando detecções
        detections = []
        output_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Expande um pouco a região para capturar melhor a placa
                height, width = image.shape[:2]
                y1 = max(0, y1 - 20)
                y2 = min(height, y2 + 20)
                x1 = max(0, x1 - 20)
                x2 = min(width, x2 + 20)
                
                # Recortando a região do veículo
                vehicle_region = image[y1:y2, x1:x2]
                
                # Processando OCR na região
                plate_text = process_plate(vehicle_region, reader)
                
                if plate_text:
                    # Salvando thumbnail
                    os.makedirs('thumbnails', exist_ok=True)
                    thumbnail_path = f"thumbnails/{plate_text}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(thumbnail_path, vehicle_region)
                    
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
        
        # Mostrando e salvando detecções
        if detections:
            st.subheader("Registro de Detecções")
            df = pd.DataFrame(st.session_state.detections_data)
            st.dataframe(df, use_container_width=True)
            
            # Salvando em CSV
            df.to_csv('deteccoes.csv', index=False)
            
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
