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
    return YOLO('yolov8n.pt')

def enhance_image(image):
    """Melhora a imagem para detecção de placas"""
    # Converte para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aumenta o contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Redução de ruído
    denoised = cv2.fastNlMeansDenoising(enhanced)
    
    # Binarização adaptativa
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Operações morfológicas para limpar ruídos
    kernel = np.ones((2,2), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return clean

def find_plate_candidates(image):
    """Encontra regiões candidatas a serem placas"""
    # Melhora a imagem
    enhanced = enhance_image(image)
    
    # Detecção de bordas
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Encontra contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    height, width = image.shape[:2]
    min_area = (width * height) * 0.01  # 1% da área da imagem
    max_area = (width * height) * 0.15  # 15% da área da imagem
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 1.5 <= aspect_ratio <= 4.5:  # Proporção típica de placas
                candidates.append((x, y, w, h))
    
    return candidates

def is_plate_format(text):
    """Valida o formato da placa brasileira (antiga e Mercosul)"""
    # Remove espaços e caracteres especiais
    text = ''.join(e for e in text if e.isalnum()).upper()
    
    if len(text) != 7:
        return False
        
    # Caracteres comumente confundidos
    text = text.replace('O', '0').replace('I', '1').replace('Z', '2')
    text = text.replace('S', '5').replace('B', '8').replace('Q', '0')
    
    # Verifica formato antigo (ABC1234)
    if text[:3].isalpha() and text[3:].isdigit():
        return True
    
    # Verifica formato Mercosul (ABC1D23)
    if (text[:3].isalpha() and text[3].isdigit() and 
        text[4].isalpha() and text[5:].isdigit()):
        return True
        
    # Tenta correções comuns para placas Mercosul
    if len(text) >= 7:
        # Tenta interpretar caracteres confusos
        potential_plate = list(text)
        # Tenta converter números em letras nas posições corretas
        if text[4].isdigit():
            letter_map = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}
            if text[4] in letter_map:
                potential_plate[4] = letter_map[text[4]]
                new_text = ''.join(potential_plate)
                if (new_text[:3].isalpha() and new_text[3].isdigit() and 
                    new_text[4].isalpha() and new_text[5:].isdigit()):
                    return True
    
    return False

def process_image_region(image, reader, x=None, y=None, w=None, h=None):
    """Processa uma região da imagem para encontrar placa"""
    if x is not None:
        region = image[y:y+h, x:x+w]
    else:
        region = image
    
    # Redimensiona a imagem para um tamanho maior
    scale = 2
    height, width = region.shape[:2]
    region = cv2.resize(region, (width * scale, height * scale))
    
    # Lista de transformações para tentar
    transforms = [
        lambda img: img,  # Original
        lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),  # Escala de cinza
        lambda img: enhance_image(img),  # Melhorada
        lambda img: cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1],  # Binarização simples
        lambda img: cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # Binarização Otsu
        # Transformações específicas para Mercosul
        lambda img: cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)[1],  # Alto threshold para fundo branco
        lambda img: cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11)  # Adaptativo mais agressivo
    ]
    
    best_result = None
    best_confidence = 0
    
    for transform in transforms:
        try:
            processed = transform(region)
            results = reader.readtext(processed)
            
            for bbox, text, conf in results:
                clean_text = ''.join(e for e in text if e.isalnum()).upper()
                if is_plate_format(clean_text) and conf > best_confidence:
                    best_result = (clean_text, conf)
                    best_confidence = conf
        except:
            continue
    
    return best_result

def main():
    st.title("Sistema de Detecção de Placas de Veículos")
    
    with st.sidebar:
        st.header("Configurações")
        confidence_threshold = st.slider("Limite de Confiança", 0.0, 1.0, 0.2)
    
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
        
        output_image = image.copy()
        detections = []
        
        # Primeiro tenta processar a imagem inteira
        plate_result = process_image_region(image, reader)
        
        if plate_result:
            plate_text, conf = plate_result
            if conf > confidence_threshold:
                detections.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'placa': plate_text,
                    'confianca': conf
                })
                cv2.putText(output_image, f"{plate_text} ({conf:.2f})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Se não encontrou na imagem inteira, procura por regiões candidatas
        if not detections:
            candidates = find_plate_candidates(image)
            
            for x, y, w, h in candidates:
                plate_result = process_image_region(image, reader, x, y, w, h)
                
                if plate_result:
                    plate_text, conf = plate_result
                    if conf > confidence_threshold:
                        detections.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'placa': plate_text,
                            'confianca': conf
                        })
                        cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(output_image, f"{plate_text} ({conf:.2f})", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        with col2:
            st.subheader("Detecções")
            st.image(output_image, channels="BGR")
        
        if detections:
            st.subheader("Registro de Detecções")
            df = pd.DataFrame(detections)
            st.dataframe(df, use_container_width=True)
            
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
