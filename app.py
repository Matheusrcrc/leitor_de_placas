import streamlit as st
import cv2
import numpy as np
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
    reader = easyocr.Reader(['en', 'pt'], gpu=False)
    reader.detector.eval()
    return reader

def compress_image(image, max_size=800):
    """Comprime a imagem mantendo a proporção"""
    height, width = image.shape[:2]
    
    # Calcula a nova dimensão mantendo a proporção
    if width > height:
        if width > max_size:
            ratio = max_size / width
            new_width = max_size
            new_height = int(height * ratio)
        else:
            return image
    else:
        if height > max_size:
            ratio = max_size / height
            new_height = max_size
            new_width = int(width * ratio)
        else:
            return image
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def find_plate_region(image):
    """Encontra a região retangular da placa na imagem"""
    # Converte para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplica blur para reduzir ruído
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detecta bordas
    edges = cv2.Canny(blurred, 50, 150)
    
    # Encontra contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plate_candidates = []
    height, width = image.shape[:2]
    min_area = (width * height) * 0.01  # 1% da área da imagem
    max_area = (width * height) * 0.15  # 15% da área da imagem
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Aproxima o contorno para um polígono
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # Verifica se é um retângulo (4 vértices)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                
                # Verifica se tem proporção típica de placa (3:1 até 4:1)
                if 2.5 <= aspect_ratio <= 4.5:
                    plate_candidates.append((x, y, w, h, area))
    
    # Retorna o candidato com maior área
    if plate_candidates:
        return max(plate_candidates, key=lambda x: x[4])[:4]
    return None

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
    
    return False

def process_image_region(image, reader, x=None, y=None, w=None, h=None):
    """Processa uma região da imagem para encontrar placa"""
    if x is not None:
        # Expande um pouco a região para garantir
        height, width = image.shape[:2]
        x = max(0, x - 5)
        y = max(0, y - 5)
        w = min(width - x, w + 10)
        h = min(height - y, h + 10)
        region = image[y:y+h, x:x+w]
    else:
        region = image
    
    # Redimensiona a região para um tamanho padrão
    target_width = 400  # Largura padrão para processamento
    aspect_ratio = region.shape[1] / region.shape[0]
    target_height = int(target_width / aspect_ratio)
    region = cv2.resize(region, (target_width, target_height))
    
    # Lista de transformações para tentar
    transforms = [
        lambda img: img,  # Original
        lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),  # Escala de cinza
        lambda img: cv2.threshold(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
            0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]  # Binarização Otsu
    ]
    
    best_result = None
    best_confidence = 0
    
    for transform in transforms:
        try:
            processed = transform(region)
            results = reader.readtext(
                processed,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                batch_size=1,
                paragraph=False,
                height_ths=0.5,
                width_ths=0.5
            )
            
            for bbox, text, conf in results:
                clean_text = ''.join(e for e in text if e.isalnum()).upper()
                if is_plate_format(clean_text) and conf > best_confidence:
                    best_result = (clean_text, conf)
                    best_confidence = conf
        except Exception as e:
            continue
    
    return best_result if best_result and best_result[1] > 0.2 else None

def main():
    st.title("Sistema de Detecção de Placas de Veículos")
    
    with st.sidebar:
        st.header("Configurações")
        confidence_threshold = st.slider("Limite de Confiança", 0.0, 1.0, 0.2)
    
    # Carregando modelos com indicador de progresso
    with st.spinner("Carregando modelos..."):
        reader = load_ocr()
    
    # Interface para upload de imagem
    uploaded_file = st.file_uploader("Escolha uma imagem", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Lê e comprime a imagem
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = compress_image(image)
        
        # Criando colunas para exibição
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagem Original")
            st.image(image, channels="BGR")
        
        # Processa a imagem
        output_image = image.copy()
        detections = []
        
        # Tenta encontrar a região da placa
        plate_rect = find_plate_region(image)
        
        if plate_rect:
            x, y, w, h = plate_rect
            # Desenha retângulo na região detectada
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Processa a região da placa
            plate_result = process_image_region(image, reader, x, y, w, h)
            
            if plate_result:
                plate_text, conf = plate_result
                if conf > confidence_threshold:
                    detections.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'placa': plate_text,
                        'confianca': conf
                    })
                    # Adiciona texto acima do retângulo
                    cv2.putText(output_image, f"{plate_text} ({conf:.2f})", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                              (0, 255, 0), 2)
        
        # Se não encontrou com a detecção de região, tenta processar a imagem inteira
        if not detections:
            plate_result = process_image_region(image, reader)
            if plate_result:
                plate_text, conf = plate_result
                if conf > confidence_threshold:
                    detections.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'placa': plate_text,
                        'confianca': conf
                    })
                    # Adiciona texto no topo da imagem
                    cv2.putText(output_image, f"{plate_text} ({conf:.2f})", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                              (0, 255, 0), 2)
        
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
