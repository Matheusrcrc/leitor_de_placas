import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import pandas as pd
from datetime import datetime
import requests
import json
import os
from PIL import Image
import io

# Configuração da página Streamlit
st.set_page_config(page_title="Detecção de Placas", layout="wide")

# Inicialização do EasyOCR
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

# Inicialização do modelo YOLO
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

# Função para processar a placa e extrair o texto
def process_plate(image, ocr_reader):
    try:
        results = ocr_reader.readtext(image)
        if results:
            # Pega o texto com maior confiança
            text = max(results, key=lambda x: x[2])[1]
            # Remove espaços e caracteres especiais
            plate_text = ''.join(e for e in text if e.isalnum())
            return plate_text
        return None
    except Exception as e:
        st.error(f"Erro ao processar placa: {str(e)}")
        return None

# Função para consultar informações do veículo
def get_vehicle_info(plate):
    try:
        # Exemplo de API fictícia - substitua pela API real
        api_url = f"https://api.exemplo.com/consulta/{plate}"
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Função para salvar dados em CSV
def save_to_csv(data):
    df = pd.DataFrame(data)
    if not os.path.exists('deteccoes.csv'):
        df.to_csv('deteccoes.csv', index=False)
    else:
        df.to_csv('deteccoes.csv', mode='a', header=False, index=False)

def main():
    st.title("Sistema de Detecção de Placas de Veículos")
    
    # Carregando modelos
    model = load_model()
    reader = load_ocr()
    
    # Interface para upload de imagem
    uploaded_file = st.file_uploader("Escolha uma imagem", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Convertendo a imagem
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Mostrando a imagem original
        st.image(image, caption='Imagem Original', channels="BGR")
        
        # Detectando objetos na imagem
        results = model(image)
        
        # Processando detecções
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Verificando se é uma placa (classe 0 geralmente é para veículos)
                if box.cls == 0:  
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
                        
                        # Buscando informações do veículo
                        vehicle_info = get_vehicle_info(plate_text)
                        
                        # Preparando dados para salvar
                        detection_data = {
                            'timestamp': datetime.now(),
                            'placa': plate_text,
                            'thumbnail_path': thumbnail_path,
                            'confianca': float(box.conf[0]),
                            'marca': vehicle_info.get('marca', 'N/A') if vehicle_info else 'N/A',
                            'modelo': vehicle_info.get('modelo', 'N/A') if vehicle_info else 'N/A',
                            'ano': vehicle_info.get('ano', 'N/A') if vehicle_info else 'N/A'
                        }
                        
                        detections.append(detection_data)
                        
                        # Desenhando retângulo na imagem
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, plate_text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Mostrando imagem com detecções
        st.image(image, caption='Detecções', channels="BGR")
        
        # Salvando detecções
        if detections:
            save_to_csv(detections)
            
            # Mostrando tabela de detecções
            st.subheader("Detecções Realizadas")
            df = pd.DataFrame(detections)
            st.dataframe(df)
        else:
            st.warning("Nenhuma placa detectada na imagem.")

if __name__ == "__main__":
    main()
