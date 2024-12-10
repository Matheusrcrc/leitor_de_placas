# Sistema de Detecção de Placas de Veículos

Um sistema de detecção e reconhecimento de placas de veículos brasileiros utilizando Python, Streamlit, YOLOv8 e EasyOCR. O sistema é capaz de detectar e reconhecer tanto placas no formato antigo (ABC-1234) quanto no novo padrão Mercosul (ABC1D23).

![Exemplo de Detecção](docs/images/example.png)

## Características

- Interface web amigável usando Streamlit
- Suporte para placas antigas e padrão Mercosul
- Processamento de imagem avançado para melhor reconhecimento
- Registro de detecções com timestamp e nível de confiança
- Exportação de dados em formato CSV
- Processamento em tempo real
- Visualização da detecção na imagem

## Requisitos

```bash
Python 3.8+
OpenCV
Streamlit
YOLOv8
EasyOCR
Pandas
NumPy
```

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/leitor-de-placas.git
cd leitor-de-placas
```

2. Crie um ambiente virtual (opcional, mas recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

1. Inicie a aplicação:
```bash
streamlit run app.py
```

2. Acesse a interface web no seu navegador (geralmente http://localhost:8501)

3. Use a interface para:
   - Fazer upload de imagens
   - Ajustar o limite de confiança
   - Visualizar detecções
   - Baixar registros em CSV

## Como Funciona

O sistema utiliza uma abordagem em múltiplas etapas:

1. **Pré-processamento da Imagem**
   - Conversão para escala de cinza
   - Equalização de histograma adaptativa
   - Redução de ruído
   - Binarização adaptativa

2. **Detecção de Placas**
   - Análise da imagem completa
   - Busca por regiões candidatas baseada em:
     - Detecção de bordas
     - Proporção de área
     - Aspect ratio típico de placas

3. **OCR e Validação**
   - Reconhecimento de caracteres usando EasyOCR
   - Validação de formato de placa
   - Correção de caracteres comumente confundidos

## Estrutura do Projeto

```
leitor-de-placas/
├── app.py              # Aplicação principal
├── requirements.txt    # Dependências do projeto
├── README.md          # Este arquivo
├── docs/              # Documentação
│   └── images/        # Imagens para documentação
└── thumbnails/        # Pasta para thumbnails salvos
```

## Contribuindo

Contribuições são bem-vindas! Por favor, siga estes passos:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Configuração Avançada

### Ajuste de Parâmetros

O sistema possui vários parâmetros que podem ser ajustados para melhorar o desempenho em diferentes cenários:

```python
# Limite de confiança para detecção (ajustável via interface)
confidence_threshold = 0.2

# Parâmetros de pré-processamento
min_area = width * height * 0.01  # Área mínima da placa
max_area = width * height * 0.15  # Área máxima da placa
aspect_ratio_range = (1.5, 4.5)   # Proporção esperada da placa
```

### Personalização do OCR

O sistema usa EasyOCR com suporte para português e inglês. Você pode adicionar mais idiomas ou ajustar os parâmetros:

```python
reader = easyocr.Reader(['en', 'pt'])  # Adicione mais idiomas conforme necessário
```

## Problemas Conhecidos

- O desempenho pode variar dependendo da qualidade da imagem
- Placas muito inclinadas podem não ser detectadas
- O processamento pode ser mais lento em CPUs menos potentes

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE.md](LICENSE.md) para detalhes.

## Contato

Seu Nome - [@seu_twitter](https://twitter.com/seu_twitter)

Link do Projeto: [https://github.com/seu-usuario/leitor-de-placas](https://github.com/seu-usuario/leitor-de-placas)
