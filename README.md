# Sistema de Detecção de Placas de Veículos

Um sistema de reconhecimento automático de placas de veículos brasileiros (ALPR - Automatic License Plate Recognition) que suporta tanto o formato antigo quanto o novo padrão Mercosul. Desenvolvido com Python, utiliza tecnologias modernas como Streamlit para interface web, EasyOCR para reconhecimento de caracteres e OpenCV para processamento de imagens.

## Características

- Interface web intuitiva desenvolvida com Streamlit
- Suporte completo para placas no formato Mercosul
- Conversão automática do formato antigo para Mercosul
- Processamento de imagem avançado para melhor reconhecimento
- Registro de detecções com timestamp e nível de confiança
- Exportação de dados em formato CSV
- Gerenciamento eficiente de recursos e memória
- Validação conforme regras oficiais do Denatran

## Formatos de Placas Suportados

1. **Formato Antigo:**
   - Padrão: ABC-1234
   - Três letras seguidas de quatro números

2. **Formato Mercosul:**
   - Padrão: ABC1D23
   - Letras nas posições 1-3
   - Número na posição 4
   - Letra ou número na posição 5 (conforme tabela de conversão)
   - Números nas posições 6-7

## Requisitos

```bash
Python 3.8+
OpenCV
Streamlit
EasyOCR
NumPy
Pandas
Pillow
PyTorch
```

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/leitor-de-placas.git
cd leitor-de-placas
```

2. Crie um ambiente virtual (recomendado):
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

2. Acesse a interface web (geralmente http://localhost:8501)

3. Funcionalidades disponíveis:
   - Upload de imagens de placas
   - Ajuste do nível de confiança da detecção
   - Visualização em tempo real das detecções
   - Download dos registros em CSV
   - Limpeza de cache e recursos

## Como Funciona

O sistema utiliza uma abordagem em múltiplas etapas:

1. **Pré-processamento da Imagem**
   - Compressão e otimização
   - Conversão de espaço de cores
   - Equalização de histograma adaptativo
   - Redução de ruído

2. **Detecção de Placas**
   - Identificação de regiões candidatas
   - Análise de proporções e características
   - Validação de formato

3. **OCR e Validação**
   - Reconhecimento de caracteres com EasyOCR
   - Validação conforme regras Mercosul
   - Conversão automática quando necessário

4. **Pós-processamento**
   - Correção de caracteres comumente confundidos
   - Validação final do formato
   - Registro dos resultados

## Estrutura do Projeto

```
leitor-de-placas/
├── app.py              # Aplicação principal
├── requirements.txt    # Dependências
├── README.md          # Documentação
├── docs/              # Documentação adicional
│   └── images/        # Imagens e exemplos
└── data/              # Dados e registros
    └── thumbnails/    # Miniaturas das detecções
```

## Recursos Avançados

### Tabela de Conversão Mercosul

Conversão automática do formato antigo para Mercosul seguindo a tabela oficial:
```
0 = A    5 = F
1 = B    6 = G
2 = C    7 = H
3 = D    8 = I
4 = E    9 = J
```

### Configurações Ajustáveis

```python
# Limite de confiança para detecção
confidence_threshold = 0.2  # Ajustável via interface

# Parâmetros de pré-processamento
min_area = width * height * 0.01  # Área mínima da placa
max_area = width * height * 0.15  # Área máxima da placa
aspect_ratio_range = (1.5, 4.5)   # Proporção esperada
```

## Resolução de Problemas

1. **Problemas de Memória:**
   - Use o botão "Limpar Cache" na barra lateral
   - Reinicie a aplicação se necessário
   - Monitore o uso de recursos

2. **Falhas na Detecção:**
   - Ajuste o limite de confiança
   - Verifique a qualidade da imagem
   - Confirme se a placa está visível e legível

3. **Erros de OCR:**
   - Verifique a iluminação da imagem
   - Certifique-se que a placa está em bom estado
   - Tente diferentes ângulos da placa

## Contribuindo

1. Fork do projeto
2. Crie sua branch de feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE.md](LICENSE.md) para detalhes.

## Contato

Seu Nome - [@seu_twitter](https://twitter.com/seu_twitter)

Link do Projeto: [https://github.com/seu-usuario/leitor-de-placas](https://github.com/seu-usuario/leitor-de-placas)
