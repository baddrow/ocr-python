import functions_framework
import numpy as np
import urllib.request
import cv2
from matplotlib import pyplot as plt
import json

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

# Função para identificar as respostas no gabarito
def detectar_marcacao(rect, idx, image, alternativas, respostas):
    x, y, w, h = cv2.boundingRect(rect)
    cropped = image[y:y+h, x:x+w]
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # Aumentar o contraste e aplicar desfoque
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar limiar para binarizar a imagem
    _, binarized = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Dimensões do recorte
    height, width = binarized.shape
    
    # Ajuste manual do grid com base nas dimensões do retângulo
    num_questoes = 10  # Número de questões por retângulo
    num_alternativas = 5  # Número de alternativas por questão
    margem_vertical = 60  # Margem superior e inferior para ajustar o grid verticalmente
    margem_horizontal = 40  # Margem esquerda e direita para ajustar o grid horizontalmente
    ajuste_horizontal = 10  # Ajuste adicional para mover o grid para a direita/esquerda
    ajuste_vertical = -25  # Ajuste adicional para mover o grid para cima/baixo
    ajuste_altura_questao = 5  # Ajuste da altura de cada linha do grid
    
    questao_height = (height - 2 * margem_vertical) // num_questoes + ajuste_altura_questao
    alternativa_width = (width - 2 * margem_horizontal) // num_alternativas
    
    # Loop para cada questão
    for i in range(num_questoes):
        y_inicio = margem_vertical + i * questao_height + ajuste_vertical
        y_fim = y_inicio + questao_height
        
        # Loop para cada alternativa
        preenchimento_alternativas = []
        for j in range(num_alternativas):
            x_inicio = margem_horizontal + j * alternativa_width + ajuste_horizontal
            x_fim = x_inicio + alternativa_width
            
            # Criar uma máscara para a região da alternativa
            mask = np.zeros_like(binarized, dtype=np.uint8)
            cv2.rectangle(mask, (x_inicio, y_inicio), (x_fim, y_fim), 255, -1)
            
            # Calcular a quantidade de preto dentro da máscara
            total_preto = cv2.countNonZero(cv2.bitwise_and(binarized, mask))
            total_pixels = (y_fim - y_inicio) * (x_fim - x_inicio)
            percentual_preenchido = (total_preto / total_pixels) * 100
            preenchimento_alternativas.append(percentual_preenchido)
        
        # Determinar qual alternativa está mais preenchida
        if max(preenchimento_alternativas) > 30:  # Ajustar o limite conforme necessário
            index_alternativa_marcada = np.argmax(preenchimento_alternativas)
            alternativa_marcada = alternativas[index_alternativa_marcada]
            respostas[f"{idx * 10 + i + 1}"] = alternativa_marcada
            
            # Desenhar todas as células em verde, exceto a marcada que ficará em azul
            for j in range(num_alternativas):
                x_inicio = margem_horizontal + j * alternativa_width + ajuste_horizontal
                x_fim = x_inicio + alternativa_width
                color = (255, 0, 0) if j == index_alternativa_marcada else (0, 255, 0)  # Azul para a marcada, verde para as demais
                cv2.rectangle(cropped, (x_inicio, y_inicio), (x_fim, y_fim), color, 2)
        else:
            respostas[f"{idx * 10 + i + 1}"] = "Não detectada"

#@functions_framework.http
def hello_http():    
    try:
        # URL da imagem
        url = 'https://static.wixstatic.com/media/e1fca1_51139647635e41eba8f66f1ec4f4d5a2~mv2.jpg'
        # Carregar a imagem
        image = url_to_image(url)
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Aumentar o contraste usando equalização de histograma
        gray = cv2.equalizeHist(gray)
        # Aplicar um desfoque para suavizar a imagem
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Aplicar limiar para binarizar a imagem
        _, binarized = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
        # Aplicar a detecção de contornos para identificar retângulos dos gabaritos
        contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filtrar os contornos para identificar os retângulos principais (gabaritos)
        rectangles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 8000 < area < 300000:  # Ajustar os valores de área conforme necessário para identificar os retângulos dos gabaritos
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Procurar contornos que sejam aproximadamente retangulares
                    rectangles.append(approx)

        # Ordenar os retângulos da esquerda para a direita
        rectangles = sorted(rectangles, key=lambda x: x[0][0][0])

        # Desenhar os retângulos identificados na imagem original
        image_with_rectangles = image.copy()
        for idx, rect in enumerate(rectangles):
            x, y, w, h = cv2.boundingRect(rect)
            cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(image_with_rectangles, f"Retângulo {idx + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Dicionário para armazenar as respostas detectadas
        alternativas = ['A', 'B', 'C', 'D', 'E']
        respostas = {}
        # Aplicar a função de identificação para cada retângulo
        for idx, rect in enumerate(rectangles):
            detectar_marcacao(rect, idx, image, alternativas, respostas)

        lista = []

        # Imprimir as respostas
        for questao, resposta in respostas.items():
            lista.append({
            'questao':questao,
            'resposta':resposta
        })
            print(f"{questao}: Resposta {resposta}")

        # Retornar a lista como JSON
        return json.dumps(lista), 200, {'Content-Type': 'application/json'}
    
    except Exception as e:
        # Retornar uma mensagem de erro em caso de exceção
        return json.dumps({'error': str(e)}), 500, {'Content-Type': 'application/json'}

hello_http()
