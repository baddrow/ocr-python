from flask import Flask, request, jsonify
import numpy as np
import urllib.request
import cv2
import json

app = Flask(__name__)

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def detectar_marcacao(rect, idx, image, alternativas, respostas):
    x, y, w, h = cv2.boundingRect(rect)
    cropped = image[y:y+h, x:x+w]
    
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binarized = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    
    height, width = binarized.shape
    
    num_questoes = 10
    num_alternativas = 5
    margem_vertical = 60
    margem_horizontal = 40
    ajuste_horizontal = 10
    ajuste_vertical = -25
    ajuste_altura_questao = 5
    
    questao_height = (height - 2 * margem_vertical) // num_questoes + ajuste_altura_questao
    alternativa_width = (width - 2 * margem_horizontal) // num_alternativas
    
    for i in range(num_questoes):
        y_inicio = margem_vertical + i * questao_height + ajuste_vertical
        y_fim = y_inicio + questao_height
        
        preenchimento_alternativas = []
        for j in range(num_alternativas):
            x_inicio = margem_horizontal + j * alternativa_width + ajuste_horizontal
            x_fim = x_inicio + alternativa_width
            
            mask = np.zeros_like(binarized, dtype=np.uint8)
            cv2.rectangle(mask, (x_inicio, y_inicio), (x_fim, y_fim), 255, -1)
            
            total_preto = cv2.countNonZero(cv2.bitwise_and(binarized, mask))
            total_pixels = (y_fim - y_inicio) * (x_fim - x_inicio)
            percentual_preenchido = (total_preto / total_pixels) * 100
            preenchimento_alternativas.append(percentual_preenchido)
        
        if max(preenchimento_alternativas) > 30:
            index_alternativa_marcada = np.argmax(preenchimento_alternativas)
            alternativa_marcada = alternativas[index_alternativa_marcada]
            respostas[f"{idx * 10 + i + 1}"] = alternativa_marcada
            
            for j in range(num_alternativas):
                x_inicio = margem_horizontal + j * alternativa_width + ajuste_horizontal
                x_fim = x_inicio + alternativa_width
                color = (255, 0, 0) if j == index_alternativa_marcada else (0, 255, 0)
                cv2.rectangle(cropped, (x_inicio, y_inicio), (x_fim, y_fim), color, 2)
        else:
            respostas[f"{idx * 10 + i + 1}"] = "Não detectada"

@app.route('/process', methods=['GET'])
def hello_http():
    try:
        url = 'https://static.wixstatic.com/media/3a8958_0edd9be7961645f4b6516403348095e3~mv2.jpg'
        image = url_to_image(url)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binarized = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 8000 < area < 300000:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    rectangles.append(approx)

        rectangles = sorted(rectangles, key=lambda x: x[0][0][0])

        alternativas = ['A', 'B', 'C', 'D', 'E']
        respostas = {}
        for idx, rect in enumerate(rectangles):
            detectar_marcacao(rect, idx, image, alternativas, respostas)

        lista = []
        for questao, resposta in respostas.items():
            lista.append({
                'questao': questao,
                'resposta': resposta
            })

        return jsonify(lista), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
