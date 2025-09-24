import numpy as np
import matplotlib.pyplot as plt

# Perceptron Simples (para prever 1 saída por vez)
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, learning_rate=0.1, epochs=100):
        error_history = []
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                z = np.dot(X[i], self.weights) + self.bias
                output = self.sigmoid(z)
                error = y[i] - output
                total_error += error ** 2

                # Atualiza os pesos
                self.weights += learning_rate * error * self.sigmoid_derivative(output) * X[i]
                self.bias += learning_rate * error * self.sigmoid_derivative(output)
            error_history.append(total_error)
        return error_history

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        output = self.sigmoid(z)
        return output


# 🔧 Dados de treino fornecidos
# Piso: 1 (Carpete), 2 (Cerâmica), 3 (Madeira)
dados_treino = [
    {"piso": 2, "poeira": 2, "obstaculos": 0, "potencia": 1, "velocidade": 3},
    {"piso": 1, "poeira": 8, "obstaculos": 2, "potencia": 3, "velocidade": 1},
    {"piso": 3, "poeira": 5, "obstaculos": 4, "potencia": 2, "velocidade": 1},
    {"piso": 2, "poeira": 1, "obstaculos": 1, "potencia": 1, "velocidade": 4},
    {"piso": 1, "poeira": 9, "obstaculos": 3, "potencia": 3, "velocidade": 2},
    {"piso": 3, "poeira": 6, "obstaculos": 0, "potencia": 2, "velocidade": 3},
    {"piso": 2, "poeira": 3, "obstaculos": 2, "potencia": 1, "velocidade": 2},
    {"piso": 1, "poeira": 7, "obstaculos": 1, "potencia": 3, "velocidade": 1},
    {"piso": 3, "poeira": 4, "obstaculos": 3, "potencia": 2, "velocidade": 4},
    {"piso": 2, "poeira": 0, "obstaculos": 0, "potencia": 1, "velocidade": 5}
]

# Prepara os dados
X = []
y_potencia = []
y_velocidade = []

for item in dados_treino:
    X.append([item["piso"], item["poeira"], item["obstaculos"]])
    y_potencia.append(item["potencia"] / 3)     # Normaliza entre 0 e 1
    y_velocidade.append(item["velocidade"] / 5) # Normaliza entre 0 e 1

X = np.array(X)
y_potencia = np.array(y_potencia)
y_velocidade = np.array(y_velocidade)

# Cria os dois perceptrons
potencia_model = Perceptron(input_size=3)
velocidade_model = Perceptron(input_size=3)

# Treina
erro_pot = potencia_model.train(X, y_potencia, learning_rate=0.1, epochs=500)
erro_vel = velocidade_model.train(X, y_velocidade, learning_rate=0.1, epochs=500)

# Teste com uma nova entrada
entrada = [2, 5, 1]  # Piso: cerâmica, poeira: 5, obstáculos: 1

potencia_pred = potencia_model.predict(entrada) * 3
velocidade_pred = velocidade_model.predict(entrada) * 5

print(f"Entrada: Piso=2 (Cerâmica), Poeira=5, Obstáculos=1")
print(f"Potência prevista: {round(potencia_pred, 2)}")
print(f"Velocidade prevista: {round(velocidade_pred, 2)}")

# Gráfico (bônus)
plt.plot(erro_pot, label='Erro Potência')
plt.plot(erro_vel, label='Erro Velocidade')
plt.xlabel("Épocas")
plt.ylabel("Erro Quadrático")
plt.title("Erro durante o Treinamento")
plt.legend()
plt.grid()
plt.show()
