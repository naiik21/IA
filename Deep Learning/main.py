from flask import Flask, render_template, request, jsonify
import pickle  # para cargar tu modelo entrenado
import torch
from torch import nn

app = Flask(__name__)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



# Define model
class NN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13, 13),
            nn.LeakyReLU(),
            nn.Dropout(0,3),
            nn.Linear(13, 6),
            nn.LeakyReLU(),
            nn.Dropout(0,3),
            nn.Linear(6, 6),
            nn.LeakyReLU(),
            nn.Dropout(0,3),
            nn.Linear(6, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



# Ruta para cargar el formulario
@app.route('/')
def home():
    return render_template('index.html')  # Renderiza el archivo HTML


@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos enviados desde el formulario
    data = request.form
    edad = int(data['age'])
    sexo = int(data['sex'])
    cp = int(data['cp'])
    presion = int(data['trestbps'])
    colesterol = int(data['chol'])
    glucemia = int(data['fbs'])
    ecg = int(data['restecg'])
    frecuencia = int(data['thalach'])
    angina = int(data['exang'])
    oldpeak = float(data['oldpeak'])
    pendiente = int(data['slope'])
    vasos = int(data['ca'])
    tal = int(data['thal'])
    
    # Crear tensor a partir de los datos de entrada
    input_data = torch.tensor([[edad, sexo, cp, presion, colesterol, glucemia, ecg, frecuencia, angina, oldpeak, pendiente, vasos, tal]],
                              dtype=torch.float32).to(device)
    
    # Cargar el modelo
    model = NN_model()
    model.load_state_dict(torch.load('model.pth', map_location=device))  # Carga los pesos
    model.eval()  # Cambiar a modo evaluación
    
    # Realizar predicción
    with torch.no_grad():
        logits = model(input_data)
        prediccion = torch.argmax(logits, dim=1).item()  # Obtener la clase predicha

    # Retornar la predicción
    return jsonify({'prediccion': prediccion})

if __name__ == '__main__':
    app.run(debug=True)
