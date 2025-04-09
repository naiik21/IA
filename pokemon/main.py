from flask import Flask, render_template, request, jsonify
from PIL import Image  # Importa Pillow para manejar imágenes
import torch
from torchvision import transforms
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights

# Cargar el dataset para obtener el mapeo de clases
train_dataset = ImageFolder("./data/train", transform=None)  # Solo necesitas cargarlo una vez
classes = train_dataset.classes  # Diccionario inverso


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

        self.linear_relu_stack = nn.Sequential(

            # nn.Conv2d(in_channels=3,out_channels=64, kernel_size=5, stride=3), 
            # nn.MaxPool2d(3), 
            # nn.LeakyReLU(0.1),
            # nn.Conv2d(64, 128, 3, stride=2), 
            # # nn.MaxPool2d(2), 
            # # nn.ReLU(),
            # nn.Conv2d(128, 128, 3), 
            # nn.MaxPool2d(2), 
            # nn.LeakyReLU(0.1),
            # nn.Conv2d(128, 300, 3),
            # #nn.MaxPool2d(2), 
            # #nn.ReLU(),
            # #nn.Conv2d(300, 300, 3), 
            # #nn.LeakyReLU(0.1),
            
            # nn.Flatten(),          
            resnet18(ResNet18_Weights), 
            # nn.Linear(1200 , 10000),
            # nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(1000 , 5000),
            nn.LeakyReLU(0.1),
            nn.Linear(5000 , 1000),


        )

    def forward(self, x):

        logits = self.linear_relu_stack(x)
        return logits
    

# Ruta para cargar el formulario
@app.route('/')
def home():
    return render_template('index.html')  # Renderiza el archivo HTML


@app.route('/predict', methods=['POST'])
def predict():
    if 'pokemonImage' not in request.files:
        return "No se envió ninguna imagen", 400
    
    # Obtén el archivo de imagen
    image = request.files['pokemonImage']
    
    try:
        # Abre la imagen usando Pillow
        img = Image.open(image).convert('RGB')  # Asegúrate de convertirla a RGB
        
        # Define las transformaciones
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Redimensionar a 224x224 píxeles
            transforms.ToTensor(),         # Convertir la imagen a un tensor
        ])
        
        # Aplica las transformaciones
        input_tensor = transform(img).unsqueeze(0).to(device)  # Agrega una dimensión batch
        
        # Cargar el modelo
        model = NN_model()
        model.load_state_dict(torch.load('modelRes.pth', map_location=device))  # Carga los pesos
        model.to(device)  # Mueve el modelo al mismo dispositivo que los datos
        model.eval()  # Cambiar a modo evaluación
        
        # Realizar predicción
        with torch.no_grad():
            logits = model(input_tensor)
            prediccion = torch.argmax(logits, dim=1).item()  # Obtener la clase predicha
            
        print(prediccion)
        
        # Retornar la predicción
        return jsonify({'prediccion': classes[prediccion]})
    
    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
