import onnx
import onnxruntime as ort
import numpy as np
import cv2

# Cargar el modelo ONNX
model_path = 'best.onnx'
session = ort.InferenceSession(model_path)

# Cargar el modelo con onnx para verificar metadatos
onnx_model = onnx.load(model_path)

# Verificar si hay metadatos en el modelo que contengan nombres de clase
def get_class_names_from_onnx(onnx_model):
    class_names = []
    for meta in onnx_model.metadata_props:
        if meta.key == 'names':
            class_names = meta.value.split(',')
            break
    return class_names

# Obtener los nombres de las clases del modelo ONNX
class_names = get_class_names_from_onnx(onnx_model)
print(f'Nombres de las clases: {class_names}')

# Función para preprocesar la imagen
def preprocess_image(image_path, input_shape):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_shape)
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))  # HWC a CHW
    image = np.expand_dims(image, axis=0)  # Añadir dimensión de batch
    image = image / 255.0  # Normalizar
    return image

# Preprocesar la imagen
input_shape = (320, 320)  # Cambiar el tamaño a 320x320
image = preprocess_image('viruela_avanzada_1717384346.jpg', input_shape)

# Obtener el nombre de las entradas y salidas del modelo
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Realizar la predicción
results = session.run([output_name], {input_name: image})[0]

# Postprocesamiento
def postprocess_results(results, class_names):
    probabilities = results[0]
    class_probabilities = {class_names[i]: prob for i, prob in enumerate(probabilities)}
    return class_probabilities

class_probabilities = postprocess_results(results, class_names)
print('Probabilidades de cada clase:')
for class_name, probability in class_probabilities.items():
    print(f'{class_name}: {probability:.4f}')
