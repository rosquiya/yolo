from ultralytics import YOLO
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def main():
    # Cargar el modelo preentrenado YOLO
    model_path = './runs/classify/train/weights/best.pt'
    model = YOLO(model_path)
    
    # Directorio de datos de validación
    val_data_dir = r'C:\Users\Rosario\Videos\ultimo_aliento\tesis_2\data\data_ros_aug\val'
    
    # Listas para almacenar etiquetas verdaderas y predichas
    true_labels = []
    predicted_labels = []
    
    # Recorrer las imágenes en el directorio de validación
    for clase in os.listdir(val_data_dir):
        clase_path = os.path.join(val_data_dir, clase)
        
        if os.path.isdir(clase_path):
            for img_file in os.listdir(clase_path):
                img_path = os.path.join(clase_path, img_file)
                
                # Hacer predicciones con YOLO
                results = model(img_path)
                
                # Iterar sobre cada objeto Results en la lista results
                for result in results:
                    names_dict = result.names
                    probs = result.probs.data.tolist()
                    max_prob_index = np.argmax(probs)
                    
                    # Guardar etiquetas verdaderas y predichas
                    true_labels.append(clase)
                    predicted_labels.append(names_dict[max_prob_index])
    
    # Generar la matriz de confusión
    class_labels = sorted(os.listdir(val_data_dir))  # Obtener las clases de las carpetas
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
    
    # Calcular métricas
    acc = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    
    # Calcular la especificidad para cada clase
    specificity_list = []
    for i in range(len(class_labels)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp)
        specificity_list.append(specificity)
    specificity = np.mean(specificity_list)
    
    # Imprimir las métricas
    print(f"Exactitud (Accuracy): {acc * 100:.2f}%")
    print(f"Sensibilidad (Recall): {recall * 100:.2f}%")
    print(f"Precisión (Precision): {precision * 100:.2f}%")
    print(f"Especificidad (Specificity): {specificity * 100:.2f}%")

if __name__ == '__main__':
    main()