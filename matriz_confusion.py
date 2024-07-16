from ultralytics import YOLO
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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
    
    # Calcular accuracy
    acc = accuracy_score(true_labels, predicted_labels)
    
    # Mostrar la matriz de confusión con seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, cbar=False)
    
    # Añadir porcentajes a la matriz de confusión debajo de los números existentes
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            plt.text(j + 0.5, i + 0.7, "{:.2f}%".format(cm[i, j] / np.sum(cm[i]) * 100), ha='center', va='center', fontsize=8)
    
    # Ajustes adicionales
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.xlabel('Predicciones')
    plt.ylabel('Etiquetas Verdaderas', rotation=90)
    plt.title('Matriz de Confusión YoloV8\nAccuracy: {:.2f}%'.format(acc * 100))
    plt.tight_layout()
    
    # Guardar la imagen de la matriz de confusión
    plt.savefig('matriz_confusion_yolov8.png')
    
    plt.show()

if __name__ == '__main__':
    main()
