
import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"C:\Users\Rosario\Videos\ultimo_aliento\tesis_2\data\modelo_yolov8\runs\classify\train\results copy.csv"
data = pd.read_csv(csv_path)

data.columns = data.columns.str.strip()

print(data.head())

if 'train/loss' in data.columns and 'val/loss' in data.columns:
    epochs = data['epoch']
    train_loss = data['train/loss']
    val_loss = data['val/loss']
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Pérdida de Entrenamiento', marker='o')
    plt.plot(epochs, val_loss, label='Pérdida de Validación', marker='x')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend(loc='upper right')
    plt.title('Pérdida de Entrenamiento y Validación YoloV8')
    plt.grid(True)
    

    plt.savefig('loss_plot_yolov8.jpg')
    plt.show()
else:
    print("El archivo CSV no contiene las columnas necesarias 'train/loss' y 'val/loss'.")
