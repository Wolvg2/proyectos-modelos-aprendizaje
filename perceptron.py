import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

"""
    Implementación de un perceptron simple en Python para entrenar un modelo de clasificación binaria.
    
    Implementado a partir de la solucion de Kaggle:
    https://www.kaggle.com/code/kuzagaya/implementing-a-perceptron-in-python
 
"""

# Cargar el dataset de entrenamiento
pd.set_option('display.max_columns', None)
df = pd.read_csv('train.csv')

# Seleccionar las características de entrada y la variable objetivo
X = df[['battery_power', 'ram', 'int_memory','touch_screen']]
y = (df['price_range'] >= 2).astype(int)

# Convertir los datos a arreglos de numpy para facilitar el cálculo
X_arr = X.to_numpy()
y_arr = y.to_numpy()

# Normalizar los datos de entrada para mejorar la convergencia del modelo
X_norm = (X_arr - X_arr.mean(axis=0)) / X_arr.std(axis=0)


# lr -> learning rate
# epochs -> numero de epocas
# num_features -> numero de caracteristicas de entrada
def train_perceptron(X_norm, y_arr, lr=0.01, epochs=100):
    num_samples, num_features = X_norm.shape

    weights = np.zeros(num_features)
    bias = 0

    # Guardar los errores por epoca para analizar el desempeño del modelo
    err_epochs = []

    for epoch in range(epochs):
        err_in_epoch = 0

        for i in range(num_samples):
            # Calcular la linea de activacion 
            active_function = np.dot(X_norm[i], weights) + bias

            # Aplicar la funcion de activacion (step function)
            if active_function >= 0:
                y_pred = 1
            else:
                y_pred = 0

            # Calcular el error
            error = y_arr[i] - y_pred

            if error != 0:
                err_in_epoch += 1

            # Actualizar los pesos y el bias
            bias += lr * error
            for j in range(num_features):
                weights[j] += lr * error * X_norm[i][j]
        err_epochs.append(err_in_epoch)
    return weights, bias, err_epochs

# Predecir la clase de un nuevo ejemplo utilizando los pesos y bias entrenados
def predict(X_norm, weights, bias):
    active_function = np.dot(X_norm, weights) + bias
    predictions = (active_function >= 0).astype(int)
    
    return predictions

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
def train_test_split(X_norm, y_arr, test_size=0.2):
    num_samples = X_norm.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    test_count = int(num_samples * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = X_norm[train_indices]
    y_train = y_arr[train_indices]
    X_test = X_norm[test_indices]
    y_test = y_arr[test_indices]

    return X_train, X_test, y_train, y_test, test_indices

#Confusion matriz
def plot_confusion_matrix(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i][j], ha='center', va='center', color='black')

    plt.show()



# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test, test_indices = train_test_split(X_norm, y_arr, test_size=0.2)
X_test_original = X_arr[test_indices]  # las mismas filas, pero sin normalizar
weights, bias, err_epochs = train_perceptron(X_train, y_train)
predictions = predict(X_test, weights, bias)




def main():
    plot_confusion_matrix(y_test, predictions)
    print("Accuracy:",accuracy_score(y_test, predictions))
    print("Precisión:",precision_score(y_test, predictions))
    print("Recall:",recall_score(y_test, predictions))
    print("F1-Score:",f1_score(y_test, predictions))
    print("Reporte de Clasificación:\n", classification_report(y_test, predictions))
    df_ejemplos = pd.DataFrame(X_test_original[:10], columns=['battery_power','ram','int_memory','touch_screen'])
    df_ejemplos['Predicción'] = predictions[:10]
    df_ejemplos['Real'] = y_test[:10]
    print(df_ejemplos)


main()