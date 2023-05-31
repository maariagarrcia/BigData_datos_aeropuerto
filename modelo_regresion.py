from dask import dataframe as dd
from colorama import Fore
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix,roc_curve, roc_auc_score, auc

#from analisis_POO import *

class RegressionModel:
    def __init__(self, filename):
        self.filename = filename
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self):
        self.df = dd.read_csv(self.filename)

    def prepare_data(self):
        # Seleccionar las columnas relevantes para el modelo
        df = self.df[['Month', 'Passenger Count']]

        # Convertir el dataframe Dask a pandas dataframe
        df = df.compute()

        # Dividir los datos en conjunto de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df['Month'], df['Passenger Count'], test_size=0.2, random_state=42)

    def train_model(self):
        # Crear el modelo de regresión lineal
        self.model = LinearRegression()

        # Ajustar el modelo a los datos de entrenamiento
        self.model.fit(self.X_train.values.reshape(-1, 1), self.y_train)

    def evaluate_model(self):
        # Realizar predicciones en el conjunto de prueba
        y_pred = self.model.predict(self.X_test.values.reshape(-1, 1))

        # Calcular el error cuadrático medio en el conjunto de prueba
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(Fore.CYAN+"Error cuadrático medio:"+Fore.WHITE, mse)
        print()
        print(Fore.CYAN+"Error absoluto medio:"+Fore.WHITE, mae)
        print()
        print(Fore.CYAN+"Coeficiente de determinación (R^2):"+Fore.WHITE, r2)

    def generate_predictions(self):
        # En este caso específico, se definen nuevos datos en forma de una lista de meses: ['May', 'June', 'July']. 
        # Estos nuevos datos se utilizan como entrada para la función predict del modelo de regresión lineal, 
        # después de aplicar la transformación de etiquetas utilizando LabelEncoder. La función fit_transform se 
        # utiliza para ajustar y transformar los datos en función de las etiquetas existentes en el modelo. Luego,
        #  se realiza una transformación adicional con reshape(-1, 1) para darle la forma adecuada al conjunto de datos de entrada.

        # Generar predicciones para nuevos datos
        new_data = ['May', 'June', 'July']  # Ejemplo de nuevos datos
        predictions = self.model.predict(LabelEncoder().fit_transform(new_data).reshape(-1, 1))
        print(Fore.CYAN+"Predicciones:"+Fore.WHITE, predictions)

    def classification_report(self):
        # El classification_report es una función o método que proporciona un informe completo de las métricas 
        # de evaluación del rendimiento de un modelo de clasificación. Está disponible en la mayoría de las 
        # bibliotecas de aprendizaje automático, como scikit-learn.

        # Precisión (Precision): La proporción de predicciones positivas correctas (TP) sobre el total de predicciones 
        # positivas realizadas (TP + FP). Mide la capacidad del modelo para no etiquetar incorrectamente una instancia negativa como positiva.
        
        # Recall (Recall): La proporción de instancias positivas correctamente detectadas (TP) sobre el total de instancias 
        # positivas reales (TP + FN). Mide la capacidad del modelo para encontrar todas las instancias positivas.
        
        # F1-Score: Es una medida que combina precisión y recall en un único valor que resume el rendimiento del modelo.
        # Se calcula como la media armónica de la precisión y el recall. El F1-score es útil cuando hay un desequilibrio entre las clases.
       
        # Support: El número de instancias de cada clase en los datos de prueba.

        #cEl classification_report proporciona estas métricas para cada clase en el problema de clasificación, 
        # así como también calcula un promedio ponderado y promedios macro y micro de estas métricas para obtener 
        # una visión general del rendimiento del modelo en general.


        # Promedio Macro (Macro Average): El promedio macro calcula las métricas de evaluación para cada clase por separado y 
        # luego toma el promedio de esas métricas. Esto significa que todas las clases tienen el mismo peso en el cálculo del 
        # promedio macro. Es útil cuando todas las clases tienen igual importancia y se desea evaluar el rendimiento general 
        # del modelo sin tener en cuenta el desequilibrio entre las clases.

        # Promedio Micro (Micro Average): El promedio micro calcula las métricas de evaluación agregando las cantidades 
        # verdaderos positivos, falsos positivos y falsos negativos de todas las clases y luego calcula las métricas a 
        # partir de esas cantidades totales. En otras palabras, el promedio micro considera todas las predicciones y 
        # resultados verdaderos de todas las clases como una sola entidad. Es útil cuando se quiere evaluar el rendimiento 
        # global del modelo y se tiene en cuenta el desequilibrio entre las clases.
        # Convertir el problema de regresión en un problema de clasificación

        threshold = self.y_train.mean()  # Valor umbral para clasificar como 0 o 1
        y_train_class = (self.y_train > threshold).astype(int)
        y_test_class = (self.y_test > threshold).astype(int)

        # Calcular el classification_report
        y_pred_class = (self.model.predict(self.X_test.values.reshape(-1, 1)) > threshold).astype(int)
        report = classification_report(y_test_class, y_pred_class)

        print(Fore.CYAN+"Classification Report:"+Fore.WHITE)
        print(report)

    def confusion_matrix(self):
        # Representa el rendimiento de un modelo clasificador mediante la visualización de la cantidad 
        # de muestras que fueron clasificadas correctamente y las que fueron clasificadas incorrectamente en cada una de las clases.

        # La matriz de confusión es una tabla cuadrada que muestra en sus filas las clases reales y en sus columnas las 
        # clases predichas por el modelo. Cada celda de la matriz representa el recuento de muestras que fueron 
        # clasificadas en esa combinación de clase real y clase predicha.

        #                   Predicted Positive    Predicted Negative
        #    Actual Positive      TP                   FN
        #    Actual Negative      FP                   TN



        # Convertir el problema de regresión en un problema de clasificación
        threshold = self.y_train.mean()  # Valor umbral para clasificar como 0 o 1
        y_train_class = (self.y_train > threshold).astype(int)
        y_test_class = (self.y_test > threshold).astype(int)

        y_pred_class = (self.model.predict(self.X_test.values.reshape(-1, 1)) > threshold).astype(int)


        # Calcular la matriz de confusión
        cm = confusion_matrix(y_test_class, y_pred_class)

        # Crear la figura y el eje de la matriz de confusión
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

        # Configurar las etiquetas de los ejes
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        # Configurar las etiquetas de las clases
        classes = [0, 1]  # Clases posibles en el problema de clasificación binaria
        tick_marks = range(len(classes))
        ax.set_xticks([t + 0.5 for t in tick_marks])
        ax.set_xticklabels(classes)
        ax.set_yticks([t + 0.5 for t in tick_marks])
        ax.set_yticklabels(classes)

        # Mostrar la matriz de confusión
        plt.show()

    def modelo_regresion(self):
        self.load_data()
        self.prepare_data()
        self.train_model()
        self.evaluate_model()
        print()
        self.generate_predictions()
        print()
        self.classification_report()
        print()
        self.confusion_matrix()


# Instanciar el objeto RegressionModel
model = RegressionModel('/Users/mariagarcia/Documents/BigData_datos_aeropuerto/dataset_limpiado.csv')
model.modelo_regresion()

