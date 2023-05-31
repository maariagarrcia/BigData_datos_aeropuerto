from dask import dataframe as dd
from colorama import Fore
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Data Cleaning ---> Este proceso ayuda a obtener resultados confiables de los datos,
# identificando y corrigiendo errores en los datos.

# Como podemos oobservar tenemos un dataset de un conjunto de datos sobre el tráfico en el aeropuerto
# de San Francisco.




class AnalizarDatos:
    def __init__(self, file_path):
        self.df = dd.read_csv(file_path)

    def InitialAnalysis(self):
        # numero de filas y columnas
        print(Fore.LIGHTMAGENTA_EX +
              "· CARACTERÍSTICAS DE LOS DATOS ·" + Fore.WHITE)
        print(Fore.LIGHTCYAN_EX+"Filas del dataframe: " +
              Fore.WHITE, len(self.df))
        # columnas
        print(Fore.LIGHTCYAN_EX+"Columnas del dataframe: " +
              Fore.WHITE, len(self.df.columns))
        print()

    def dtypes(self):
        # vamos a ver los datos por columnas que tenemos
        print(Fore.LIGHTMAGENTA_EX + "· DATOS POR COLUMNAS ·" + Fore.WHITE)
        print(self.df.columns)
        print()

        # vamos a ver que tipo de datos tenemos
        print(Fore.LIGHTMAGENTA_EX + "· TIPOS DE DATOS ·" + Fore.WHITE)
        print(self.df.dtypes)
        print()

        # ahora separaremos los datos dependiendo de su tipo
        df_num = self.df.select_dtypes(include=['int64', 'float64'])
        df_cat = self.df.select_dtypes(include=['object'])
        df_bool = self.df.select_dtypes(include=['bool'])
    

        # vamos a ver los datos numericos
        print(Fore.LIGHTMAGENTA_EX + "· DATOS NUMÉRICOS ·" + Fore.WHITE)
        print(df_num.head())
        print()
        # vamos a ver los datos categoricos
        print(Fore.LIGHTMAGENTA_EX + "· DATOS CATEGÓRICOS ·" + Fore.WHITE)
        print(df_cat.head())

        # vamos a ver los datos booleanos
        print(Fore.LIGHTMAGENTA_EX + "· DATOS BOOLEANOS ·" + Fore.WHITE)
        print(df_bool.head())
        print()
   
    def convert_categorical_to_numeric(self):
        categorical_columns = self.df.select_dtypes(include='object').columns
        for col in categorical_columns:
            categories = self.df[col].unique().compute()
            mapping = {cat: i for i, cat in enumerate(categories)}
            self.df[col] = self.df[col].map(mapping)
        
        print(self.df.dtypes)
        #print(self.df.head())

    def save_dataset(self, output_file):
        self.df.to_csv(output_file, index=False, single_file=True)
      
    def OperatingAirlines(self):
        print(Fore.LIGHTMAGENTA_EX + "· COMPAÑÍAS ·" + Fore.WHITE)
        print(self.df['Operating Airline'].unique().compute())
        print()

        #el gráfico de recuento muestra la cantidad de veces que aparece cada compañía operadora en el conjunto de datos. 
        # en el eje x no va asalie el nombre de las compañias porque hemos convertido los datos categoricos a numericos
        # pero antes de la conversion se veia que la compañia que mas vuelos tenia era United Airlines - Pre 07/01/2013
        # Countplot de las compañías operadoras
        sns.set(style="darkgrid")
        plt.figure(figsize=(12, 6))
        plt.xticks(rotation=90)
        plt.title("Compañías Operadoras")
              
        sns.countplot(data=self.df.compute(), x='Operating Airline')
        plt.show()

    def MediaPasajeros(self):
        # ¿Cuántos pasajeros tienen de media los vuelos de cada compañía?
        print(Fore.LIGHTMAGENTA_EX + "· MEDIA DE PASAJEROS ·" + Fore.WHITE)
        medias_registros = self.df.groupby('Operating Airline')[
            'Adjusted Passenger Count'].mean().compute()
        print(medias_registros)
        print()
        return medias_registros

    def EliminarDuplicados(self):
        print(Fore.LIGHTMAGENTA_EX + "· ELIMINAR DUPLICADOS ·" + Fore.WHITE)
        registros_mantenidos = self.df.groupby(
            'GEO Region')['Adjusted Passenger Count'].max().compute()
        print(registros_mantenidos)
        print()
        return registros_mantenidos

    def nuevo_csv(self):
        # crear un nuevo csv  y despues  añadir los datos
        print(Fore.LIGHTMAGENTA_EX + "· CREAR NUEVO CSV ·" + Fore.WHITE)
        registros_mantenidos = self.EliminarDuplicados()
        registros_mantenidos.to_csv('nuevo.csv')
        # añadir una columna con la media de pasajeros
        medias_registros = self.MediaPasajeros()
        self.df['Media Pasajeros'] = self.df['Operating Airline'].map(
            medias_registros)
        print()
    
    def calculos_descriptivos(self):
        #  calcular la media y la desviación estándar de cada elemento del conjunto de datos.
        print(Fore.LIGHTMAGENTA_EX + "· CÁLCULOS DESCRIPTIVOS ·" + Fore.WHITE)
        # TODOS LOS DATOS DEL DATAFRAME CALCULADOS
        print(self.df.describe().compute())
        print()
        # conclusiones de los calculos descriptivos de los años
        print(Fore.LIGHTMAGENTA_EX + "· CONCLUSIONES ·" + Fore.WHITE)
        
        print("Basándonos únicamente en la media y la desviación estándar, podemos extraer las siguientes conclusiones:")
        print()
        print(Fore.CYAN+"1."+Fore.WHITE+" La media de la columna Passenger Count es de aproximadamente 29,241, mientras que la media de la columna Adjusted Passenger Count es de aproximadamente 29,332. Esto indica que, en promedio, la cantidad de pasajeros registrados y la cantidad de pasajeros ajustada son similares.")
        print()
        print(Fore.CYAN+"2."+Fore.WHITE+" La desviación estándar de la columna Passenger Count es de aproximadamente 58,319, mientras que la desviación estándar de la columna Adjusted Passenger Count es de aproximadamente 58,284. Estos valores indican una alta variabilidad en los datos, lo que sugiere que hay una amplia dispersión en la cantidad de pasajeros registrados y ajustados en los períodos de actividad.")
        print()
        print(Fore.CYAN+"3."+Fore.WHITE+" La diferencia entre las medias y las desviaciones estándar de las columnas Passenger Count y Adjusted Passenger Count es mínima, lo que indica que los ajustes realizados no tienen un impacto significativo en la variabilidad de los datos.")
        print()
        print("En resumen, la media y la desviación estándar proporcionan una medida de tendencia central y una medida de dispersión de los datos respectivamente. Estos valores sugieren que hay una variabilidad considerable en la cantidad de pasajeros registrados y ajustados en los períodos de actividad, y que los ajustes realizados no parecen tener un impacto significativo en esta variabilidad.")
        
    def eliminar_nulos(self):
        # vamos a ver el total de nulos
        print(Fore.LIGHTMAGENTA_EX + "· TOTAL DE NULOS ·" + Fore.WHITE)
        print(self.df.isnull().sum().compute())

        print(Fore.LIGHTMAGENTA_EX + "· ELIMINAR NULOS SI HAY MAS DE 10^4·" + Fore.WHITE)
        print()
        # vamos a eliminar las columnas que tengan mas de 10000 nulos
        print(Fore.CYAN+"1."+Fore.WHITE+" Eliminar columnas con mas de 10000 nulos")
        print()
        print('Nº de columnas antes de eliminar: ',len(self.df.columns))
        print(self.df.isnull().sum()[self.df.isnull().sum()>1e4])

        print('Nº de columnas a eliminar: ',len(self.df.isnull().sum()[self.df.isnull().sum()>1e4]))
        print('Nº de columnas después de eliminar: ',len(self.df.columns))

        # si no  se eliminan entonces podemos rellenar los nulos con la media
        print(Fore.CYAN+"2."+Fore.WHITE+" Rellenar los nulos con la media")
        print()
        print('Nº de columnas a rellenar: ',len(self.df.isnull().sum()[self.df.isnull().sum()>0]))

        # como los nulos son categoricos no podemos rellenar con la media ---> 
        # Como son pocos valoores hay dos  oopciones de relleno:
        # rellenarlos con la moda ya que no puede sesgar los resultados ni distorsionar la distribución de los datos.
        # o al ser categoricos podemos rellenarlos con la palabra null 

        self.df = self.df.fillna('null')
        print(self.df.isnull().sum().compute())
  
    #Una vez esto haremos un análisis de la correlación cuyo resultado debe ser una matriz de correlación de datos que represente de qué manera están relacionadas las diferentes variables. Pa
    def correlacion(self):
        print(Fore.LIGHTMAGENTA_EX + "· MATRIZ DE CORRELACIÓN ·" + Fore.WHITE)
        correlation_matrix = self.df.corr().compute()
        print(correlation_matrix)
        print()

        print(Fore.CYAN+"1."+Fore.WHITE+" La correlación entre  columnas que es de aproximadamente 0.999, indica que las dos variables están altamente correlacionadas.")
        print("Esto sugiere que los ajustes realizados no tienen un impacto significativo en la cantidad de pasajeros registrados.")
        print("Los que son menos de 0.5 no estan correlacionados entre si")
        print()

        plt.figure(figsize=(10,8))
        sns.heatmap(self.df.corr().compute(), annot=True, fmt='.1g', cmap='summer')
        plt.title('Matriz de correlación')
        plt.show()

    def remove_highly_correlated_columns(self, threshold=0.5):
        correlation_matrix = self.df.corr().compute()

        columns_to_remove = set()

        for i, column in enumerate(correlation_matrix.columns):
            correlated_columns = correlation_matrix.columns[i+1:]
            high_correlations = correlation_matrix[column][correlated_columns].abs() >= threshold

            for correlated_column in high_correlations[high_correlations].index:
                columns_to_remove.add(correlated_column)

        self.df = self.df.drop(columns=list(columns_to_remove))

        # comprobamos que se han eliminado las columnas
        print(Fore.LIGHTMAGENTA_EX + "· COLUMNAS ELIMINADAS ·" + Fore.WHITE)
        print(columns_to_remove)

    def modelo_regresion(self):

        # Seleccionar las columnas relevantes para el modelo
        df = self.df[['Month', 'Passenger Count']]

        # Convertir el dataframe Dask a pandas dataframe
        df = df.compute()

        # Dividir los datos en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(df['Month'], df['Passenger Count'], test_size=0.2, random_state=42)

        # Crear el modelo de regresión lineal
        model = LinearRegression()

        # Ajustar el modelo a los datos de entrenamiento
        model.fit(X_train.values.reshape(-1, 1), y_train)

        # Realizar predicciones en el conjunto de prueba
        y_pred = model.predict(X_test.values.reshape(-1, 1))

        # Calcular el error cuadrático medio en el conjunto de prueba
        mse = mean_squared_error(y_test, y_pred)

        print("Error cuadrático medio:", mse)

    def show(self):
        #analisis.dtypes()
        analisis.convert_categorical_to_numeric()

        # Guardar el dataset modificado
        output_file = 'datos_modificados.csv'
        analisis.save_dataset(output_file)
        #analisis.OperatingAirlines()
        #analisis.MediaPasajeros()
        #analisis.EliminarDuplicados()
        #analisis.nuevo_csv()
        #analisis.calculos_descriptivos()
        #analisis.eliminar_nulos()
        #analisis.correlacion()
        analisis.remove_highly_correlated_columns()
        # comprobar que se han eliminado las columnas  con la correlacion 
        #analisis.correlacion()
        #analisis.modelo_regresion()
        output_file= "dataset_limpiado.csv"
        analisis.save_dataset(output_file)


      
# Ruta del DataFrame
file_path = '/Users/mariagarcia/Documents/BigData_datos_aeropuerto/air_traffic_data.csv'
analisis = AnalizarDatos(file_path)
analisis.show()
