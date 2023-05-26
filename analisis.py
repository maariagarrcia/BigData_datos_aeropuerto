# vamos a cargar los datos del archivo de datos
# y vamos a hacer un analisis de los datos con dask
# para ver si podemos hacer un analisis de los datos

import dask.dataframe as dd
from colorama import Fore

# vamos a cargar los datos
#df = dd.read_csv('/Users/mariagarcia/Documents/BigData_datos_aeropuerto/air_traffic_data.csv')

#VAMPOS A CARGAR LOS  DATOOS EN UNA FUNCION
def cargar_datos():
    df = dd.read_csv('/Users/mariagarcia/Documents/BigData_datos_aeropuerto/air_traffic_data.csv')
    return df

def analizar_datos(df):
    # vamos a ver los datos por columnas que tenemos
    print(Fore.LIGHTMAGENTA_EX + "·DATOS POR COLUMNAS·" + Fore.WHITE)
    print(df.columns)
    print()

    # vamos a ver que tipo de datos tenemos
    print(Fore.LIGHTMAGENTA_EX+"·TIPOS DE DATOS·"+Fore.WHITE)
    print(df.dtypes)
    print()

    # ahora separaremos los datos dependiendo de su tipo
    #datos numericos
    df_num = df.select_dtypes(include=['int64','float64'])
    # vamos a separar los datos categoricos
    df_cat = df.select_dtypes(include=['object'])

    # vamos a ver los datos numericos
    print(Fore.LIGHTMAGENTA_EX+"·DATOS NUMÉRICS·"+Fore.WHITE)
    print(df_num.head())
    print()
    # vamos a ver los datos categoricos
    print(Fore.LIGHTMAGENTA_EX+"·DATOS CATEGÓRICOS·"+Fore.WHITE)
    print(df_cat.head())



analizar_datos(cargar_datos())