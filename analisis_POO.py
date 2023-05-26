from dask import dataframe as dd
from colorama import Fore


class CargarDatos:
    def __init__(self, ruta):
        self.ruta = ruta

    def cargar_datos(self):
        df = dd.read_csv(self.ruta)
        return df


class AnalizarDatos:
    def __init__(self, df):
        self.df = df

    def analisis_inicial(self):
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

        # vamos a ver los datos numericos
        print(Fore.LIGHTMAGENTA_EX + "· DATOS NUMÉRICOS ·" + Fore.WHITE)
        print(df_num.head())
        print()
        # vamos a ver los datos categoricos
        print(Fore.LIGHTMAGENTA_EX + "· DATOS CATEGÓRICOS ·" + Fore.WHITE)
        print(df_cat.head())

    def OperatingAirlines(self):
        print(Fore.LIGHTMAGENTA_EX + "· COMPAÑÍAS ·" + Fore.WHITE)
        print(self.df['Operating Airline'].unique().compute())
        print()

    #¿Cuántos pasajeros tienen de media los vuelos de cada compañía?
    def MediaPasajeros(self):
        print(Fore.LIGHTMAGENTA_EX + "· MEDIA DE PASAJEROS ·" + Fore.WHITE)
        print(self.df.groupby('Operating Airline')['Adjusted Passenger Count'].mean().compute())
        print()






dataframe = CargarDatos(
    '/Users/mariagarcia/Documents/BigData_datos_aeropuerto/air_traffic_data.csv')
analisis = AnalizarDatos(dataframe.cargar_datos())
analisis.analisis_inicial()
# analisis.dtypes()
# analisis.OperatingAirlines()
analisis.MediaPasajeros()
