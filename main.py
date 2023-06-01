from ANALISIS.analisis_POO import *
from ANALISIS.modelo_regresion import *
from ANALISIS.forest import *

import helpers
from menu import Menu


def limpieza():
    analisis.show()

def regresion():
    modelo.show_modelo_regresion()

def arboles():
    model.show_all()

#
#   I N I C I O    P R O G R A M A
#

helpers.clear()  # Limpia la terminal

mi_menu = Menu("ANALISIS DE DATOS")
mi_menu.addOption("Analisis de datos", limpieza)
mi_menu.addOption("Modelo de regresion", regresion)
mi_menu.addOption("Modelo de arboles", arboles)




                  
