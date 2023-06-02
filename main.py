from ANALISIS.analisis_POO import *
from ANALISIS.modelo_regresion import *
from ANALISIS.forest import *

import helpers
from menu import Menu


def limpieza():
    analisis.show()
    
    nfts = GestionNFT()
    nfts.comprar_nft("NFT 1")
    nfts.comprar_nft("NFT 2")
    nfts.comprar_nft("NFT 3")
    nfts.mostrar_nfts()

    nfts.vender_nft("NFT 2")
    nfts.mostrar_nfts()

    # Cálculo del valor de Bitcoin en euros
    bitcoin_amount = 0.5
    bitcoin_value_euros = 40000
    euros_value = analisis.bitcoinToEuros(bitcoin_amount, bitcoin_value_euros)
    print("El valor de Bitcoin en euros es:", euros_value)

    # Correlación del valor de Bitcoin en euros con los datos de tráfico aéreo
    if euros_value < 30000:
        print("¡Alerta! El valor de Bitcoin está por debajo de 30,000€.")
    else:
        print("El valor de Bitcoin está por encima de 30,000€.")

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




                  
