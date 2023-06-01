#
# Funciones auxiliares
#

from colorama import Fore
import os

# Limpiar la terminal.
# No funciona al ejecutar en ventana interactiva

def underline_it(color_to_use, char_to_use, line_to_underline):
    print(color_to_use + line_to_underline)
    print(char_to_use * len(line_to_underline) + Fore.WHITE)


def clear():
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")


def esperarIntro():
    input(Fore.GREEN + '\nPulsa <INTRO> para continuar... ' + Fore.WHITE)

# Acepta un número entero


def inputInt(descripcion, min=None, max=None):
    while (True):
        inputUsuario = input('· ' + descripcion + '? ')

        if (inputUsuario == 'F' or inputUsuario == 'f'):
            # Finalizar
            return None
        elif (testInputInt(inputUsuario, min, max)):
            return int(inputUsuario)
        else:
            # Opción incorrecta
            print('  ATENCION: Valor NO válido ...')


# Valida si una cadena es número ENTERO ente min y max
# DEVUELVE:
#   True -> Numero entre min y max
#   Falso -> No cumple los requisitos
def testInputInt(numero, min=None, max=None):
    try:
        # Convertir le input del usuario a tipo entero
        numero = int(numero)

    except:
        # El input no se ha podido convertir a tipo entero
        return False

    else:
        # Comprobar si esta en el rango especificado
        ok = True
        if (min != None):
            ok = (min <= numero)

        if (max != None):
            ok = ok and (numero <= max)

        return ok


# Acepta un número REAL
def inputReal(descripcion, min=None, max=None):
    while (True):
        inputUsuario = input('· ' + descripcion + '? ')

        if (inputUsuario == 'F' or inputUsuario == 'f'):
            # Finalizar
            return None
        elif (testInputReal(inputUsuario, min, max)):
            return float(inputUsuario)
        else:
            # Opción incorrecta
            print('  ATENCION: Valor NO válido ...')

# Valida si una cadena es número REAL ente min y max
# DEVUELVE:
#   True -> Numero entre min y max
#   Falso -> No cumple los requisitos


def testInputReal(numero, min=None, max=None):
    try:
        # Convertir le input del usuario a tipo entero
        numero = float(numero)

    except:
        # El input no se ha podido convertir a tipo entero
        return False

    else:
        # Comprobar si esta en el rango especificado
        ok = True
        if (min != None):
            ok = (min <= numero)

        if (max != None):
            ok = ok and (numero <= max)

        return ok


def mostrar_resultado(descripcion, resultado):
    print(Fore.WHITE + "> " + descripcion + Fore.YELLOW, resultado, Fore.WHITE)


def mostrar_resultado_lista(descripcion, lista_resultado):
    print(Fore.WHITE + "> " + descripcion + Fore.YELLOW)

    for r in lista_resultado:
        print("  " + r.__str__())

    print(Fore.WHITE)
