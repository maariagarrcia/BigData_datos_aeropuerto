### I M P O R T S
import helpers
from colorama import Fore

### F U N C I O N E S

class Menu:
    option_list = []
    function_list = []
    title = ""

    def __init__(self, menu_title):
        self.title = menu_title

    def addOption(self, newOption, newFunction):
        self.option_list.append(newOption)
        self.function_list.append(newFunction)

    def inputOption(self):
        option = -1
        while (True):
            user_input = input(Fore.GREEN + '· Dime, ¿que opción deseas? ' + Fore.WHITE)

            if (user_input == 'F' or user_input == 'f'): 
                # Finalizar
                return -1 # =========================================>

            elif (helpers.testInputInt(user_input, 1, len(self.option_list))): 
                # Seleccionada una opción correcta
                return int(user_input) # ==========================>
            else:
                # Opción incorrecta
                print(Fore.RED + '* ATENCION:  Selecciona una opción valida ...\n' + Fore.WHITE)

    def show(self):
        helpers.clear()
 
        print(Fore.GREEN + self.title)
        print()
        helpers.underline_it(Fore.GREEN, "=", 'MENU')

        for idx, val in enumerate(self.option_list):
           print(Fore.GREEN + str(idx+1) + Fore.WHITE, '', val)

        print()
        print(Fore.GREEN +"F" + Fore.WHITE, '', "Finalizar")
        print(Fore.WHITE)

    def start(self):
        helpers.clear()  # Limpia la terminal

        while (True):
            self.show()
            option = self.inputOption()

            if (option == -1):  # Salir
                print(Fore.GREEN + '> Nos vemos otro dia :-)')
                print(Fore.WHITE)
                break  # =============================>
                pass
            else:
                helpers.clear()
                helpers.underline_it(Fore.GREEN, "=", self.option_list[option-1].upper())
                self.function_list[option-1]()
                helpers.esperarIntro()
