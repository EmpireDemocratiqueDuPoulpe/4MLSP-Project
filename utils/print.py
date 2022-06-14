# #################################################################################################################### #
#       print.py                                                                                                       #
#           Makes nice output from values                                                                              #
# #################################################################################################################### #

import shutil
from colorama import Fore


def title(text, char="#"):
    terminal_width = shutil.get_terminal_size().columns
    txt = f"{Fore.GREEN}{char * 4} {text}"

    print(f"\n{txt} {char * (terminal_width - len(txt))}")
