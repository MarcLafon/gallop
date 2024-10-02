from typing import Union


def boolean_flags(flag: Union[bool, str]) -> bool:
    if isinstance(flag, bool):
        return flag

    flag = flag.lower()
    if flag in ['true', 't', '1', 'yes', 'y']:
        return True
    elif flag in ['false', 'f', '0', 'no', 'n']:
        return False
