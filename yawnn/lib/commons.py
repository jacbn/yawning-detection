from typing import TypeVar, Callable
from os import listdir
from os.path import isfile, join

T = TypeVar('T')

def mapToDirectory(f : Callable[[str], T], path : str) -> list[T]:
    return [f(join(path,file)) for file in listdir(path) if isfile(join(path, file))]