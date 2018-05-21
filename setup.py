import sys 
from cx_Freeze import setup, Executable

setup(
    name ="aitakatta",
    version ="1.0",
    description="yolo wololo",
    executables=[Executable("Integration_2.py")]
)