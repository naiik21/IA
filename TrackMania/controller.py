import pydirectinput
import time

# Ejemplo: Simular teclas para mover el coche
def accelerate():
    pydirectinput.keyDown('up')  # Mantener acelerador presionado

def brake():
    pydirectinput.keyDown('down')  # Frenar

def turn_left():
    pydirectinput.keyDown('left')  # Girar izquierda
    time.sleep(0.1)  # Pequeña pausa para el giro
    pydirectinput.keyUp('left')   # Soltar tecla

def turn_right():
    pydirectinput.keyDown('right')
    time.sleep(0.1)
    pydirectinput.keyUp('right')

# Detener todas las teclas
def release_keys():
    pydirectinput.keyUp('up')
    pydirectinput.keyUp('down')
    pydirectinput.keyUp('left')
    pydirectinput.keyUp('right')
    
    
    
def take_action(action):
    release_keys()  # Limpiar acciones anteriores
    if action == 0:
        accelerate()
    elif action == 1:
        brake()
    elif action == 2:
        turn_left()
    elif action == 3:
        turn_right()