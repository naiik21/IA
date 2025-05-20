from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import NatureCNN
from gym import spaces, Env
import numpy as np
from stable_baselines3 import PPO
import numpy as np
import cv2
import numpy as np
from mss import mss  # Librería para captura de pantalla eficiente
from screeninfo import get_monitors

class TrackManiaEnv(Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4)  # 4 acciones
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        
         # Configuración de captura de pantalla
        default_monitor = get_monitors()[0]
        self.monitor = {"top": 0, "left": 0, "width": default_monitor.width, "height": default_monitor.height}
        self.sct = mss()
        
        # Variables para el estado del juego
        self.current_frame = np.zeros((84, 84, 1), dtype=np.uint8)
        
    def detect_track(self, edges):
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(edges, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Dibujar líneas blancas
        return edges
    
    def preprocess(self, frame):
        frame = cv2.resize(frame, (84, 84))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges = self.detect_track(edges)
        return np.expand_dims(edges, axis=-1)  # Añadir dimensión del canal

    
    def get_game_screen(self):
        screen = np.array(self.sct.grab(self.monitor))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        processed_frame = self.preprocess(frame)
        self.current_frame = processed_frame
        return processed_frame
    
    
    def step(self, action):
        self.take_action(action)  # Ejecutar acción
        observation = self.get_game_screen()  # Capturar pantalla
        reward = self.calculate_reward()      # Calcular recompensa
        done = self.check_if_crashed()       # Verificar si chocó
        info = {}                            # Info adicional (opcional)
        
        return observation, reward, done, info
    
    def take_action(self, action):
        # Implementa las acciones del juego aquí
        # Ejemplo: 0=adelante, 1=izquierda, 2=derecha, 3=frenar
        pass
    
    def calculate_reward(self):
        # Implementa tu lógica de recompensa aquí
        return 0
    
    def check_if_crashed(self):
        # Implementa la detección de colisión aquí
        return False
    
    def reset(self):
        self.restart_game()  # Reiniciar nivel
        return self.get_game_screen()
    
    def restart_game(self):
        # Implementa el reinicio del juego aquí
        pass
    
    def render(self, mode='human'):
        if mode == 'human':
            cv2.imshow("TrackMania RL Agent", self.current_frame)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return self.current_frame


# Ejemplo de uso
if __name__ == "__main__":
    env = TrackManiaEnv()
    obs = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()  # Acción aleatoria
        obs, reward, done, info = env.step(action)
        env.render()
        
        if done:
            obs = env.reset()
    
    env.close()