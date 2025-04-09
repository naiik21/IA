import threading
import time
from model import Network, QTrainer
from snakeGame import Point, Direction, SnakeGame
import numpy as np
import torch
import random
from collections import deque



INPUT=15
HIDDEN=1024
OUTPUT=3
LR= 0.001
NUM_AGENTS = 4  # Número de agentes corriendo en la misma ventana

class Agent:
    def __init__(self):
        self.game_counter = 0
        self.gamma = 1 - LR
        self.model = Network(INPUT, HIDDEN, OUTPUT)
        self.trainer = QTrainer(self.model, LR, self.gamma)
        self.memory = deque(maxlen=100000)
    
    def state(self, game):
        head = game.snake[0]
        
        roll=game.ROLL
        
        point_l = Point(head.x - game.BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - game.BLOCK_SIZE)
        point_r = Point(head.x + game.BLOCK_SIZE, head.y)
        point_d = Point(head.x, head.y + game.BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_u = game.direction == Direction.UP
        dir_r = game.direction == Direction.RIGHT
        dir_d = game.direction == Direction.DOWN

        die_l = (dir_u and game.is_collision(point_l)) or \
                (dir_l and game.is_collision(point_d)) or \
                (dir_d and game.is_collision(point_r)) or \
                (dir_r and game.is_collision(point_u))
        die_s = (dir_u and game.is_collision(point_u)) or \
                (dir_l and game.is_collision(point_l)) or \
                (dir_d and game.is_collision(point_d)) or \
                (dir_r and game.is_collision(point_r)) 
        die_r = (dir_u and game.is_collision(point_r)) or \
                (dir_l and game.is_collision(point_u)) or \
                (dir_d and game.is_collision(point_l)) or \
                (dir_r and game.is_collision(point_d))
        
        
        pos_s = False
        pos_l = False
        pos_r = False
        
        if game.direction == Direction.RIGHT:
            for i in range(int(head.x), game.BLOCK_SIZE):
                point=Point(i, head.y)
                pos_s= game.is_collision(point)
        
            for i in range(0, int(head.y)):
                point=Point(head.x, i)
                pos_l= game.is_collision(point)

            for i in range(int(head.y), game.BLOCK_SIZE):
                point=Point(head.x, i)
                pos_r= game.is_collision(point)
                
        if game.direction == Direction.LEFT:
            for i in range(0, int(head.x)):
                point=Point(i, head.y)
                pos_s= game.is_collision(point)
              
            for i in range(int(head.y), game.BLOCK_SIZE):
                point=Point(head.x, i)
                pos_l= game.is_collision(point)
  
            for i in range(0, int(head.y)):
                point=Point(head.x, i)
                pos_r= game.is_collision(point)
                
        if game.direction == Direction.UP:
            for i in range(0, int(head.y)):
                point=Point(head.x, i)
                pos_s= game.is_collision(point)

            for i in range(int(head.x), game.BLOCK_SIZE):
                point=Point(i, head.y)
                pos_r= game.is_collision(point)
        
            for i in range(0, int(head.x)):
                point=Point(i, head.y)
                pos_l= game.is_collision(point) 
            
        if game.direction == Direction.DOWN:
            for i in range(int(head.y), game.BLOCK_SIZE):
                point=Point(head.x, i)
                pos_s= game.is_collision(point)
             
            for i in range(int(head.x), game.BLOCK_SIZE):
                point=Point(i, head.y)
                pos_l= game.is_collision(point)
        
            for i in range(0, int(head.x)):
                point=Point(i, head.y)
                pos_r= game.is_collision(point)  
 
     
        
        apple_l = game.food.x < head.x
        apple_u = game.food.y < head.y
        apple_r = game.food.x > head.x
        apple_d = game.food.y > head.y

        state = [
            dir_l,
            dir_u,
            dir_r,
            dir_d,
            die_l,
            die_s,
            die_r,
            apple_l,
            apple_u,
            apple_r,
            apple_d,
            pos_s,
            pos_l,
            pos_r, 
            roll
        ]

        return np.array(state, dtype=int)
    
    
    def action(self, state):
        move=[0,0,0]
        epsilon= 100-self.game_counter
        if random.randint(0, 200) < epsilon:
            index= random.randint(0,2)
        else:
            state_tensor=torch.tensor(state, dtype=torch.float)
            pred= self.model(state_tensor)
            index= torch.argmax(pred).item()
        move[index]=1
        return move


    def train(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def save_step(self, state, action,reward, next_state, done):
        self.memory.append((state, action,reward, next_state, done))
        
    def train_long(self):
        if len(self.memory)> 1000:
            sample=random.sample(self.memory, 1000)
        else:
            sample= self.memory
        states, actions, rewrds, new_states, dones=zip(*sample)
        self.trainer.train_step(states, actions, rewrds, new_states, dones)
        

def play():
    max_score=0
    agent=Agent()
    game= SnakeGame()
    while True:
        state= agent.state(game)
        action= agent.action(state)
        reward, done, score= game.play_step(action)
        new_state= agent.state(game)
        
        agent.train(state, action, reward, new_state, done)
        agent.save_step(state, action, reward,  new_state, done) 
        
        if done:
            game.reset()
            agent.game_counter+= 1
            agent.train_long()
            
            if score> max_score:
                max_score= score
                agent.model.save()
                
            print('Game: ', agent.game_counter, 'Score: ', score, 'Max Score: ', max_score)  

            

if __name__ == '__main__':
    play()        