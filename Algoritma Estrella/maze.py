import pyamaze as maze
from queue import PriorityQueue

ROWS = 20
COLS = 20

def distance(cell1, cell2):
    return abs(cell1[0]-cell2[0])+abs(cell1[1]-cell2[1])

#print(distance((1,1),(1,2)))
    

def aStar(m:maze):
    #Variables
    start=(1,1)
    end=(ROWS, COLS)
    forwardPath={}
    distancia_recorrida={cell:float("inf") for cell in m.grid}
    distancia_recorrida[start]=0
    
    #Crear e inicializar PriorityQueue
    open=PriorityQueue()
    open.put((0+distance(start,end),start))
    
    
    while not open.empty():
        current=open.get()[1]
        if current == end:
            break
        
        for direccion in "SNEW":
            #Busca el camino 
            if m.maze_map[current][direccion] == 1:
                match direccion:
                    case 'E':
                        nextPos=(current[0],current[1]+1)
                    case 'W':
                        nextPos=(current[0],current[1]-1)
                    case 'N':
                        nextPos=(current[0]-1,current[1])
                    case 'S':
                        nextPos=(current[0]+1,current[1])
                recorrido_auxiliar = distancia_recorrida[current] + 1 
                #Mejor recorrido
                if distancia_recorrida[nextPos] > recorrido_auxiliar:
                    distancia_recorrida[nextPos] = recorrido_auxiliar
                    forwardPath[nextPos] = current
                    open.put((distancia_recorrida[nextPos]+distance(nextPos, end) , nextPos))
                    
            #en caso de poder atravesar una pared a cambio de 5     
            elif m.maze_map[current][direccion] == 0: 
                match direccion:
                    case 'E':
                        nextPos=(current[0],current[1]+1)
                        if nextPos[1]>COLS:
                            nextPos=(current[0],COLS)
                    case 'W':
                        nextPos=(current[0],current[1]-1)
                        if nextPos[1]<=0:
                            nextPos=(current[0],1)
                    case 'N':
                        nextPos=(current[0]-1,current[1])
                        if nextPos[0]<=0:
                            nextPos=(1,current[1])
                    case 'S':
                        nextPos=(current[0]+1,current[1])
                        if nextPos[0]>ROWS:
                            nextPos=(ROWS, current[1])
                recorrido_auxiliar = distancia_recorrida[current] +5
                #Mejor recorrido
                if distancia_recorrida[nextPos] > recorrido_auxiliar:
                    distancia_recorrida[nextPos] = recorrido_auxiliar
                    forwardPath[nextPos] = current
                    open.put((distancia_recorrida[nextPos]+distance(nextPos, end) , nextPos))
                
    return forwardPath
            
                       
    #print(m.grid)
    #print(labertinto[a.position]) {N / S / E / W}
    #print(labertinto[a.position].get(x)) 0 / 1
    

m=maze.maze(ROWS,COLS)
m.CreateMaze()
#pre_Astar = time.time()
path = aStar(m)
#post_Astar = time.time()
#print(post_Astar - pre_Astar)
a=maze.agent(m,footprints=True)
m.tracePath({a:path},delay=300)  
m.run()
