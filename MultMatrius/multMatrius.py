import numpy as np

def matmol(A, B):
    if len(A[0]) != len(B):
        return('NO')
    

    result=[]
    for x in range(len(A)):
        matAux=[]
        for y in range(len(B[0])):
            aux=0
            for z in range (len(B)):
                #a=A[x][z]
                #b=B[z][y]
                a=A[x][z]
                b=B[z][y]
                #print(x, y, z)
                #print(str(aux) + '+ (' + str(a) + '*' + str(b) +')')
                aux= aux + (a*b)
                #print(aux)
            matAux.append(aux)
        result.append(matAux)    
    print(result)
            
#matmol([[1,2],[3,4]],[[1,2],[3,4]])
#matmol([[2,-3,-5],[-1,4,5],[1,-3,-4]],[[2,2,0],[-1,-1,0],[1,2,1]])
#matmol([[1,2,3],[4,5,6],[7,8,9]],[[1,2],[3,4],[5,6]])