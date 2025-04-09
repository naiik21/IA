# with open("Kibalion.txt", "r", encoding="utf-8") as file:
#     txt=file.read()

txt="El esoterismo es rico en palabras claves, símbolos y «esencias» conceptuales. Su transmisión, a través de las edades, implicó un esforzado aprendizaje, una memorización de significados, «acentos» y una persistente custodia de sus valores originales para que nada de lo preservado perdiera su color, su sabor, su propósito y su intensidad. Al amparo de tales premisas fue creciendo paulatinamente el árbol de la ciencia hermética que reconoce como sus raíces a El Kybalion. Y este último resumen de un conocimiento intemporal, encontró en Hermes Trismegisto a su más con sumado mentor y mensajero. En estas páginas redactadas con hondura y ex actitud por tres iniciados, es posible pasar revista a tópicos realmente sapi enciales sobre la filosofía oculta. Sus principios rectores (en los que el men talismo, la correspondencia, la vibración, la polaridad, causa y efecto, y la generación juegan papeles preponderantes); la transmutación mental, la to talidad, el universo mental, la paradoja divina y los axiomas herméticos son tan sólo algunos de los temas tan bien expuestos aquí. El Kybalion es, pues, una exposición sincera y rotunda de los esquemas básicos del esoterismo, y como muy bien lo señalan los tres iniciados, no se proponen erigir un nuevo templo de la sabiduría, sino poner manos del investigador la llave que abrirá las numerosas puertas internas que conducen hacia el Templo del Misterio. Y, en rigor de la verdad, las muchas reediciones de esta obra, su constante renovación, a través de los distintos círculos herméticos del mundo en sus reflexiones, pláticas, conferencias y clases, son ratificación elocuentísima de las bondades de una doctrina que ilumina a la humanidad desde hace siglos."
tokens={}
# print(txt)

#Quitamos espacios
txt=txt.replace(" ", "")

#Diccionario ASCII
diccionario = {i:chr(i) for i in range(256)}

#Transformamos el texto en ASCII
txt_ascii = [ord(i) for i in txt]

n_tokens=100

for n in range(n_tokens):
    #Recorremos en busca de la pareja mas utilizada
    for x in range(len(txt_ascii) -1):
        chars=(txt_ascii[x], txt_ascii[x+1])
        # print(chars)
        if chars in tokens:
            tokens[chars]+=1
        else:
            tokens[chars]= 1
            
    #Ordenamos y cogemos el token mas utilizado
    token_list=list(sorted(tokens.items(), key=lambda item: item[1], reverse=True))
    # print("////")
    # print(token_list)
    token=token_list[0][0]
    #print(token)
    #Añadimos un nuevo valor ASCII con el token mas utilizado 
    new_token=len(diccionario)+1
    
    diccionario[new_token]= token
    

    # Subsituimos al nuevo ASCII
    txt_new = []
    i = 0
    while i < len(txt_ascii)-1:
        # Verificar si la subsecuencia coincide
        if (txt_ascii[i],txt_ascii[i+1]) == token:
            txt_new.append(new_token)  # Reemplazar con el nuevo valor
            i += len(token)  # Saltar los elementos de la secuencia
        else:
            txt_new.append(txt_ascii[i])  # Conservar el elemento original
            i += 1

    txt_ascii=txt_new
    
#print(txt_ascii)
# print(diccionario)

# print("/////")

def encodign(diccionario, txt_ascii):
    # Convertir el array a texto usando el 
    text = []
    for token in txt_ascii:
        text.append(diccionario[token])
    #    print(texto)


    
    
encodign(diccionario, txt_ascii)
