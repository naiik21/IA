backpack_weight_limit = 15
objects = [{"weight": 1, "value": 1},
           {"weight": 6, "value": 4},
           {"weight": 4, "value": 7},
           {"weight": 5, "value": 6},
           {"weight": 1, "value": 3},
           {"weight": 6, "value": 8},
           {"weight": 3, "value": 6},
           {"weight": 10, "value": 11},
           {"weight": 4, "value": 4},
           {"weight": 7, "value": 3}]

# Función utilizada para buscar la mejor combinación comprobando todas las alternativas de manera recursiva
def calc_best_combination(weight_limit, objects):
  # Comprobamos si quedan objetos que pesen menos o igual que el limite de peso restante y que el límite de peso no sea 0
  if weight_limit == 0 or not check_weight_and_objects(weight_limit, objects):
    return [0, []]
  else:
    for element in objects:
      # Compruebo que el objeto quepa en la bolsa
      if element["weight"] <= weight_limit:
        # Hago una copia del listado de objetos y elimino este
        other_objects = objects.copy()
        other_objects.remove(element)
        
        # Busco de manera recursiva la mejor combinación partiendo de que he seleccionado temporalmente este elemento
        best_combination = calc_best_combination(weight_limit - element["weight"], other_objects)
        best_value = best_combination[0] + element["value"]
        best_objects = best_combination[1]
        best_objects.insert(0, element)

        if not 'greatest_value' in locals() or best_value > greatest_value:
          greatest_value = best_value
          greatest_objects = best_objects

    return [greatest_value, greatest_objects]

# Función utilizada para comprobar que dentro del límite de peso, existan objectos que quepan
def check_weight_and_objects(weight_limit, objects):
  space_left = False
  for element in objects:
    if element["weight"] <= weight_limit:
      space_left = True
      break
  return space_left



best_value, best_combination = calc_best_combination(backpack_weight_limit, objects)

print("El mayor valor obtenido es: ", best_value)
print("La selección de objectos sería la siguiente: ", best_combination)