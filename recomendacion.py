import numpy as np

# Lista de lugares de interés (id, latitud, longitud)
places_of_interest = [(1, 40.7128, -74.0060),  # Nueva York
                      (2, 34.0522, -118.2437),  # Los Ángeles
                      (3, 51.5074, -0.1278),    # Londres
                      # Agrega más lugares de interés según sea necesario
                     ]

# Parámetros del algoritmo Q-learning
learning_rate = 0.1
discount_factor = 0.9
exploration_prob = 0.2
num_episodes = 1000

# Inicialización de la matriz Q con valores arbitrarios
num_places = len(places_of_interest)
Q = np.zeros((num_places, num_places))

# Función de selección de acción (exploración y explotación)
def select_action(state):
    if np.random.uniform(0, 1) < exploration_prob:
        return np.random.choice(num_places)  # Exploración aleatoria
    else:
        return np.argmax(Q[state, :])  # Explotación

# Simulación de episodios de aprendizaje
for episode in range(num_episodes):
    current_place = np.random.randint(num_places)  # Lugar actual aleatorio
    
    while True:
        next_place = select_action(current_place)  # Selección de lugar de interés
        
        # Simulación de recompensa (en un escenario real, esto sería interactuar con el usuario y calcular la recompensa)
        reward = -1  # Penalización por cambiar de lugar
        
        # Actualización de la matriz Q usando la fórmula de Q-learning
        Q[current_place, next_place] = (1 - learning_rate) * Q[current_place, next_place] + learning_rate * (reward + discount_factor * np.max(Q[next_place, :]))
        
        current_place = next_place  # Transición al siguiente lugar
        
        if episode == num_episodes - 1:
            print("Matriz Q después del entrenamiento:")
            print(Q)
            break

# Recomendación para un usuario en un lugar específico (ubicación actual)
user_location = 0  # Por ejemplo, la ubicación del usuario es el primer lugar de interés en la lista
recommended_place = np.argmax(Q[user_location, :])
print("Lugar de interés recomendado para el usuario:", recommended_place + 1)  # Sumamos 1 para ajustar el índice a partir de 1
