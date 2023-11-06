import numpy as np

# Lista de lugares de interés (id, latitud, longitud)
places_of_interest = [(1, 40.7128, -74.0060),  # Nueva York
                      (2, 34.0522, -118.2437),  # Los Ángeles
                      (3, 51.5074, -0.1278)]    # Londres

# Probabilidades de éxito inicial para cada lugar de interés (pueden ser ajustadas según el tipo de destino)
probabilities = np.array([0.7, 0.6, 0.8])

# Parámetros del algoritmo Q-learning
learning_rate = 0.1
discount_factor = 0.9
exploration_prob = 0.2
num_episodes = 1000
max_steps = 100

# Inicialización de la matriz Q con valores en cero
num_places = len(places_of_interest)
Q = np.zeros((num_places, num_places))

# Probabilidades de éxito ajustadas basándose en las preferencias del usuario
user_preferences = np.array([0.8, 0.3, 0.9])  # Ejemplo de preferencias del usuario
adjusted_probabilities = probabilities * user_preferences

# Función de selección de acción considerando las probabilidades de éxito ajustadas
def select_action(state):
    if np.random.uniform(0, 1) < exploration_prob:
        return np.random.choice(num_places)  # Exploración aleatoria
    else:
        # Selección de acción basada en las probabilidades de éxito ajustadas
        return np.random.choice(num_places, p=adjusted_probabilities / np.sum(adjusted_probabilities))

# Simulación de episodios de aprendizaje
for episode in range(num_episodes):
    current_place = np.random.randint(num_places)  # Lugar actual aleatorio
    steps = 0  # Contador de pasos
    
    while steps < max_steps:
        next_place = select_action(current_place)  # Selección de lugar de interés
        
        # Probabilidad de éxito para la acción
        success_probability = adjusted_probabilities[current_place]
        
        # Simulación de recompensa considerando la incertidumbre
        reward = -1 * (1 - success_probability)  # Penalización considerando la probabilidad de éxito
        
        # Actualización de la matriz Q usando la fórmula de Q-learning con la nueva recompensa
        Q[current_place, next_place] = (1 - learning_rate) * Q[current_place, next_place] + learning_rate * (reward + discount_factor * np.max(Q[next_place, :]))
        
        current_place = next_place  # Transición al siguiente lugar
        steps += 1  # Incrementar el contador de pasos
        
    if episode == num_episodes - 1:
        print("Matriz Q después del entrenamiento:")
        print(Q)
        break

# Recomendación para un usuario en un lugar específico (ubicación actual)
user_location = 0  # Por ejemplo, la ubicación del usuario es el primer lugar de interés en la lista
recommended_place = np.argmax(Q[user_location, :])
print("Lugar de interés recomendado para el usuario:", recommended_place + 1)  # Sumamos 1 para ajustar el índice a partir de 1
