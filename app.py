import random
import mido
from mido import Message, MidiFile, MidiTrack
from sklearn.cluster import KMeans
import numpy as np

def calculate_euclidean_distance(melody1, melody2):
    # Supondo que as melodias são representadas como sequências de notas e durações
    distance = sum((note1 - note2)**2 for note1, note2 in zip(melody1, melody2))**0.5
    return distance

def calculate_correlation(melody1, melody2):
    # Certifique-se de que ambas as melodias têm o mesmo comprimento
    assert len(melody1) == len(melody2), "As melodias devem ter o mesmo comprimento para calcular a correlação."

    # Converta as melodias para arrays do NumPy para facilitar os cálculos
    melody1_array = np.array(melody1)
    melody2_array = np.array(melody2)

    # Calcule a correlação de Pearson entre as duas melodias
    correlation = np.corrcoef(melody1_array, melody2_array)[0, 1]

    return correlation

def diversity_score(melodies):
    num_melodies = len(melodies)
    total_distance = 0

    for i in range(num_melodies):
        for j in range(i + 1, num_melodies):
            distance = calculate_euclidean_distance(melodies[i], melodies[j])
            total_distance += distance

    average_distance = total_distance / (num_melodies * (num_melodies - 1) / 2)
    diversity = 1 / (1 + average_distance)  # Quanto menor a distância média, maior a pontuação de diversidade

    return diversity

def fitness_function_by_diversity(melody, target_melody):
    correlation_score = calculate_correlation(melody, target_melody)
    diversity = diversity_score(population)

    # Combine as métricas em uma pontuação geral de fitness
    # Aqui, você pode ajustar os pesos relativos para a diversidade e outras métricas
    alpha = 0.5
    beta = 0.5
    fitness = alpha * correlation_score + beta * diversity

    return fitness

def fitness_function(melody, target_sequence):
    score = 0
    for i in range(len(melody) - len(target_sequence) + 1):
        if melody[i:i+len(target_sequence)] == target_sequence:
            score += 1
    return score

def generate_random_melody(melody_length):
    return [random.randint(50, 80) for _ in range(melody_length)]

def roulette_selection(population, fitness_fn):
    total_fitness = sum(fitness_fn(melody) for melody in population)
    probabilities = [fitness_fn(melody) / total_fitness for melody in population]
    return random.choices(population, probabilities, k=len(population))

def crossover(parent1, parent2):
    # Implemente um operador de crossover adequado para a representação de música
    # Por exemplo, pode ser um ponto de corte aleatório para trocar partes das melodias
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutation_note(melody, scale):
    # Selecionar uma nota aleatória na melodia
    index = random.randint(0, len(melody) - 1)
    note, duration = melody[index]

    # Escolher uma nova nota aleatória dentro da escala musical
    new_note = random.choice(scale)

    # Criar uma nova melodia com a nota mutada
    new_melody = melody.copy()
    new_melody[index] = (new_note, duration)

    return new_melody

def mutation_duration(melody, max_duration):
    # Selecionar uma nota aleatória na melodia
    index = random.randint(0, len(melody) - 1)
    note, duration = melody[index]

    # Escolher uma nova duração aleatória entre 1 e a duração máxima permitida
    new_duration = random.randint(1, max_duration)

    # Criar uma nova melodia com a duração mutada
    new_melody = melody.copy()
    new_melody[index] = (note, new_duration)

    return new_melody

def mutation_insert_remove(melody, max_duration, scale):
    # Escolher aleatoriamente entre inserir ou remover uma nota
    if random.random() < 0.5:  # Inserir uma nota
        # Escolher uma nota e duração aleatórias dentro da escala e duração máxima
        new_note = random.choice(scale)
        new_duration = random.randint(1, max_duration)

        # Inserir a nova nota na melodia em uma posição aleatória
        index = random.randint(0, len(melody))
        new_melody = melody.copy()
        new_melody.insert(index, (new_note, new_duration))
    else:  # Remover uma nota
        if len(melody) > 1:
            # Selecionar uma nota aleatória na melodia e removê-la
            index = random.randint(0, len(melody) - 1)
            new_melody = melody.copy()
            del new_melody[index]
        else:
            # Se a melodia tem apenas uma nota, não é possível remover, retornar a mesma melodia
            new_melody = melody

    return new_melody

def mutation_transpose(melody, transpose_range):
    # Escolher um valor aleatório para transposição dentro do intervalo especificado
    transpose_value = random.randint(-transpose_range, transpose_range)

    # Transpor todas as notas da melodia pelo valor escolhido
    new_melody = [(note + transpose_value, duration) for note, duration in melody]

    return new_melody

def mutation_swap(melody):
    # Selecionar duas posições aleatórias diferentes na melodia
    index1, index2 = random.sample(range(len(melody)), 2)

    # Criar uma nova melodia com as duas posições trocadas
    new_melody = melody.copy()
    new_melody[index1], new_melody[index2] = new_melody[index2], new_melody[index1]

    return new_melody

def mutate(melody, mutation_rate):
    # Implemente um operador de mutação adequado para a representação de música
    # Por exemplo, pode ser uma pequena alteração aleatória em algumas notas
    mutated_melody = [note + random.randint(-1, 1) for note in melody]
    return mutated_melody

def next_generation(population, fitness_fn, mutation_rate):
    new_population = []
    for _ in range(len(population) // 2):
        parent1, parent2 = roulette_selection(population, fitness_fn)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        new_population.extend([child1, child2])
    return new_population

def extract_melody(midi_file_path):
    melody = []
    midi_file = MidiFile(midi_file_path)
    for msg in midi_file:
        if msg.type == 'note_on':
            melody.append(msg.note)
    return melody

def cluster_melodies(melodies, num_clusters):
    X = np.array(melodies)
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(X)
    return cluster_labels

def create_midi_file(melody, filename='output.mid'):
    midi_file = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    midi_file.tracks.append(track)

    # Defina alguns parâmetros MIDI
    tempo = 500000  # Valor em microsegundos (120 BPM)
    track.append(Message('set_tempo', tempo=tempo))

    # Percorra a melodia otimizada e adicione as mensagens MIDI ao arquivo
    for note in melody:
        note_on_msg = Message('note_on', note=note, velocity=64, time=0)
        note_off_msg = Message('note_off', note=note, velocity=64, time=480)  # Duração de 480 ticks (1 beat)

        track.append(note_on_msg)
        track.append(note_off_msg)

    # Salve o arquivo MIDI
    midi_file.save(filename)

# Função de perturbação - Exemplo: Mutação de Nota
def perturbation(melody, scale):
    # Implemente a perturbação da melhor melodia aqui
    # Exemplo de perturbação - mutação de nota
    index = random.randint(0, len(melody) - 1)
    note, duration = melody[index]
    new_note = random.choice(scale)
    melody[index] = (new_note, duration)

    return melody

# Função de busca local - Exemplo: Mutação de Nota
def local_search(best_melody, scale, num_mutations):
    # Faça uma cópia da melhor melodia para trabalhar nela
    current_melody = best_melody.copy()

    for _ in range(num_mutations):
        # Realize a perturbação (mutação de nota)
        current_melody = perturbation(current_melody, scale)

    return current_melody

# Implementação do VNS na melhor melodia de cada geração do AG
def apply_vns(best_melody, scale, max_iterations):
    # Faça uma cópia da melhor melodia para trabalhar nela
    current_melody = best_melody.copy()

    # Defina um conjunto de vizinhanças (exemplo: perturbação de nota, transposição, etc.)
    neighborhoods = [perturbation]

    iteration = 0
    while iteration < max_iterations:
        # Escolha uma vizinhança aleatoriamente
        neighborhood = random.choice(neighborhoods)

        # Aplique a busca local (perturbação) na melhor melodia
        perturbed_melody = neighborhood(current_melody, scale)

        # Avalie a qualidade da melodia perturbada usando a função fitness
        perturbed_fitness = fitness_function(perturbed_melody)

        # Se a melodia perturbada for melhor do que a melhor melodia atual, atualize-a
        if perturbed_fitness > fitness_function(current_melody):
            current_melody = perturbed_melody

        iteration += 1

    return current_melody

if __name__ == '__main__':    
    pop_size = 100 
    melody_length = 16
    population = [generate_random_melody(melody_length) for _ in range(pop_size)]

    num_clusters = 5
    cluster_labels = cluster_melodies(population, num_clusters)

    target_midi_file_path = 'Kirbys Return to Dream Land Title Theme 8 Bit.mid'
    target_sequence = extract_melody(target_midi_file_path)  
    mutation_rate = 0.1  

    num_generations = 100
    for generation in range(num_generations):
        # Avalie a população atual
        population_fitness = [(melody, fitness_function(melody, target_sequence)) for melody in population]

        # Encontre o melhor indivíduo nesta geração
        best_melody, best_fitness = max(population_fitness, key=lambda x: x[1])
        print(f"Generation {generation}, Best Fitness: {best_fitness}")

        if generation <= num_generations - 1:
            filename = f'output_generation_{generation}.mid'
            create_midi_file(best_melody, filename)

            best_melody = apply_vns(best_melody, 0.1, max_iterations=10)
            filename = f'output_generation_{generation}_vns.mid'
            create_midi_file(best_melody, filename)

        # Verifique o critério de parada
        if best_fitness >= len(target_sequence):
            break

        # Crie a próxima geração
        population = next_generation([melody for melody, _ in population_fitness], fitness_function, mutation_rate)