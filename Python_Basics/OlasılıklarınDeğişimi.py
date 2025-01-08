# Zarı Her 100 Atışta Olasılıkların Değişimi

import random
from collections import Counter
import matplotlib.pyplot as plt

def roll_die():
    return random.randint(1, 6)

def simulate_rolls(num_rolls):
    return [roll_die() for _ in range(num_rolls)]

def calculate_probabilities(results):
    total_rolls = len(results)
    counts = Counter(results)
    probabilities = {outcome: count / total_rolls for outcome, count in counts.items()}
    return probabilities

# Her 100 atış için olasılıkları hesaplama fonksiyonu
def track_probabilities_over_time(total_rolls, step):
    results = []
    probabilities_over_time = {}
    
    for current_roll in range(step, total_rolls + 1, step):
        results.extend(simulate_rolls(step))  # Yeni atışları ekle
        probabilities = calculate_probabilities(results)
        probabilities_over_time[current_roll] = probabilities
    
    return probabilities_over_time

def plot_probabilities_over_time(probabilities_over_time):
    steps = list(probabilities_over_time.keys())
    outcomes = sorted(next(iter(probabilities_over_time.values())).keys())  # Zar sonuçları (1-6)
    
    for outcome in outcomes:
        probs = [probabilities_over_time[step].get(outcome, 0) for step in steps]
        plt.plot(steps, probs, label=f'Sonuç {outcome}')
    
    plt.xlabel('Zar Atış Sayısı')
    plt.ylabel('Olasılık')
    plt.title('Her 100 Atışta Olasılıkların Değişimi')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    total_rolls = 1000  # Toplam zar atış sayısı
    step = 100         # Her kaç atışta bir olasılık hesaplanacak
    
    probabilities_over_time = track_probabilities_over_time(total_rolls, step)
    plot_probabilities_over_time(probabilities_over_time)