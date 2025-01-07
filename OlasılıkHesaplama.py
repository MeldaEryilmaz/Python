# Zar atışında her sayının kaç kez geldiğini hesaplama kodu

import random
import matplotlib.pyplot as plt
from collections import Counter

def roll_die():
    return random.randint(1,6)

def simulate_rolls(num_rolls):
    results = [roll_die() for _ in range(num_rolls)]
    return results

def calculate_probabilities(results):
    total_rolls = len(results)
    counts = Counter(results) # her sayının kaç kez geldiğini hesaplar
    probabilities = {outcome: count / total_rolls for outcome, count in counts.items()}
    return probabilities

def plot_probabilities(probabilities):
    outcomes = list(probabilities.keys())
    probs = list(probabilities.values())

    plt.bar(outcomes, probs, color='skyblue')
    plt.xlabel('zar sonuçları')
    plt.ylabel('olasılık')
    plt.title('zar atma olasılıkları')
    plt.xticks(outcomes)
    plt.show()

if __name__=="__main__":
    num_rolls = 1000 # 1000 zar atışı dedik
    results = simulate_rolls(num_rolls)

    probabilities = calculate_probabilities(results)

    print("olasılıklar: ")
    for outcome, prob in probabilities.items():
        print(f"{outcome}: {prob:.2%}")
    
    plot_probabilities(probabilities)