import json
import random
from hamming_language_classifier_maxnet import HammingLanguageClassifierWithMaxNet

def load_and_prepare_data(filename, test_size=0.2):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = []
    labels = []
    for label, samples in data.items():
        texts.extend(samples)
        labels.extend([label] * len(samples))
    
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts[:], labels[:] = zip(*combined)
    
    split = int((1 - test_size) * len(texts))
    return texts[:split], labels[:split], texts[split:], labels[split:]

def main():
    # Cargar datos
    train_texts, train_labels, test_texts, test_labels = load_and_prepare_data(
        "romance_text_dataset.json"
    )
    
    # Entrenar y evaluar
    clf = HammingLanguageClassifierWithMaxNet()
    clf.fit(train_texts, train_labels)
    accuracy, results = clf.evaluate(test_texts, test_labels)
    
    # Resultados
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("\nEjemplos de clasificación:")
    for text, pred, true in results[:20]:  # Mostrar solo los primeros 20
        print(f"Frase: '{text[:30]}...' | Predicho: {pred} | Real: {true} | {'✓' if pred == true else '✗'}")
    
    # Estadísticas por clase
    from collections import defaultdict
    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for _, pred, true in results:
        stats[true]['total'] += 1
        if pred == true:
            stats[true]['correct'] += 1
    
    print("\nPrecisión por idioma:")
    for lang, counts in stats.items():
        print(f"{lang}: {counts['correct']/counts['total']*100:.2f}% ({counts['correct']}/{counts['total']})")

if __name__ == "__main__":
    main()