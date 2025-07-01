import numpy as np
from collections import Counter

class HammingLanguageClassifierWithMaxNet:
    def __init__(self, inhibition_factor=0.05, max_iterations=50):
        self.patterns = []
        self.labels = []
        self.inhibition_factor = inhibition_factor
        self.max_iterations = max_iterations
        self.alphabet = 'abcdefghijklmnopqrstuvwxyzàâáãçéêèíîìóôòõúûùñ'
        self.common_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'je', 'tu', 
                           'o', 'e', 'no', 'si', 'com', 'ma', 'di', 'che', 'uma', 'dos']
        self.endings = ['o', 'a', 'e', 's', 'ão', 'ción', 'ment', 're', 'nte', 'que',
                       'ent', 'are', 'ere', 'ire', 'ette', 'eau', 'oi', 'ais', 'ção', 'inho']

    def _preprocess_text(self, text):
        text = text.lower()
        return ''.join(c for c in text if c.isalpha() or c in 'àâáãçéêèíîìóôòõúûùñ')

    def _text_to_vector(self, text, vector_size=100):
        text = self._preprocess_text(text)
        
        # 1. Frecuencia de letras
        letters = [c for c in text if c in self.alphabet]
        letter_counts = Counter(letters)
        total_letters = max(1, len(letters))
        letter_vec = [letter_counts.get(c, 0)/total_letters for c in self.alphabet[:30]]
        
        # 2. Caracteres especiales
        special_chars = ['à','â','á','ã','ç','é','ê','è','í','î']
        special_vec = [1 if any(c in text for c in special_chars) else 0]
        
        # 3. Terminaciones
        ending_vec = [1 if text.endswith(e) else 0 for e in self.endings[:10]]
        
        # 4. Palabras comunes
        words = text.split()
        common_vec = [1 if w in words else 0 for w in self.common_words[:20]]
        
        # 5. Estadísticas del texto
        stats_vec = [
            sum(len(w) for w in words)/max(1, len(words)),  # avg word length
            len([w for w in words if w.endswith('s')])/max(1, len(words)),  # plural ratio
            len([c for c in text if c in 'aeiou'])/max(1, len(text))  # vowel ratio
        ]
        
        # Combinar todas las características
        full_vec = letter_vec + special_vec + ending_vec + common_vec + stats_vec
        
        # Asegurar tamaño consistente
        return np.array(full_vec[:vector_size])

    def fit(self, texts, labels):
        self.patterns = []
        self.labels = []
        for text, label in zip(texts, labels):
            vec = self._text_to_vector(text)
            self.patterns.append(vec)
            self.labels.append(label)

    def _hamming_distance(self, a, b):
        return np.sum(np.abs(a - b))

    def _apply_maxnet(self, activations):
        A = np.array(activations, dtype=float)
        for _ in range(self.max_iterations):
            A = A - self.inhibition_factor * (np.sum(A) - A)
            A[A < 0] = 0
            if np.count_nonzero(A) <= 1:
                break
        return np.argmax(A)

    def predict(self, texts, k=3):
        results = []
        for text in texts:
            vec = self._text_to_vector(text)
            distances = [(self._hamming_distance(vec, p), i) 
                        for i, p in enumerate(self.patterns)]
            distances.sort()
            top_k = [self.labels[i] for _, i in distances[:k]]
            pred = Counter(top_k).most_common(1)[0][0]
            results.append(pred)
        return results

    def evaluate(self, texts, true_labels):
        preds = self.predict(texts)
        correct = sum(1 for p, t in zip(preds, true_labels) if p == t)
        accuracy = correct / len(true_labels)
        return accuracy, list(zip(texts, preds, true_labels))