# 09. Typoglycemia
import random

def shuffle_word(word):
    if len(word) <= 4:
        return word
    else:
        first_letter = word[0]
        last_letter = word[-1]
        middle_letters = list(word[1:-1])
        random.shuffle(middle_letters)
        shuffled_word = first_letter + ''.join(middle_letters) + last_letter
        return shuffled_word

def process_sentence(sentence):
    words = sentence.split()
    processed_words = [shuffle_word(word) for word in words]
    processed_sentence = ' '.join(processed_words)
    return processed_sentence

# Example usage
sentence = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind"
processed_sentence = process_sentence(sentence)
print("Original sentence:", sentence)
print("Processed sentence:", processed_sentence)