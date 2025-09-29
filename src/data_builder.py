import random
from constants import SOS_TOKEN, EOS_TOKEN, UNK_TOKEN

class DataBuilder:
    def __init__(self):
        self.people = ['boy', 'girl', 'man', 'woman', 'child', 'adult', 'teacher', 'student', 'painter', 'professor', 'artist', 'engineer', 'writer', 'actor', 'actress']
        self.animals = ['dog', 'cat', 'cow', 'horse', 'rabbit', 'chicken', 'sheep', 'duck', 'turkey']
        self.nouns = self.people + self.animals

        self.people_adjectives = ['beautiful', 'happy', 'smart', 'young', 'polite', 'kind', 'tall', 'agreable', 'ambitious', 'calm']
        self.animals_adjectives = ['playful', 'beautiful', 'agile', 'sleepy', 'calm', 'friendly', 'fast', 'shy', 'timid', 'elegant']
        self.adjectives = self.people_adjectives + self.animals_adjectives

        self.people_verbs = ['is reading', 'is eating', 'is studying', 'is singing', 'is laughing', 'is speaking', 'is thinking', 'is working', 'is writing', 'is drawing']
        self.animals_verbs = ['is eating', 'is sleeping', 'is drinking', 'is exploring', 'is playing', 'is running', 'is resting']
        self.verbs = self.people_verbs + self.animals_verbs

        self.people_location = ['in the park', 'at home', 'at the restaurant', 'at the beach', 'in the classroom', 'in the garden']
        self.animals_location = ['on the farm', 'in the barn', 'in the field']
        self.locations = self.people_location + self.animals_location

        unique_words = self.nouns + self.adjectives

        for verb in self.verbs:
            parts = verb.split()
            unique_words.extend(parts)

        for location in self.locations:
            parts = location.split()
            unique_words.extend(parts)

        unique_words += ["the"]
        unique_words = sorted(set(unique_words))
        print('number of unique words: ', len(unique_words))
        self.all_words = [SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + unique_words
        self.word2idx = {w: i for i, w in enumerate(self.all_words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def choose_adj(self, noun):
        if noun in self.people:
            return random.choice(self.people_adjectives)

        if noun in self.animals:
            return random.choice(self.animals_adjectives)

    def choose_verb(self, noun):
        if noun in self.people:
            return random.choice(self.people_verbs)

        if noun in self.animals:
            return random.choice(self.animals_verbs)
            
    def choose_loc(self, noun):
        if noun in self.people:
            return random.choice(self.people_location)

        if noun in self.animals:
            return random.choice(self.animals_location)

    def generate_sentence(self):
        noun = random.choice(self.nouns)
        adj = self.choose_adj(noun)
        verb = self.choose_verb(noun)
        loc = self.choose_loc(noun)

        src = f"the {adj} {noun} {verb} {loc}"
        target = f"the {noun} {verb}"
        return (src.split(), target.split())

    def sentence_to_indices(self, tokens):
        indices = [self.word2idx.get(SOS_TOKEN)] + [self.word2idx.get(w) for w in tokens] + [self.word2idx.get(EOS_TOKEN)]
        return indices
    
    def generate_large_dataset(self, num_samples=5000, filename="data/dataset.txt"):
        X, Y = [], []

        with open(filename, "w", encoding="utf-8") as f:  # overwrite if file exists
            for _ in range(num_samples):
                src_tokens, tgt_tokens = self.generate_sentence()

                X.append(self.sentence_to_indices(src_tokens))
                Y.append(self.sentence_to_indices(tgt_tokens))

                f.write(" ".join(src_tokens) + "\t" + " ".join(tgt_tokens) + "\n")

        return X, Y
