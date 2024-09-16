from pickle import FALSE
from collections import defaultdict
from typing import List, Dict
import re
from tqdm import tqdm

_unwanted_chars = """­Ö¢«/\ê$ìà#Ù†ë™ä·ã³'±]_›>´çö¹¬®½:ÃØÈ¶[&ôËù¾€ò»Ò„üß0²ªè|°@{îºæ}^Çï5õ¨+âøÀ"%*<=`~£¥©µ¼Å×ð÷ûĀāēěīıłōœşūżǒǜȳəɛɪʃˆˈ˚̩́̄ΓΔΣΩαβγδεθκλμνπρσψगयो​‍–—‖‘’“”•…⁰⁴⁷⁺⁻₀₁₂₄₆₹℃℉⅓⅔←→↓↔⇌∂∃∑−∙√∞∠∣∧∨∩≈≠≡≤≥⋅⏰─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬█░▓▼☀☁★☕☠♀♂♕♖♙♛♜♟♻⚠⚽⛰✅✈✨❄❤➕➖➡⭐。「」あいうがくござすたちとはまりるを一仁千困坂夕夢天太失子強得徳必愚明智月有本標準皇目私者聖虑語難馬龍ﬁ️，；�🆕🇦🇧🇨🇩🇪🇫🇬🇭🇮🇯🇰🇱🇲🇳🇴🇵🇶🇷🇸🇹🇺🇽🇿🌀🌅🌊🌍🌎🌏🌞🌟🌧🌱🌲🌳🌴🌸🌻🌿🍃🍊🍌🍎🍓🍕🍞🍩🍪🍯🍳🍴🍿🎂🎈🎉🎊🎓🎥🎧🎨🎬🎮🎵🎶🎼🎾🏀🏃🏈🏊🏒🏖🏡🏻🏽🏾🐈🐘🐛🐥🐨🐮🐰🐱🐶🐹🐾👇👊👌👏👓👠👨👩💆💉💕💖💙💚💡💦💪💫💯💰💳💸💻📈📖📚📢📣📲📺🔍🔒🔥🔹🕷🗳🗺😀😁😂😆😈😊😋😍😎😔😱😴🙌🙏🚀🚂🚆🚇🚌🚗🚨🚪🚰🚴🚶🛍🟢🤖🤝🤩🥕🥦🦁🦇🦊🦸🧁🧘🧴🧸🧹🧼🪀"""

class WordPieceTokenizer:

    def __init__(self, part_of_word_token:str='<pow>', unk_token:str='<unk>', spe_tokens:List[str]=['', '<pad>', '<cls>', '<sep>', '<mask>']):
        self.word_freqs: defaultdict
        self.alphabet: List[str]
        self.vocab: List[str]
        self.splits: Dict[str, List[str]]
        self.vocab: List[str]
        self.part_of_word_token = part_of_word_token
        self.unk_token = unk_token
        self.spe_tokens: List[str] = spe_tokens + [unk_token, part_of_word_token]

    def get_vocab(self):
      return self.vocab

    def load_vocab(self, vocab):
      self.vocab = vocab

    def get_metadata(self):
        return {
            "vocab_size": self.vocab_size,
            "part_of_word_token": self.part_of_word_token,
            "unk_token": self.unk_token,
            "spe_tokens": self.spe_tokens
        }

    def load_metadata(self, metadata:dict):
        self.vocab_size = metadata.get("vocab_size", self.vocab_size)
        self.part_of_word_token = metadata.get("part_of_word_token", self.part_of_word_token)
        self.unk_token = metadata.get("unk_token", self.unk_token)
        self.spe_tokens = set(metadata.get("spe_tokens", self.spe_tokens))

    def adapt(self, corpus: List[str], vocab_size:int):
        self.vocab_size: int = vocab_size

        self.word_freqs = self._get_word_freqs(corpus)
        self.alphabet = self._get_alphabet()
        self.vocab = self._set_vocab()
        self.splits = self._get_split_words()
        self.vocab = self._make_vocab()

    def pretokenize(self, text: List[str], unwant_chars:str = _unwanted_chars, to_lower=False):
        tokens = [re.findall(r'\w+|[^\w\s]', self.clean_text(sentence, unwant_chars), re.UNICODE) for sentence in text]
        return tokens

    def clean_text(self, text:str, unwant_chars:str=_unwanted_chars, to_lower=False):
        if to_lower: text = text.lower()

        regex = f"[{re.escape(unwant_chars)}]"
        text = re.sub(regex, "", text)
        text = re.sub(r"[\n\t\r]+", ' ', text)
        text = re.sub(r"\s+", ' ', text)
        text = re.sub(r"\.{3,}", "...", text)
        text = re.sub(r"\.\.+", ".", text)

        text = text.strip()
        return text

    def encode_word(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return [self.unk_token]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f'{self.part_of_word_token}{word}'
        return tokens

    def tokenize(self, text: List[str]):
        tokens = []
        text = self.pretokenize(text)
        for sentence in text:
            sentence_tokens = [token for word in sentence for token in self.encode_word(word)]
            tokens.append(sentence_tokens)
        return tokens

    def encode(self, text: List[str], max_tokens: int = None):
        tokens = self.tokenize(text)

        if not hasattr(self, 'vocab_to_index'):
            self.vocab_to_index = {token: idx for idx, token in enumerate(self.vocab)}

        encoded_text = []
        for sentence in tokens:
            if max_tokens:
                encoded_sentence = [
                    self.vocab_to_index.get(token, self.vocab_to_index.get(self.unk_token, 0))
                    for token in sentence[:max_tokens]
                ]
                if len(encoded_sentence) < max_tokens:
                  encoded_sentence += [0] * (max_tokens - len(encoded_sentence))
            else:
                encoded_sentence = [
                    self.vocab_to_index.get(token, self.vocab_to_index.get(self.unk_token, 0))
                    for token in sentence
                ]

            encoded_text.append(encoded_sentence)



        return encoded_text

    def decode(self, encoded_text: List[List[int]]):
        if not hasattr(self, 'index_to_vocab'):
            self.index_to_vocab = {idx: token for idx, token in enumerate(self.vocab)}

        decoded_text = []
        for sentence in encoded_text:
            decoded_sentence = [self.index_to_vocab.get(index, self.unk_token) for index in sentence]
            joined_sentence = self._join_tokens(decoded_sentence)
            decoded_text.append(joined_sentence)

        return decoded_text

    def _join_tokens(self, tokens: List[str]):
        sentence = ""
        for token in tokens:
            if token.startswith(self.part_of_word_token) or token == self.spe_tokens[0]:
                sentence += token[len(self.part_of_word_token):]
            else:
                if sentence:
                    sentence += " "
                sentence += token
        return sentence

    def _get_word_freqs(self, corpus):
      word_freqs = defaultdict(int)
      for text in self.pretokenize(corpus):
          for word in text:
              word_freqs[word] += 1
      return word_freqs

    def _get_alphabet(self):
      alphabet = []
      for word in self.word_freqs.keys():
          if word[0] not in alphabet:
              alphabet.append(word[0])
          for letter in word[1:]:
              if f"{self.part_of_word_token}{letter}" not in alphabet:
                  alphabet.append(f"{self.part_of_word_token}{letter}")
      alphabet.sort()
      return alphabet

    def _set_vocab(self):
      vocab = self.spe_tokens + self.alphabet.copy()
      return vocab

    def _get_split_words(self):
      splits = {
        word: [c if i == 0 else f"<pow>{c}" for i, c in enumerate(word)] for word in self.word_freqs.keys()
      }
      return splits

    def _compute_pair_scores(self):
      letter_freqs = defaultdict(int)
      pair_freqs = defaultdict(int)
      for word, freq in self.word_freqs.items():
          split = self.splits[word]
          if len(split) == 1:
              letter_freqs[split[0]] += freq
              continue
          for i in range(len(split) - 1):
              pair = (split[i], split[i + 1])
              letter_freqs[split[i]] += freq
              pair_freqs[pair] += freq
          letter_freqs[split[-1]] += freq

      scores = {
          pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
          for pair, freq in pair_freqs.items()
      }
      return scores

    def _merge_pair(self, a, b):
      for word in self.word_freqs:
          split = self.splits[word]
          if len(split) == 1:
              continue
          i = 0
          while i < len(split) - 1:
              if split[i] == a and split[i + 1] == b:
                  merge = a + b.replace(self.part_of_word_token, "", 1) if b.startswith("<pow>") else a + b
                  split = split[:i] + [merge] + split[i + 2 :]
              else:
                  i += 1
          self.splits[word] = split
      return self.splits

    def _combine_tokens(self, token_pair):
      tokens = []

      for i, token in enumerate(token_pair):

          if token.startswith(self.part_of_word_token):
            tokens.append(token.replace(self.part_of_word_token, "", 1))
          else:
            tokens.append(token)
      combined_token = ''.join(tokens)

      return combined_token

    def _make_vocab(self):

      with tqdm(total=self.vocab_size, desc="Procesing") as pbar:
        vocab_len = 0
        while vocab_len < self.vocab_size:
            try:
                scores = self._compute_pair_scores()
                best_pair, max_score = "", None
                for pair, score in scores.items():
                    if max_score is None or max_score < score:
                        best_pair = pair
                        max_score = score
                self.splits = self._merge_pair(*best_pair)
                new_token = (
                    best_pair[0] + best_pair[1].replace(self.part_of_word_token, "", 1)
                    if best_pair[1].startswith(self.part_of_word_token)
                    else best_pair[0] + best_pair[1]
                )
                self.vocab.append(new_token)

                vocab_len = len(self.vocab)
                delta = vocab_len - pbar.n

                pbar.update(delta)


            except:
                break

      return self.vocab
