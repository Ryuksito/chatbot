from src import WordPieceTokenizer
import os
import pandas as pd
import json

vocab_path = '../data/vocab.txt'
vocab_metadata_path = '../data/metadata.json'
text_path = '../data/text.txt'
instructions_path = '../data/instructions.json'

tokenizer = WordPieceTokenizer()



def make_corpus():
  corpus = []
  all_text = ''


  with open(instructions_path, 'r', encoding='utf8') as f:
    instructions = json.load(f)

  for i, data in enumerate(instructions['data']):
    all_text += '\n' + data['instruction'] + ' ' + data['answer']

  for i in range(0, len(all_text), 400):
    corpus.append(''.join(all_text[i:i+400]))

  return corpus

def tokenize_all_text():
  corpus = make_corpus()

  tokenizer.adapt(corpus, 10000)
  
  with open(vocab_path, 'w', encoding='utf-8') as file:
    file.write('\n'.join(tokenizer.get_vocab()))

  with open(vocab_metadata_path, 'w', encoding='utf-8') as file:
    json.dump(tokenizer.get_metadata(), file, indent=4)

if __name__ == '__main__':
  if not os.path.exists(vocab_path):
    tokenize_all_text()
    print('Process finished...')
  else: 
    with open(vocab_path, 'r', encoding='utf-8') as file: 
      vocab = file.read().split('\n')

    tokenizer.load_vocab(vocab)

    print('Vocab file already exists...')

