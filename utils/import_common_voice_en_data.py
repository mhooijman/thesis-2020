import json
from sox import Transformer
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas
import unicodedata


data_dir = './data'
all_data = open(data_dir + '/CommonVoice/validated.tsv', 'r').read().split('\n')
train_data = open(data_dir + '/CommonVoice/train.tsv', 'r').read().split('\n')
keys = all_data.pop(0).split('\t')
items = [dict(zip(keys, line.split('\t'))) for line in all_data]
train_items = [dict(zip(keys, line.split('\t'))) for line in train_data]

train_paths = [item['path'] for item in train_items if 'path' in item]
sentences = {item['path']:item['sentence'] for item in items if 'sentence' in item and int(item['down_votes']) < 1 and int(item['up_votes']) >= 3}

for path in train_paths:
    try:
        del sentences[path]
    except:
        pass

sentences = list(sentences.values())

result = {}
male = 0
female = 0
for d in items:
    if 'sentence' in d and int(d['down_votes']) < 1 and int(d['up_votes']) >= 3 and d['gender'] in ['male', 'female']:
        if d['sentence'] not in result: result[d['sentence']] = []
        result[d['sentence']].append(d)

        if d['gender'] == 'male': male+=1
        elif d['gender'] == 'female': female+=1
        else:
            other+=1


selection = [{'set_id':i, 'set_length':len(item), 'set_items':item} for i, item in enumerate(list(result.values())) if len(item) >= 10]
json.dump(selection, open('./data/pertubed_input_sets.json', 'w+'), indent=4)

print('{} sets found (male: {}, female: {}.).'.format(len(selection), male, female))

print('Balance out gender in each set...')
balanced_sets = []
for set in selection:
    male = 0
    female = 0
    for item in set['set_items']:
        if item['gender'] == 'male': male+=1
        if item['gender'] == 'female': female+=1
    items_balanced = []
    if male < female: n = male
    else: n = female
    male = 0
    female = 0
    for item in set['set_items']:
        if item['gender'] == 'male' and male < n:
            items_balanced.append(item)
            male+=1
        if item['gender'] == 'female' and female < n:
            items_balanced.append(item)
            female+=1
    if len(items_balanced) >= 10:
        set['set_items'] = items_balanced
        set['set_length'] = len(items_balanced)
        balanced_sets.append(set)

json.dump(balanced_sets, open('./data/pertubed_input_sets_balanced.json', 'w+'), indent=4)


# Preprocess audio files
source_dir = data_dir + '/CommonVoice/clips'
target_dir = data_dir + '/CommonVoice/pertubed_sets'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

files = []
SAMPLE_RATE = 16000
for set in selection:
    for item in set['set_items']:
        filename =  source_dir + '/' + item['path']
        transcript = item['sentence']
        transcript = unicodedata.normalize('NFKD', transcript)  \
                                .encode('ascii', 'ignore')      \
                                .decode('ascii', 'ignore')

        transcript = transcript.lower().strip()

        # Convert corresponding MP3 to a WAV
        mp3_file =  source_dir + '/' + item['path']
        wav_file = target_dir + '/' + item['path'].replace('.mp3', '.wav')

        if not os.path.exists(wav_file):
            tfm = Transformer()
            tfm.set_output_format(rate=SAMPLE_RATE)
            tfm.build(mp3_file, wav_file)
        wav_filesize = os.path.getsize(wav_file)

        files.append((os.path.abspath(wav_file), wav_filesize, transcript))

data_info = pandas.DataFrame(data=files, columns=['wav_filename', 'wav_filesize', 'transcript'])
data_info.to_csv(data_dir + '/common-voice-pertubed_sets.csv', index=False)
