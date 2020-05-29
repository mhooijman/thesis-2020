import csv
from shutil import copyfile
from utils import create_dir_if_not_exists


def prepare_speaker_data(file_path):
    speakers = open(file_path, 'r')
    speakers_data = {
        l.split('|')[0].strip(): l.split('|')[1].strip() 
        for l in speakers.readlines() if not l.startswith(';')
                                            and l.split('|')[2].strip() == 'test-clean'}
    return speakers_data

def get_input_meta_data(file_path):
    with open(file_path) as f:
        records = csv.DictReader(f)
        return [dict(r) for r in records]

meta = get_input_meta_data('./data/librivox-test-clean.csv')
meta_sorted_by_filezize = sorted(meta, key=lambda k: int(k['wav_filesize']), reverse=False)
speaker_data = prepare_speaker_data('./data/LibriSpeech/SPEAKERS.TXT')

m = 0
f = 0
n_select = 100
selected_files = []
while m < int(n_select/2) or f < int(n_select/2):
    item = meta_sorted_by_filezize.pop(0)
    filename = item['wav_filename'].split('/')[-1]
    gender = speaker_data[filename.split('-')[0]]
    if gender == 'M' and m < 50:
        selected_files.append(filename)
        m+=1

    if gender == 'F' and f < 50:
        selected_files.append(filename)
        f+=1

source = './data/LibriSpeech/test-clean-wav'
dest = './data/LibriSpeech/test-clean-wav-100'
create_dir_if_not_exists(dest)
for f in selected_files:
    copyfile('{}/{}'.format(source, f), '{}/{}'.format(dest, f))
