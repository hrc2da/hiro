import os
from gtts import gTTS
import argparse
import string

ap = argparse.ArgumentParser()
ap.add_argument('text')
args = ap.parse_args()

if args.text == None:
    raise('Must pass text to record')

fname = f"{args.text.translate(str.maketrans('', '', string.punctuation)).replace(' ','_')}.mp3"

tts = gTTS(text=args.text, lang='en')
tts.save(fname)
os.system(f'mpg321 {fname}')