import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from IPython.display import Audio
import nltk  # we'll use this to split into sentences
import numpy as np

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE


preload_models()

script = """
Yes, children who participate in team sports will develop better socially.
This is because the children in a team, they need to learn how to maintain good relationships.
So, they can develop better socially in the future.
For example, my friend Alex, he is a nice guy, and when he was a junior high school student, he joined a basketball team.
to maintain a very great relationship with his teammates to win the competition.
So, because of this experience, now he has very great social skills.
""".replace("\n", " ").strip()

sentences = nltk.sent_tokenize(script)
SPEAKER = "v2/en_speaker_6"
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

pieces = []
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]

Audio(np.concatenate(pieces), rate=SAMPLE_RATE)
