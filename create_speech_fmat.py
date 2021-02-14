# Wasserstein Barycenter Transport for Multi-source Domain Adaptation
#
# References
# ----------
# [1] Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals.
#     IEEE Transactions on speech and audio processing, 10(5), 293-302.
# [2] http://spib.linse.ufsc.br/noise.html
# [3] Turrisi, R., Flamary, R., Rakotomamonjy, A., & Pontil, M. (2020). Multi-source Domain
#     Adaptation via Weighted Joint Distributions Optimal Transport. arXiv preprint arXiv:2006.12938.

import os
import pydub
import librosa
import argparse
import numpy as np

from utils import overlay_signals, extract_features2

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--noise_path', type=str, help="""Path to noise files""")
parser.add_argument('--music_path', type=str, help="""Path to music files""")
args = parser.parse_args()

NOISE_PATH = args.noise_path
MUSIC_PATH = args.music_path
NOISE_TYPES = [
    None, # Original files
    "buccaneer2",
    "destroyerengine",
    "f16",
    "factory2"
]
GENRES = os.listdir(MUSIC_PATH)
MUSIC_DURATION = 30 # Following [1]
NOISE_DURATION = 235 # Following [2]
fmat = []

i = 0
for ndomain, noise_type in enumerate(NOISE_TYPES):
    for nclass, genre in enumerate(GENRES):
        gen_dir = os.path.join(MUSIC_PATH, genre)
        filenames = os.listdir(gen_dir)
        for filename in filenames:
            print("Processing file {}".format(i))
            print("Reading filename {} from {}".format(filename, gen_dir))
            print("Genre: {}, Noise: {}".format(genre, noise_type))
            print("Class: {}, Domain: {}".format(nclass, ndomain))
            try:
                (sig, rate) = librosa.load(os.path.join(gen_dir, filename), mono=True, duration=MUSIC_DURATION)
            except:
                print("Error while reading file {}".format(filename))

            if noise_type is not None:
                (noise, nrate) = librosa.load(os.path.join(NOISE_PATH, noise_type + '.wav'), mono=True, duration=NOISE_DURATION)
                _, sig, rate = overlay_signals(sig1=sig, rate1=rate, sig2=noise, rate2=nrate)

            fvec = extract_features2(sig, rate)
            print("fvec shape: ", len(fvec))
            fvec += [nclass, ndomain]
            fmat.append(fvec)
            i += 1
fmat = np.array(fmat)

np.save('./data/MusicSpeech.npy', fmat)
