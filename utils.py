import pydub
import librosa
import numpy as np

from skimage.transform import resize


AMPLITUDE = 32767
MUSIC_DURATION = 30
NOISE_DURATION = 235


def overlay_signals(sig1, rate1, sig2, rate2):
    sig1_int32 = (sig1 * AMPLITUDE).astype(np.int16)
    sig2_int32 = (sig2 * AMPLITUDE).astype(np.int16)

    asegment1 = pydub.AudioSegment(
        sig1_int32.tobytes(),
        frame_rate=rate1,
        sample_width=2,
        channels=1
    )

    asegment2 = pydub.AudioSegment(
        sig2_int32.tobytes(),
        frame_rate=rate2,
        sample_width=2,
        channels=1
    )

    overlayed = asegment1.overlay(asegment2)
    sig = np.array(overlayed.get_array_of_samples()).reshape(-1,).astype(float) / AMPLITUDE
    rate = overlayed.frame_rate

    return  overlayed, sig, rate


def extract_features(signal, rate):
    win_length = calc_window_length(ms=10, rate=rate)
    chroma_stft = librosa.feature.chroma_stft(y=signal, sr=rate,
                                              win_length=win_length,
                                              window=hamming)
    spec_cent = librosa.feature.spectral_centroid(y=signal, sr=rate)
    spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=rate)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=rate)
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    mfcc = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=13)

    fvec = [
        np.mean(chroma_stft),
        np.mean(spec_cent),
        np.mean(spec_bw),
        np.mean(rolloff),
        np.mean(zcr)
    ] + [np.mean(e) for e in mfcc]

    return fvec


def extract_features2(signal, rate):
    chroma_stft = librosa.feature.chroma_stft(y=signal, sr=rate)
    rms = librosa.feature.rms(y=signal)
    spec_cent = librosa.feature.spectral_centroid(y=signal, sr=rate)
    spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=rate)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=rate)
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    harmonic, perceptr = librosa.effects.hpss(y=signal)
    tempo, _ = librosa.beat.beat_track(y=signal, sr=rate)
    mfcc = librosa.feature.mfcc(y=signal, sr=rate)
    features = [chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr,
                harmonic, perceptr, tempo] + [e for e in mfcc]

    fvec = []

    for feature in features:
        fvec.extend([np.mean(feature), np.var(feature)])

    return fvec


def calc_window_length(ms, rate):
    return int((ms * rate) / 1000)


def stratified_sampling(X, y, n_samples, shuffle=True):
    """Stratified sampling of vectors X and y according to categories in y.

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Numpy array of shape (N, ...) consisting of features or raw images. This array will be sampled alongside
        its first axis.
    y : :class:`numpy.ndarray`
        Numpy array of shape (N, ) consisting of labels. The categories within y will determine the sampling procedure.
    n_samples : int
        Number of samples per category of y.
    shuffle : bool
        Whether or not shuffle the final samples
    """
    N = len(y)
    categories, c = np.unique(y, return_counts=True)
    n = len(categories) * n_samples

    if N < n:
        raise ValueError("Expected y to have at least {} samples for 'n_samples' = {}".format(N, n))

    Xsampled = []
    ysampled = []
    for category in categories:
        ind_cat = np.where(y == category)[0]
        Xsampled.append(X[ind_cat[:n_samples]])
        ysampled.append(y[ind_cat[:n_samples]])

    Xsampled = np.concatenate(Xsampled, axis=0)
    ysampled = np.concatenate(ysampled, axis=0)

    if shuffle:
        ind = np.arange(len(Xsampled))
        np.random.shuffle(ind)
        Xsampled = Xsampled[ind]
        ysampled = ysampled[ind]

    return Xsampled, ysampled


def gen_crossval_indices(y, n_folds=5):
    """Generate crossvalidation indices with stratified sampling. Each index will be assigned to a fold, and on each
    fold, the distribution of classes is uniform.

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        Numpy array containing the labels of each sample
    n_folds : int
        Number of folds
    """
    n_samples = len(y)
    categories = np.unique(y)
    n_categories = len(categories)
    if n_samples % n_folds != 0:
        raise ValueError("""Expected number of samples ({}) to be divisible by number of folds ({})
                         """.format(n_samples, n_folds))

    # Example:
    # n_samples = 5000
    # n_folds = 5
    # n_categories = 10
    # n_samples_per_fold = 5000 / 5 = 1000
    # n_samples_per_class = 1000 / 10 = 100
    n_samples_per_fold = n_samples // n_folds
    n_samples_per_class = n_samples_per_fold // n_categories
    indices_crossval = np.zeros([n_samples,])

    for category in range(n_categories):
        # Loops over each class
        ind_category = np.where(y == category)[0]
        for fold in range(n_folds):
            # Loops over each fold
            ind_fold = ind_category[fold * n_samples_per_class: (fold + 1) * n_samples_per_class]
            indices_crossval[ind_fold] = fold

    return indices_crossval


def resize_batch(image_batch, image_shape):
    """Resizes batch of images

    Parameters
    ----------
    image_batch : :class:`numpy.ndarray`
        Numpy array of shape (batch_size, height, width) if images are grayscale or
        (batch_size, height, width, channels) if images are RGB.
    image_shape : :class:`numpy.ndarray`
        Numpy array of shape (2,) containing the new height and new width.
    """
    X = np.zeros([image_batch.shape[0], *image_shape])
    for i in range(len(X)):
        im = image_batch[i].copy()
        im = resize(im, image_shape, anti_aliasing=True)
        X[i] = im.copy()
    return X
