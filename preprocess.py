from scipy.io import loadmat
from scipy.io import savemat
import torch
import numpy as np
import glob
#import madmom
import os
import configurations.preprocessing_parameters as ppp
import warnings
from joblib import Parallel, delayed
import multiprocessing
#from madmom.io import midi
from enum import Enum
warnings.filterwarnings("ignore")


class Fold(Enum):
    """Distinguish the different folds the model is trained on."""
    fold_1 = 0
    fold_2 = 1
    fold_3 = 2
    fold_4 = 3
    fold_benchmark = 4
    fold_single_note = 5


def wav_to_spec(base_dir, filename, _audio_options):
    """Transforms the contents of a wav file into a series of spec frames."""
    audio_filename = os.path.join(base_dir, filename + '.wav')

    spec_type, audio_options = get_spec_processor(_audio_options, madmom.audio.spectrogram)

    # it's necessary to cast this to np.array, b/c the madmom-class holds references to way too much memory
    spectrogram = np.array(spec_type(audio_filename, **audio_options))
    return spectrogram


def wav_to_hpcp(base_dir, filename):
    """Transforms the contents of a wav file into a series of spec frames."""
    audio_filename = os.path.join(base_dir, filename + '.wav')
    audio_options = ppp.get_hpcp_parameters()
    fmin = audio_options['fmin']
    fmax = audio_options['fmax']
    hpcp_processor = getattr(madmom.audio.chroma, 'HarmonicPitchClassProfile')
    audio_options['fmin'] = fmin[0]
    audio_options['fmax'] = fmax[0]
    hpcp = np.array(hpcp_processor(audio_filename, **audio_options))

    for index in range(1, 7):
        audio_options['fmin'] = fmin[index]
        audio_options['fmax'] = fmax[index]
        hpcp = np.append(hpcp, np.array(hpcp_processor(audio_filename, **audio_options)), axis=1)
    audio_options['fmin'] = fmin[-1]
    audio_options['fmax'] = fmax[-1]
    #audio_options['num_classes'] = 8
    hpcp = np.append(hpcp, np.array(hpcp_processor(audio_filename, **audio_options)[:, :int(audio_options['num_classes']/3)]), axis=1)
    # post-processing,
    # normalize hpcp by max value per frame. Add a small value to avoid division by zero
    #norm_vec = np.max(hpcp, axis=1) + 1e-7

    #hpcp = hpcp/norm_vec[:, None]
    hpcp = np.log10(hpcp + 1.0)
    hpcp = hpcp/np.max(hpcp)
    return hpcp


def get_spec_processor(_audio_options, madmom_spec):
    """Returns the madmom spectrogram processor as defined in audio options."""
    audio_options = dict(_audio_options)

    if 'spectrogram_type' in audio_options:
        spectype = getattr(madmom_spec, audio_options['spectrogram_type'])
        del audio_options['spectrogram_type']
    else:
        spectype = getattr(madmom_spec, 'LogarithmicFilteredSpectrogram')

    if 'filterbank' in audio_options:
        audio_options['filterbank'] = getattr(madmom_spec, audio_options['filterbank'])
    else:
        audio_options['filterbank'] = getattr(madmom_spec, 'LogarithmicFilterbank')

    return spectype, audio_options


def midi_to_groundtruth(base_dir, filename, dt, n_frames, is_chroma=False):
    """Computes the frame-wise ground truth from a piano midi file, as a note or chroma vector."""
    midi_filename = os.path.join(base_dir, filename + '.mid')
    notes = midi.load_midi(midi_filename)
    ground_truth = np.zeros((n_frames, 12 if is_chroma else 88)).astype(np.int64)
    for onset, _pitch, duration, velocity, _channel in notes:
        pitch = int(_pitch)
        frame_start = int(np.round(onset / dt))
        frame_end = int(np.round((onset + duration) / dt))
        label = np.mod(pitch - 21, 12) if is_chroma else pitch - 21
        ground_truth[frame_start:frame_end, label] = 1
    return ground_truth


def preprocess_fold(fold, mode, norm=False):
    """Preprocess an entire fold as defined in the preprocessing parameters.
        fold - Fold.fold_1, Fold.fold_2, Fold.fold_3, Fold.fold_4, Fold.fold_benchmark
        mode - 'train', 'valid' or 'test' to address the correct config parameter
    """
    config = ppp.get_preprocessing_parameters(fold.value)
    audio_config = config['audio_config']

    # load fold
    filenames = open(config[mode+'_fold'], 'r').readlines()
    filenames = [f.strip() for f in filenames]

    for file in filenames:
        # split file path string at "/" and take the last split, since it's the actual filename
        write_file_to_tfrecords(config['dataset_'+mode+'_fold'] + file.split('/')[-1],
                                config['audio_path'], file, audio_config, norm,
                                config['is_chroma'], config['is_hpcp'])


def preprocess_fold_parallel(fold, mode, norm=False):
    """Parallel preprocess an entire fold as defined in the preprocessing parameters.
        This seems only to work on Win with Anaconda!
        fold - Fold.fold_1, Fold.fold_2, Fold.fold_3, Fold.fold_4, Fold.fold_benchmark
        mode - 'train', 'valid' or 'test' to address the correct config parameter
    """
    config = ppp.get_preprocessing_parameters(fold.value)
    audio_config = config['audio_config']

    # load fold
    filenames = open(config[mode+'_fold'], 'r').readlines()
    filenames = [f.strip() for f in filenames]

    def parallel_loop(file):
        # split file path string at "/" and take the last split, since it's the actual filename
        write_file_to_tfrecords(config['dataset_'+mode+'_fold'] + file.split('/')[-1],
                                config['audio_path'], file, audio_config, norm,
                                config['is_chroma'], config['is_hpcp'])

    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cores)(delayed(parallel_loop)(file) for file in filenames)


def write_file_to_tfrecords(write_file, base_dir, read_file, audio_config, norm, is_chroma, is_hpcp):
    """Transforms a wav and mid file to features and writes them to a mat file."""
    if is_hpcp:
        spectrogram = wav_to_hpcp(base_dir, read_file)
    else:
        spectrogram = wav_to_spec(base_dir, read_file, audio_config)
    print(spectrogram.shape)
    ground_truth = midi_to_groundtruth(base_dir, read_file, 1. / audio_config['fps'], spectrogram.shape[0], is_chroma)

    # re-scale spectrogram to the range [0, 1]
    if norm:
        spectrogram = np.divide(spectrogram, np.max(spectrogram))

    savemat(write_file, {"features": spectrogram, "labels": ground_truth})


def stage_dataset():
    chunk = 2000
    inference_chunk = 10000
    train_files = glob.glob("./dataset/sigtia-configuration2-splits/fold_benchmark/train/*.mat")
    valid_files = glob.glob("./dataset/sigtia-configuration2-splits/fold_benchmark/valid/*.mat")
    test_files = glob.glob("./dataset/sigtia-configuration2-splits/fold_benchmark/test/*.mat")
    train_features = []
    train_labels = []
    valid_features = []
    valid_labels = []
    test_features = []
    test_labels = []
    for file in train_files:
        data = loadmat(file)
        #train_features.append(data["features"])
        tensor_features = torch.Tensor(data["features"].astype(np.float64))
        train_features.extend(tensor_features.split(chunk, dim=0))
        #train_labels.append(data["labels"])
        tensor_labels = torch.Tensor(data["labels"].astype(np.float64))
        train_labels.extend(tensor_labels.split(chunk, dim=0))
    for file in valid_files:
        data = loadmat(file)
        # train_features.append(data["features"])
        tensor_features = torch.Tensor(data["features"].astype(np.float64))
        valid_features.extend(tensor_features.split(inference_chunk, dim=0))
        # train_labels.append(data["labels"])
        tensor_labels = torch.Tensor(data["labels"].astype(np.float64))
        valid_labels.extend(tensor_labels.split(inference_chunk, dim=0))
    for file in test_files:
        data = loadmat(file)
        # train_features.append(data["features"])
        tensor_features = torch.Tensor(data["features"].astype(np.float64))
        test_features.extend(tensor_features.split(inference_chunk, dim=0))
        # train_labels.append(data["labels"])
        tensor_labels = torch.Tensor(data["labels"].astype(np.float64))
        test_labels.extend(tensor_labels.split(inference_chunk, dim=0))

    #for data in [train_features, train_labels, valid_features, valid_labels]:
    #    for i in range(len(data)):
    #        data[i] = torch.Tensor(data[i].astype(np.float64))

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels


def data_generator(dataset):
    if dataset == "JSB":
        print('loading JSB data...')
        data = loadmat('./mdata/JSB_Chorales.mat')
    elif dataset == "Muse":
        print('loading Muse data...')
        data = loadmat('./mdata/MuseData.mat')
    elif dataset == "Nott":
        print('loading Nott data...')
        data = loadmat('./dataset/Nottingham.mat')
    elif dataset == "Piano":
        print('loading Piano data...')
        data = loadmat('./mdata/Piano_midi.mat')

    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]

    for data in [X_train, X_valid, X_test]:
        for i in range(len(data)):
            data[i] = torch.Tensor(data[i].astype(np.float64))

    return X_train, X_valid, X_test
