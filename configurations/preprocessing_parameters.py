def get_preprocessing_parameters(fold_num):
    splits = ['sigtia-configuration2-splits/fold_1',
              'sigtia-configuration2-splits/fold_2',
              'sigtia-configuration2-splits/fold_3',
              'sigtia-configuration2-splits/fold_4',
              'sigtia-configuration2-splits/fold_benchmark',
              'single-note-splits']

    split = splits[fold_num]

    config = {'audio_path': '../../MAPS',
              'train_fold': './splits/{}/train'.format(split),
              'valid_fold': './splits/{}/valid'.format(split),
              'test_fold': './splits/{}/test'.format(split),
              'dataset_train_fold': './dataset/{}/train/'.format(split),
              'dataset_valid_fold': './dataset/{}/valid/'.format(split),
              'dataset_test_fold': './dataset/{}/test/'.format(split),
              'chord_folder': './tfrecords-dataset/chords/',
              'chord_fold': './splits/chord-splits/train',
              'chroma_folder': './chroma/',
              'is_chroma': False,
              'is_hpcp': True,
              'audio_config': {'num_channels': 1,
                               'sample_rate': 44100,
                               'filterbank': 'LogarithmicFilterbank',
                               'frame_size': 4096,
                               'fft_size': 4096,
                               'fps': 100,
                               'num_bands': 48,
                               'fmin': 30.0,
                               'fmax': 4200.0,
                               'fref': 440.0,
                               'norm_filters': True,
                               'unique_filters': True,
                               'circular_shift': False,
                               'norm': True}
              }
    return config


def get_hpcp_parameters():
    config = {'num_channels': 1,
              'sample_rate': 44100,
              'frame_size': 4096,
              'fft_size': 4096,
              'fps': 100,
              'num_classes': 12,
              'fmin': [27.5, 54.0, 107.0, 215.0, 426.0, 856.0, 1701.0, 3423.0], #[27.5, 54.0, 107.0, 215.0, 426.0, 856.0, 1701.0, 3423.0]
              'fmax': [53.0, 106.0, 214.0, 425.0, 855.0, 1700.0, 3422.0, 6644.9], #[53.0, 106.0, 214.0, 425.0, 855.0, 1700.0, 3422.0, 6644.9]
              'fref': 440.0,
              'window': 1,
              'norm_filters': False,
              'circular_shift': False,
              'norm': True}
    return config
