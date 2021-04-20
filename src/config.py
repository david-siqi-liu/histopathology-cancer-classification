args = {
    'seed': 647,
    'model_output_dir': 'trained/',
    'data': {
        'train': {
            'LUAD': 'data/extractedtrainfeatures/extracted_features_LUAD.pickle',
            'LUSC': 'data/extractedtrainfeatures/extracted_features_LUSC.pickle',
            'MESO': 'data/extractedtrainfeatures/extracted_features_MESO_256.pickle'
        },
        'dev': {
            'LUAD': 'data/extracteddevfeatures/extracted_features_LUAD_dev.pickle',
            'LUSC': 'data/extracteddevfeatures/extracted_features_LUSC_dev.pickle',
            'MESO': 'data/extracteddevfeatures/extracted_features_MESO_dev.pickle'
        }
    },
    'max_patch_count': 10,
    'num_train_sets': {
        'LUAD': int(1e4),
        'LUSC': int(1e4),
        'MESO': int(1e4)
    },
    'num_val_sets': {
        'LUAD': int(1e3),
        'LUSC': int(1e3),
        'MESO': int(1e3)
    },
    'num_dev_sets': 11
}
