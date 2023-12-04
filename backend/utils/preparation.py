import glob
import os
import pandas as pd
import numpy as np
import shutil
import librosa
from tqdm import tqdm


def extract_feature(file_name):
    """
    Extract feature from audio file `file_name`
        Feature used:
            - MEL Spectrogram Frequency (mel)
        e.g:
        `features = extract_feature(path, mel=True)`
    """


    X, sample_rate = librosa.core.load(file_name)

    result = np.array([])

    mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
    mel_mean = np.mean(mel.T, axis=0)
    result = np.hstack((result, mel_mean))

    return result

def prepare():
    dirname = "../raw_data"

    if not os.path.isdir(dirname):
        os.mkdir(dirname)


    df = pd.read_csv("../raw_data/data_bn_in_train_wav.csv", header=0)
    print("Previously:", len(df), "rows")
    # take only male & female genders (i.e droping NaNs & 'other' gender)
    new_df = df[np.logical_or(df['gender'] == 'FEMALE', df['gender'] == 'MALE')]
    print("Now:", len(new_df), "row")
    new_csv_file = os.path.join(dirname, "data_bn_in_train_npy.csv")
    # save new preprocessed CSV
    new_df_npy = new_df.copy()
    new_df_npy['filename'] = "../raw_data/train_npy/" + new_df_npy['filename'].str.replace('.wav', '.npy')
    new_df_npy.to_csv(new_csv_file, index=False)
    # get the folder name
    audio_files = glob.glob(f"{dirname}/train/*")
    all_audio_filenames = set(new_df["filename"])

    for i, audio_file in tqdm(list(enumerate(audio_files)), f"Extracting features of {dirname}/train/"):
        splited = os.path.split(audio_file)
        # audio_filename = os.path.join(os.path.split(splited[0])[-1], splited[-1])
        audio_filename = f"{splited[-1]}"
        # print("audio_filename:", audio_filename)
        if audio_filename in all_audio_filenames:
            # print("Copyying", audio_filename, "...")
            src_path = f"../raw_data/train/{audio_filename}"
            target_path = f"{dirname}/train_npy"
            #create that folder if it doesn't exist
            if not os.path.isdir(os.path.dirname(target_path)):
                os.mkdir(os.path.dirname(target_path))
            features = extract_feature(src_path)
            target_filename = audio_filename.split(".")[0]
            np.save(f"{target_path}/{target_filename}", features)
            # shutil.copyfile(src_path, target_path)