import mne

mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)


def imagined_speech_eeg(path, band_filter, low_freq, high_freq):
    raw_data = mne.io.read_raw_cnt(path, preload=True).load_data()
    raw_data.pick_channels(['FC6', 'FT8', 'C5', 'CP3', 'P3', 'T7', 'CP5', 'C3', 'CP1', 'C4'])
    raw_data = raw_data.copy()
    if band_filter:
        raw_data = raw_data.filter(low_freq, high_freq, n_jobs=4)
        return raw_data.get_data()
    return raw_data.get_data()


print(imagined_speech_eeg(path="Data/p/Acquisition 283 Data.cnt", band_filter=True, low_freq=4., high_freq=40.).shape)


