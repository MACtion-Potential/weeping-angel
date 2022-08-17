bluetooth = True

if __name__ == "__main__":
    import pandas as pd
    # Keep a dictionary so we can index each dataset by name

    name = input("Enter your username: ")

    raw_recordings = {
                    f"{name}_LeftWinks_60s_2s": pd.read_csv(f"Data/{name}/{name}_LeftWinks_60s_2s_1.csv"),
                    f"{name}_RightWinks_60s_2s": pd.read_csv(f"Data/{name}/{name}_RightWinks_60s_2s_1.csv"),
                    f"{name}_NormalBlinks_60s_2s": pd.read_csv(f"Data/{name}/{name}_NormalBlinks_60s_2s_3.csv"),
                    f"{name}_FewBlinks_60s_5s": pd.read_csv(f"Data/{name}/{name}_FewBlinks_60s_5s_1.csv")
                    }
    
    # Import MNE
    import mne
    # Tell MNE to not bother us with outputs unless something's wrong
    mne.set_log_level("WARNING")

    # Create the info object that stores our metadata
    info = mne.create_info(["timestamps","TP9","AF7","AF8", "TP10", "Right AUX"], 256, ch_types="eeg")

    # Create the MNE raw data object using the raw data (.values.T to extract the values only, and transpose to make it in the format MNE wants)
    raw_data = {dataset_name : mne.io.RawArray(raw_recordings[dataset_name].values.T/1000000, info) for dataset_name in raw_recordings.keys()}
    #notchfiltered_data = {raw_data[dataset_name].notch_filter([60, 120], picks=["TP9","AF7","AF8", "TP10"]) for dataset_name in raw_data.keys()}
    #bandpassfiltered_data = {raw_data[dataset_name].filter(0, 70, picks=["TP9","AF7","AF8", "TP10"], method="fir") for dataset_name in raw_data.keys()}

    # Create the MNE Epochs object using the Blink column
    epoched_data = {dataset_name: mne.preprocessing.create_eog_epochs(raw_data[dataset_name], ch_name=["TP9","AF7","AF8", "TP10"]) for dataset_name in raw_recordings.keys()}

    # Create the MNE Evoked object using the epochs
    evoked_data = {dataset_name : epoched.average(picks="all") for dataset_name, epoched in epoched_data.items()}


    import matplotlib.pyplot as plt
    import utils

    # Grab data for Alex
    user_dataset = {key: raw_data[key] for key in (
        "Alex_LeftWinks_60s_2s", 
        "Alex_RightWinks_60s_2s", 
        "Alex_NormalBlinks_60s_2s", 
        "Alex_FewBlinks_60s_5s"
        )}
    X, Y = utils.prepare_data(
        user_dataset=user_dataset, 
        duration_of_blink=1, 
        sampling_frequency=256, 
        duration_before_peak=0.25, 
        jitter=0.05
        )
    
    import numpy as np
    import seaborn as sns
    from sklearn import svm, model_selection, preprocessing, metrics

    import utils
    from scipy import signal

    b, a = utils.create_butterworth_filter((10, 50,), 256, order=3, filter_type="bandpass")
    #X_SVM = np.array([utils.compute_features(x.values, (b,a)) for x in X])
    X_SVM = np.array([utils.compute_features(x.values, (b,a), pca=None, use_original=False) for x in X])
    Y_SVM = Y

    # Do a 3-fold cross-validation to get an idea how the model works on the whole dataset
    train_test_splitter = model_selection.KFold(n_splits=3, shuffle=True, random_state=123)
    # Fit the classifier on the data
    for train_index, test_index in train_test_splitter.split(X_SVM):
        # Initialize the classsifier (you can tweak hyperparameters here)
        classifier = svm.SVC(kernel="linear", probability=True)
        X_train, X_test = X_SVM[train_index], X_SVM[test_index]
        Y_train, Y_test = Y_SVM[train_index], Y_SVM[test_index]
        classifier.fit(X_train, Y_train)
    classifier = svm.SVC(kernel="linear", probability=True)
    classifier.fit(X_SVM, Y_SVM)
    print(f"FINAL MODEL")
    prediction_probabilities = classifier.predict_proba(X_SVM)
    print(f"Classifier Accuracy: {classifier.score(X_test, Y_test)}")
    print(f"Classifier AUC: {metrics.roc_auc_score(y_true=Y_SVM, y_score=prediction_probabilities, multi_class='ovr')}")

    import numpy as np  # Module that simplifies computations on matrices
    import matplotlib.pyplot as plt  # Module used for plotting
    from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
    import muselsl_utils as muselsl_utils  # Our own utility functions
    import utils
    import serial
    import time

    # Initialize the bluetooth stream to the Arduino
    if bluetooth: serial_port = serial.Serial("COM5", baudrate=9600, timeout=1)

    # Handy little enum to make code more readable
    class Band:
        Delta = 0
        Theta = 1
        Alpha = 2
        Beta = 3


    blink_types = ["Right Wink", "Left Wink", "Normal Blink", "Nothing"]

    """ EXPERIMENTAL PARAMETERS """
    # Modify these to change aspects of the signal processing

    # Length of the EEG data buffer (in seconds)
    # This buffer will hold last n seconds of data and be used for calculations
    BUFFER_LENGTH = 5

    # Length of the epochs used to compute the FFT (in seconds)
    EPOCH_LENGTH = 1

    # Amount of overlap between two consecutive epochs (in seconds)
    OVERLAP_LENGTH = 0.9

    # Amount to 'shift' the start of each next consecutive epoch
    SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

    # Index of the channel(s) (electrodes) to be used
    # 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
    INDEX_CHANNEL = [0, 1, 2, 3]


    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer - 256*5 x 4 array to store the last 5 seconds
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 4))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                            SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        blink_votes = np.zeros(4)
        last_vote_time = time.time()
        while True:
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, 0:4]

            # Update EEG buffer with the new data
            eeg_buffer, filter_state = muselsl_utils.update_buffer(
                eeg_buffer, ch_data, notch=False,
                filter_state=filter_state)

            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = muselsl_utils.get_last_data(eeg_buffer,
                                            EPOCH_LENGTH * fs)

            #print('Alpha Relaxation: ', alpha_metric)
            x = utils.compute_features(data_epoch, filter=None, pca=None, use_original=True)
            probabilities = classifier.predict_proba([x])
            # If we're confident about something, send it
            if np.any(probabilities > 0.9): 
                print(f"\r{blink_types[np.argmax(probabilities)]} : {probabilities.max()}")
                blink_votes[np.argmax(probabilities)] += 1
            # If we're not, send "nothing"
            else:
                print("Nothing : Default")
                blink_votes[3] += 1

            current_time = time.time()
            if bluetooth and current_time - last_vote_time >= 1:
                serial_port.write(bytes(str(np.argmax(blink_votes)), encoding='utf-8'))
                print("VOTING: ", blink_types[np.argmax(blink_votes)])
                blink_votes = np.zeros(4)
                last_vote_time = current_time


    except KeyboardInterrupt:
        if bluetooth: serial_port.close()
        print('Closing!')