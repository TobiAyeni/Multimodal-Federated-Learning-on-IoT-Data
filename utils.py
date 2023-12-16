import os
import requests
import numpy as np
import pandas as pd
import torch
import datetime

from scipy.io import savemat, loadmat
from scipy.stats import zscore
from models import ResNetMapper

N_DIV_OPP = 100
N_DIV_MHEALTH = 100
N_DIV_URFALL = 10
N_LABEL_DIV_OPP = 15
N_LABEL_DIV_MHEALTH = 9
N_LABEL_DIV_URFALL = 9


def fill_nan(matrix):
    """Fill NaN values with the value of the same column from previous row

    Args:
        matrix: a 2-d numpy matrix
    Return:
        A 2-d numpy matrix with NaN values filled
    """
    m = matrix
    np.nan_to_num(x=m[0, :], copy=False, nan=0.0)
    for row in range(1, m.shape[0]):
        for col in range(m.shape[1]):
            if np.isnan(m[row, col]):
                m[row, col] = m[row-1, col]
    return m


def gen_mhealth(data_path):
    """Generates subjects' data in .mat format from the mHealth dataset.

    The experiments on the mHealth dataset are done in the fashion of leave-one-subject-off.
    So the .mat data is indexed by subjects instead of "training", "validating", and "testing".

    Args:
        data_path: the path of the mHealth dataset.

    Returns:
        None
    """
    acce_columns = [0, 1, 2, 5, 6, 7, 14, 15, 16]
    gyro_columns = [8, 9, 10, 17, 18, 19]
    mage_columns = [11, 12, 13, 20, 21, 22]
    y_column = 23
    mdic = {}
    labels = set()
    shape_list = []
    for i in range(1, 11):
        s_data = np.loadtxt(os.path.join(
            data_path, "mhealth", f"mHealth_subject{i}.log"))
        x_acce = fill_nan(s_data[:, acce_columns])
        x_gyro = fill_nan(s_data[:, gyro_columns])
        x_mage = fill_nan(s_data[:, mage_columns])
        y = s_data[:, y_column]
        mdic[f"s{i}_acce"] = x_acce
        mdic[f"s{i}_gyro"] = x_gyro
        mdic[f"s{i}_mage"] = x_mage
        mdic[f"s{i}_y"] = y
        labels = labels.union(set(y))
        print(f"shape of participant {i}: {s_data.shape}")
        shape_list.append(s_data.shape[0])

    print(f"mean:{np.mean(shape_list)}, std:{np.std(shape_list)}")
    unique_y = list(labels)
    unique_y.sort()

    y_map = {}
    for idx, y in enumerate(unique_y):
        y_map[y] = idx
    for i in range(1, 11):
        mdic[f"s{i}_y"] = np.squeeze(
            np.vectorize(y_map.get)(mdic[f"s{i}_y"]))
    savemat(os.path.join(data_path, "mhealth", "mhealth.mat"), mdic)


def gen_opp(data_path):
    """Generates training, validating, and testing data from Opp datasets

    Args:
        data_path: the path of the Opportunity challenge dataset

    Returns:
        None
    """
    acce_columns = [i-1 for i in range(2, 41)]
    acce_columns.extend([46, 47, 48, 55, 56, 57, 64, 65, 66, 73, 74,
                         75, 85, 86, 87, 88, 89, 90, 101, 102, 103, 104, 105, 106, ])
    gyro_columns = [40, 41, 42, 49, 50, 51,
                    58, 59, 60, 67, 68, 69, 66, 67, 68, ]
    # Loads the run 2 from subject 1 as validating data
    data_valid = np.loadtxt(os.path.join(data_path, "opp", "S1-ADL2.dat"))
    x_valid_acce = fill_nan(data_valid[:, acce_columns])
    x_valid_gyro = fill_nan(data_valid[:, gyro_columns])
    y_valid = data_valid[:, 115]

    # Loads the runs 4 and 5 from subjects 2 and 3 as testing data
    runs_test = []
    idxs_test = []
    for r in [4, 5]:
        for s in [2, 3]:
            runs_test.append(np.loadtxt(os.path.join(
                data_path, "opp", f"S{s}-ADL{r}.dat")))
            idxs_test.append((r, s))
    data_test = np.concatenate(runs_test)
    x_test_acce = fill_nan(data_test[:, acce_columns])
    x_test_gyro = fill_nan(data_test[:, gyro_columns])
    y_test = data_test[:, 115]

    # Loads the remaining runs as training data
    runs_train = []
    for r in range(1, 6):
        for s in range(1, 5):
            if (r, s) not in idxs_test:
                runs_train.append(np.loadtxt(os.path.join(
                    data_path, "opp", f"S{s}-ADL{r}.dat")))
    data_train = np.concatenate(runs_train)
    x_train_acce = fill_nan(data_train[:, acce_columns])
    x_train_gyro = fill_nan(data_train[:, gyro_columns])
    y_train = data_train[:, 115]

    # Changes labels to (0, 1, ...)
    unique_y = list(set(y_train).union(set(y_valid)).union(set(y_test)))
    unique_y.sort()

    y_map = {}
    for idx, y in enumerate(unique_y):
        y_map[y] = idx
    y_train = np.vectorize(y_map.get)(y_train)
    y_valid = np.vectorize(y_map.get)(y_valid)
    y_test = np.vectorize(y_map.get)(y_test)

    mdic = {}
    mdic["x_train_acce"] = x_train_acce
    mdic["x_train_gyro"] = x_train_gyro
    mdic["y_train"] = np.squeeze(y_train)
    mdic["x_valid_acce"] = x_valid_acce
    mdic["x_valid_gyro"] = x_valid_gyro
    mdic["y_valid"] = np.squeeze(y_valid)  # This only has 17 classes
    mdic["x_test_acce"] = x_test_acce
    mdic["x_test_gyro"] = x_test_gyro
    mdic["y_test"] = np.squeeze(y_test)

    savemat(os.path.join(data_path, "opp", "opp.mat"), mdic)


def gen_ur_fall(data_path):
    """Generates training and testing data for UR Fall datasets.

    Args:
        data_path: the path of the UR Fall datasets.

    Returns:
        None
    """
    # headers
    # fall (0 or 1), run (1-40 for fall=0, 1-30 for fall=1), frame, HeightWidthRatio, MajorMinorRatio, BoundingBoxOccupancy, MaxStdXZ, HHmaxRatio, H, D, P40, acce_x, acce_y, acce_z, y
    a_list = []
    runs = [40, 30]
    shape_list = []
    for fall in range(2):
        prefix = "fall" if fall == 1 else "adl"
        f_labelled = os.path.join(
            data_path, "ur_fall", prefix, f"urfall-features-cam0-{prefix}s.csv")
        df_labelled = pd.read_csv(
            f_labelled, delimiter=",", header=None, usecols=list(range(11)))
        for run in range(1, runs[fall]+1):
            f_acc = os.path.join(data_path, "ur_fall", prefix,
                                 "acc", f"{prefix}-{str(run).zfill(2)}-acc.csv")
            f_sync = os.path.join(
                data_path, "ur_fall", prefix, "sync", f"{prefix}-{str(run).zfill(2)}-data.csv")
            data_acce = np.genfromtxt(f_acc, delimiter=",")
            data_sync = np.genfromtxt(f_sync, delimiter=",")
            df_label_part = df_labelled[df_labelled[0]
                                        == f"{prefix}-{str(run).zfill(2)}"]
            n_rows = df_label_part.shape[0]
            a = np.zeros([n_rows, 15])
            a[:, 0] = fall
            a[:, 1] = run
            a[:, 2] = df_label_part[1].to_numpy()
            a[:, 3:11] = df_label_part[df_label_part.columns.intersection(
                list(range(3, 11)))].to_numpy()
            a[:, 14] = df_label_part[2].to_numpy()
            mask = [x in a[:, 2] for x in data_sync[:, 0]]
            timestamps = data_sync[mask, 1]
            acce_xyz = np.empty((0, 3), dtype=np.float64)
            row_acce_data = 0
            for ts in timestamps:
                while row_acce_data < data_acce.shape[0] and data_acce[row_acce_data, 0] < ts:
                    row_acce_data += 1
                if row_acce_data >= data_acce.shape[0]:
                    break
                if abs(data_acce[row_acce_data, 0] - ts) < abs(data_acce[row_acce_data-1, 0] - ts):
                    acce_xyz = np.append(
                        acce_xyz, [data_acce[row_acce_data, 2:5]], axis=0)
                else:
                    acce_xyz = np.append(
                        acce_xyz, [data_acce[row_acce_data-1, 2:5]], axis=0)
            if acce_xyz.shape[0] < a.shape[0]:
                n = a.shape[0] - acce_xyz.shape[0]
                a = a[:-n, :]
            a[:, 11:14] = acce_xyz
            a_list.append(a)
            shape_list.append(a.shape[0])
            print(f"shape: {a.shape}")
    print(f"mean:{np.mean(shape_list)}, std:{np.std(shape_list)}")

    data = np.concatenate(a_list)
    mdic = {}
    mdic["depth"] = data[:, 0:11]
    mdic["acce"] = data[:, [0, 1, 2, 11, 12, 13]]
    mdic["y"] = data[:, [0, 1, 2, 14]]
    idxs_rgb = data[:, [0, 1, 2]]
    rgb_features = ResNetMapper.map(idxs_rgb).numpy()
    mdic["rgb"] = np.empty((data.shape[0], rgb_features.shape[1]+3))
    mdic["rgb"][:, [0, 1, 2]] = idxs_rgb
    mdic["rgb"][:, range(3, rgb_features.shape[1]+3)] = rgb_features

    y_old = data[:, 14]
    unique_y = list(set(y_old))
    unique_y.sort()

    y_map = {}
    for idx, y in enumerate(unique_y):
        y_map[y] = idx
    mdic["y"][:, 3] = np.vectorize(y_map.get)(y_old)

    savemat(os.path.join(data_path, "ur_fall", "ur_fall.mat"), mdic)


def load_data(config):
    """Loads the dataset of the FL simulation.


    Args:
        config: a map of configurations of the simulation

    Returns:
        A dictionary containing training and testing data for modality A&B and labels.
    """

    data = config["SIMULATION"]["data"]
    # data_path = config["SIMULATION"]["data_path"]
    data_path = r"C:\Users\Ayeni\iotdi22-mmfl\data"
    modality_A = config["SIMULATION"]["modality_A"]
    modality_B = config["SIMULATION"]["modality_B"]

    if data == "opp":
        modalities = ["acce", "gyro"]
        assert (
            modality_A in modalities and modality_B in modalities), "Modality is neither acce nor gyro."
        #mat_data = loadmat(os.path.join("opp.mat"))
        print(os.path.join(data_path, "opp", "opp.mat"))
        # path = 'C:\Users\Ayeni\iotdi22-mmfl\config\opp\dccae\A0_B0_AB30_label_AB_test_B'
        # mat_data = loadmat(path)
        mat_data = loadmat(os.path.join(data_path, "opp", "opp.mat"))

        data_train = {"A": zscore(mat_data[f"x_train_{modality_A}"]), "B": zscore(
            mat_data[f"x_train_{modality_B}"]), "y": np.squeeze(mat_data["y_train"])}
        data_test = {"A": zscore(mat_data[f"x_test_{modality_A}"]), "B": zscore(
            mat_data[f"x_test_{modality_B}"]), "y": np.squeeze(mat_data["y_test"])}
        return (data_train, data_test)
    elif data == "mhealth":
        modalities = ["acce", "gyro", "mage"]
        assert (
            modality_A in modalities and modality_B in modalities), "Modality is not acce, gyro, or mage."
        mat_data = loadmat(os.path.join(data_path, "mhealth", "mhealth.mat"))
        # Randomely chooses 1 subject among all 10 subjects as testing data and the rest as training data
        s_test = np.random.randint(1, 11)
        data_train = {"A": [], "B": [], "y": []}
        data_test = {}
        for i in range(1, 11):
            if i == s_test:
                data_test["A"] = zscore(mat_data[f"s{i}_{modality_A}"])
                data_test["B"] = zscore(mat_data[f"s{i}_{modality_B}"])
                data_test["y"] = np.squeeze(mat_data[f"s{i}_y"])
            else:
                data_train["A"].append(zscore(mat_data[f"s{i}_{modality_A}"]))
                data_train["B"].append(zscore(mat_data[f"s{i}_{modality_B}"]))
                data_train["y"].append(mat_data[f"s{i}_y"])
        data_train["A"] = np.concatenate(data_train["A"])
        data_train["B"] = np.concatenate(data_train["B"])
        data_train["y"] = np.squeeze(np.concatenate(data_train["y"], axis=1))
        return (data_train, data_test)
    elif data == "ur_fall":
        modalities = ["acce", "rgb", "depth"]
        assert (
            modality_A in modalities and modality_B in modalities), "Modality is not acce, rgb, or depth."
        mat_data = loadmat(os.path.join(data_path, "ur_fall", "ur_fall.mat"))
        fall_test = np.random.choice(range(1, 31), 3, replace=False)
        adl_test = np.random.choice(range(1, 41), 4, replace=False)
        data_train = {"A": [], "B": [], "y": []}
        data_test = {"A": [], "B": [], "y": []}
        a_A = mat_data[modality_A]
        a_B = mat_data[modality_B]
        a_y = mat_data["y"]

        for i in range(1, 31):
            sub_a_A = a_A[(a_A[:, 0] == 1) & (a_A[:, 1] == i), :]
            sub_a_B = a_B[(a_B[:, 0] == 1) & (a_B[:, 1] == i), :]
            sub_a_y = a_y[(a_y[:, 0] == 1) & (a_y[:, 1] == i), :]
            if modality_A == "acce" or modality_A == "depth":
                sub_a_A[:, 3:] = zscore(sub_a_A[:, 3:])
            if modality_B == "acce" or modality_B == "depth":
                sub_a_B[:, 3:] = zscore(sub_a_B[:, 3:])

            sub_a_A = sub_a_A[:, 3:]
            sub_a_B = sub_a_B[:, 3:]
            sub_a_y = sub_a_y[:, 3]

            if i in fall_test:
                data_test["A"].append(sub_a_A)
                data_test["B"].append(sub_a_B)
                data_test["y"].append(sub_a_y)
            else:
                data_train["A"].append(sub_a_A)
                data_train["B"].append(sub_a_B)
                data_train["y"].append(sub_a_y)

        for i in range(1, 41):
            sub_a_A = a_A[(a_A[:, 0] == 0) & (a_A[:, 1] == i), :]
            sub_a_B = a_B[(a_B[:, 0] == 0) & (a_B[:, 1] == i), :]
            sub_a_y = a_y[(a_y[:, 0] == 0) & (a_y[:, 1] == i), :]
            if modality_A == "acce" or modality_A == "depth":
                sub_a_A[:, 3:] = zscore(sub_a_A[:, 3:])
            if modality_B == "acce" or modality_B == "depth":
                sub_a_B[:, 3:] = zscore(sub_a_B[:, 3:])

            sub_a_A = sub_a_A[:, 3:]
            sub_a_B = sub_a_B[:, 3:]
            sub_a_y = sub_a_y[:, 3]

            if i in adl_test:
                data_test["A"].append(sub_a_A)
                data_test["B"].append(sub_a_B)
                data_test["y"].append(sub_a_y)
            else:
                data_train["A"].append(sub_a_A)
                data_train["B"].append(sub_a_B)
                data_train["y"].append(sub_a_y)

        data_train["A"] = np.concatenate(data_train["A"])
        data_train["B"] = np.concatenate(data_train["B"])
        data_train["y"] = np.squeeze(np.concatenate(data_train["y"]))
        data_test["A"] = np.concatenate(data_test["A"])
        data_test["B"] = np.concatenate(data_test["B"])
        data_test["y"] = np.squeeze(np.concatenate(data_test["y"]))
        return (data_train,  data_test)


def split_server_train(data_train, config):
    """Extracts training data for the server.

    Args:
        data_train: a dictionary of training data of modalities A&B and labels y
        config: a map of configurations of the simulation

    Returns:
    A dictionary containing the server training data.
    """
    train_supervised_ratio = float(config["FL"]["train_supervised_ratio"])
    x_train_A = data_train["A"]
    x_train_B = data_train["B"]
    y_train = data_train["y"]
    server_train_A = np.empty((0, x_train_A.shape[1]))
    server_train_B = np.empty((0, x_train_B.shape[1]))
    server_train_y = np.empty((0))

    if config["SIMULATION"]["data"] == "opp":
        n_div = N_LABEL_DIV_OPP
    elif config["SIMULATION"]["data"] == "mhealth":
        n_div = N_LABEL_DIV_MHEALTH
    elif config["SIMULATION"]["data"] == "ur_fall":
        n_div = N_LABEL_DIV_URFALL
    n_server_train = round(n_div * train_supervised_ratio)
    n_row = len(x_train_A)
    n_sample_per_div = n_row // n_div
    idxs = np.arange(0, n_row, n_sample_per_div)
    slices_A = np.split(x_train_A, idxs)
    slices_B = np.split(x_train_B, idxs)
    slices_y = np.split(y_train, idxs)
    del slices_A[0]
    del slices_B[0]
    del slices_y[0]
    n_slices = len(slices_A)
    idxs_server_train = np.random.choice(
        np.arange(n_slices), n_server_train, replace=False)
    for i in range(n_slices):
        if i in idxs_server_train:
            server_train_A = np.concatenate((server_train_A, slices_A[i]))
            server_train_B = np.concatenate((server_train_B, slices_B[i]))
            server_train_y = np.concatenate((server_train_y, slices_y[i]))
    server_train = {"A": server_train_A,
                    "B": server_train_B, "y": server_train_y}
    return server_train


def get_seg_len(n_samples, config):
    if config["SIMULATION"]["data"] == "opp":
        n_div = N_DIV_OPP
    elif config["SIMULATION"]["data"] == "mhealth":
        n_div = N_DIV_MHEALTH
    elif config["SIMULATION"]["data"] == "ur_fall":
        n_div = N_DIV_URFALL
    return int(n_samples * float(config["FL"]["train_ratio"])//n_div)


def make_seq_batch(dataset, seg_idxs, seg_len, batch_size):
    """Makes batches of sequences from the dataset.

    Args:
        dataset: a dictionary containing data of modalities A&B and labels y
        seg_idxs: A list containing the starting indices of the segments in all samples for a client.
        seg_len: An integer indicating the length of a segment
        batch_size: An integer indicating the number of batches

    Returns:
        A tuple containing the batches of sequences of modalities A&B and labels y
    """
    samples_A = dataset["A"]
    samples_B = dataset["B"]
    samples_y = dataset["y"]

    input_size_A = len(samples_A[0])
    input_size_B = len(samples_B[0])
    # the length of each sequence
    seq_len = seg_len * len(seg_idxs) // batch_size
    if seq_len > seg_len:
        seq_len = seg_len - 1

    all_indices_start = []
    for idx in seg_idxs:
        indices_start_in_seg = list(range(idx, idx + seg_len - seq_len))
        all_indices_start.extend(indices_start_in_seg)
    indices_start = np.random.choice(
        all_indices_start, batch_size, replace=False)

    A_seq = np.zeros((batch_size, seq_len, input_size_A), dtype=np.float32)
    B_seq = np.zeros((batch_size, seq_len, input_size_B), dtype=np.float32)
    y_seq = np.zeros((batch_size, seq_len), dtype=np.uint8)

    for i in range(batch_size):
        idx_start = indices_start[i]
        idx_end = idx_start+seq_len
        A_seq[i, :, :] = samples_A[idx_start: idx_end, :]
        B_seq[i, :, :] = samples_B[idx_start: idx_end, :]
        y_seq[i, :] = samples_y[idx_start:idx_end]
    return (A_seq, B_seq, y_seq)


def client_idxs(data_train, config):
    """Generates sample indices for each client.

    Args:
        data_train: a dictionary containing training data of modalities A&B and labels y
        config: a map of configurations of the simulation

    Returns:
    A list containing the sample indices for each client. Each item in the list is a list of numbers and each number representing the starting location of a segment in the training data.
    """
    num_clients_A = int(config["FL"]["num_clients_A"])
    num_clients_B = int(config["FL"]["num_clients_B"])
    num_clients_AB = int(config["FL"]["num_clients_AB"])
    num_clients = num_clients_A+num_clients_B+num_clients_AB

    n_samples = len(data_train["A"])  # number of rows in training data
    # divide the training data into divisions
    if config["SIMULATION"]["data"] == "opp":
        n_div = N_DIV_OPP
    elif config["SIMULATION"]["data"] == "mhealth":
        n_div = N_DIV_MHEALTH
    elif config["SIMULATION"]["data"] == "ur_fall":
        n_div = N_DIV_URFALL
    # each client has (n_samples * train_ratio) data
    train_ratio = float(config["FL"]["train_ratio"])

    len_div = int(n_samples // n_div)  # the length of each division
    # Within each division, we randomly pick 1 segment. So the length of each segment is
    len_seg = get_seg_len(n_samples, config)
    starts_div = np.arange(0, n_samples-len_div, len_div)
    idxs_clients = []
    for i in range(num_clients):
        idxs_clients.append(np.array([]).astype(np.int64))
        for start in starts_div:
            idxs_in_div = np.arange(start, start + len_div - len_seg)
            idxs_clients[i] = np.append(
                idxs_clients[i], np.random.choice(idxs_in_div))
    return idxs_clients


def download_UR_fall():
    """Downloads the UR Fall datasets from http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html"""
    url = "http://fenix.univ.rzeszow.pl/~mkepski/ds"
    for i in range(1, 31):
        print(f"Downloading files {i}")
        depth_camera_0 = f"{url}/data/fall-{str(i).zfill(2)}-cam0-d.zip"
        depth_camera_1 = f"{url}/data/fall-{str(i).zfill(2)}-cam1-d.zip"
        rgb_camera_0 = f"{url}/data/fall-{str(i).zfill(2)}-cam0-rgb.zip"
        rgb_camera_1 = f"{url}/data/fall-{str(i).zfill(2)}-cam1-rgb.zip"
        sync_file = f"{url}/data/fall-{str(i).zfill(2)}-data.csv"
        acc_file = f"{url}/data/fall-{str(i).zfill(2)}-acc.csv"
        r = requests.get(depth_camera_0)
        open(
            f"download/UR_FALL/fall/cam0-d/fall-{str(i).zfill(2)}-cam0-d.zip", "wb").write(r.content)
        r = requests.get(depth_camera_1)
        open(
            f"download/UR_FALL/fall/cam1-d/fall-{str(i).zfill(2)}-cam1-d.zip", "wb").write(r.content)
        r = requests.get(rgb_camera_0)
        open(
            f"download/UR_FALL/fall/cam0-rgb/fall-{str(i).zfill(2)}-cam0-rgb.zip", "wb").write(r.content)
        r = requests.get(rgb_camera_1)
        open(
            f"download/UR_FALL/fall/cam1-rgb/fall-{str(i).zfill(2)}-cam1-rgb.zip", "wb").write(r.content)
        r = requests.get(sync_file)
        open(
            f"download/UR_FALL/fall/sync/fall-{str(i).zfill(2)}-data.csv", "wb").write(r.content)
        r = requests.get(acc_file)
        open(
            f"download/UR_FALL/fall/acc/fall-{str(i).zfill(2)}-acc.csv", "wb").write(r.content)

    for i in range(1, 41):
        print(f"Downloading files {i}")
        depth_camera_0 = f"{url}/data/adl-{str(i).zfill(2)}-cam0-d.zip"
        rgb_camera_0 = f"{url}/data/adl-{str(i).zfill(2)}-cam0-rgb.zip"
        sync_file = f"{url}/data/adl-{str(i).zfill(2)}-data.csv"
        acc_file = f"{url}/data/adl-{str(i).zfill(2)}-acc.csv"
        r = requests.get(depth_camera_0)
        open(
            f"download/UR_FALL/adl/cam0-d/adl-{str(i).zfill(2)}-cam0-d.zip", "wb").write(r.content)
        r = requests.get(rgb_camera_0)
        open(
            f"download/UR_FALL/adl/cam0-rgb/adl-{str(i).zfill(2)}-cam0-rgb.zip", "wb").write(r.content)
        r = requests.get(sync_file)
        open(
            f"download/UR_FALL/adl/sync/adl-{str(i).zfill(2)}-data.csv", "wb").write(r.content)
        r = requests.get(acc_file)
        open(
            f"download/UR_FALL/adl/acc/adl-{str(i).zfill(2)}-acc.csv", "wb").write(r.content)

    print("Downloading extracted features")
    features_fall = f"{url}/data/urfall-cam0-falls.csv"
    r = requests.get(features_fall)
    open(f"download/UR_FALL/fall/urfall-features-cam0-falls.csv",
         "wb").write(r.content)

    features_adl = f"{url}/data/urfall-cam0-adls.csv"
    r = requests.get(features_adl)
    open(f"download/UR_FALL/adl/urfall-features-cam0-adls.csv", "wb").write(r.content)


if __name__ == "__main__":
    # gen_opp("data")
    # gen_mhealth("data")
    # download_UR_fall()
    # gen_ur_fall("data")
    pass
