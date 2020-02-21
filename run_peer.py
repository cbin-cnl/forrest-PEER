from pathlib import Path
import peer_helper_functions as phf
import numpy as np
import yaml
import argparse

Path.ls = lambda x: list(x.iterdir())


def main(config_file):
    config = read_config(config_file)  # Read in config file
    train_vals_x = np.genfromtxt(
        config["train_x_fixation"], delimiter=","
    )  # Read training fixation points for x
    train_vals_y = np.genfromtxt(
        config["train_y_fixation"], delimiter=","
    )  # Read training fixation points for y
    # Mask and rearrange train fmri data to (time X voxel)
    training_arr = process_nii(config["train_image"], config["eye_mask"])
    training_x_std = np.genfromtxt(config["train_x_std"])  # Read x std
    training_y_std = np.genfromtxt(config["train_y_std"])  # Read y std
    # Run PEER training (SVR from sklearn)
    X_model, Y_model = train_peer(
        training_arr, (train_vals_x, train_vals_y), training_x_std, training_y_std
    )
    # Mask and rearrange test fMRI data to (time X voxel)
    testing_arr = process_nii(config["test_image"], config["eye_mask"])
    x_hat = X_model.predict(testing_arr)  # Run x prediction using sklearn model
    xtarget = np.genfromtxt(config["test_target_x"], delimiter=",")[
        : len(testing_arr)
    ]  # read in test target in x
    ytarget = np.genfromtxt(config["test_target_y"], delimiter=",")[
        : len(testing_arr)
    ]  # read in test target in y
    print("")
    print("Correlation in X:")
    print(
        np.corrcoef(x_hat[~np.isnan(xtarget)], xtarget[~np.isnan(xtarget)])[0][1]
    )  # print correlation in x
    y_hat = Y_model.predict(testing_arr)  # Run y prediction using sklearn model
    print("Correlation in Y:")
    print(
        np.corrcoef(y_hat[~np.isnan(ytarget)], ytarget[~np.isnan(ytarget)])[0][1]
    )  # print correlation in y
    np.savetxt(
        config["output_name"],
        np.stack([x_hat, y_hat]).T,
        header="x,y",
        delimiter=",",
        comments="",
    )  # save results


def process_nii(train_image, eye_mask):
    training_arr = phf.apply_eyemask(train_image, eye_mask)
    training_arr = phf.preprocess_array(training_arr)
    return training_arr


def train_peer(x, y, xstd, ystd):
    _xmodel = phf.default_algorithm()
    _xmodel.fit(x, y[0], sample_weight=1 / (xstd) ** 2)
    _ymodel = phf.default_algorithm()
    _ymodel.fit(x, y[1], sample_weight=1 / (ystd) ** 2)
    return _xmodel, _ymodel


def read_config(config_file):
    with open(config_file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    for k, v in config.items():
        if k == "output_name":
            continue
        if not Path(v).exists():
            print("%s does not exist" % k)
            raise FileNotFoundError
    return config


if __name__ == "__main__":
    parse = argparse.ArgumentParser("config_parser")
    parse.add_argument("config", help="path to config .yml file")
    args = parse.parse_args()
    main(args.config)
