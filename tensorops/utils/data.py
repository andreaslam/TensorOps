import requests
import shutil
import os
from tensorops.node import Node
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch


def prepare_dataset(lib, limit=None):
    # url = r"https://storage.googleapis.com/kaggle-data-sets/5784553/9504237/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241012%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241012T200623Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=0ba2a9204d9f858a3d0833f9579b7345063b700d0756f122fca4b17d5b6388d3583b321b4547a218ff9d40f62e0262efffaf9a81379ec5346f7d763db613c5c6baca5f181d84d469640d15d5c1f5efe684263a2acabc82269c07f6589a1cf9409777d0754042354aa805d9d5aced2dc7f3eace976ae4251ea6c244d9da96ea6697ee3c4aee5c6c222852fc1d2b60bcf0e13b033b28ea584e536ced0c49bf8dbecb99c1dd9ee366cb910c91bc670bdf4a016eaa36860d1553395e76591bc7b3f473af69da13ea67cd045fd7c09b5c6f0a3795b71c7c6a611f5eefd4299f548eeba881cb24e8055d9338c7eecc29073edb7bb03b626e55c58bf386a89b3a67e762"
    # r = requests.get(url, allow_redirects=True)

    path = "data"

    # if not os.path.exists(path):
    #     os.makedirs(path)

    dataset_name = f"{path}/dummy_dataset.zip"
    # open(dataset_name, "wb").write(r.content)

    shutil.unpack_archive(dataset_name, path)

    datafile_name = f"{path}/user_behavior_dataset.csv"

    f = pd.read_csv(datafile_name)
    keep_col = [
        "Operating System",
        "App Usage Time (min/day)",
        "Screen On Time (hours/day)",
        "Battery Drain (mAh/day)",
        "Number of Apps Installed",
        "Data Usage (MB/day)",
        "Age",
        "Gender",
    ]
    new_f = f[keep_col]
    new_f = new_f.copy()
    le_os = LabelEncoder()
    new_f["Operating System"] = le_os.fit_transform(new_f["Operating System"])

    le_gender = LabelEncoder()
    new_f["Gender"] = le_gender.fit_transform(new_f["Gender"])

    new_f.to_csv(datafile_name, index=False)

    inputs = new_f[["Age", "Operating System", "Gender"]]

    outputs = new_f.drop(columns=["Age", "Operating System", "Gender"])

    if not limit:
        limit = len(inputs)

    if lib == "tensorops":
        inputs_tensorops = [
            [Node(data) for data in entry] for entry in inputs.values.tolist()
        ]
        outputs_tensorops = [
            [Node(data) for data in entry] for entry in outputs.values.tolist()
        ]
        return inputs_tensorops[:limit], outputs_tensorops[:limit]

    elif lib == "pytorch":
        inputs_torch = torch.tensor(inputs.values, dtype=torch.float64)
        outputs_torch = torch.tensor(outputs.values, dtype=torch.float64)
        return inputs_torch[:limit], outputs_torch[:limit]

    else:
        raise ValueError(
            f"Invalid library option! options are 'tensorops' and 'pytorch', got '{lib}' instead"
        )
