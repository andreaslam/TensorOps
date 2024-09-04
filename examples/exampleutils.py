import os
import torch
import torchvision
import torchvision.transforms as transforms
from tensorops.node import Node

def tensor_to_node(tensor, optional_open_handle=None):
    """
    Convert an n-dimensional torch.Tensor to a corresponding n-dimensional tensorops.Node.

    Args:
        tensor (torch.Tensor): An n-dimensional torch.Tensor.
        optional_open_handle (Optional[_io.TextIOWrapper], optional): An optional binary open handle with write access for saving the tensorops.node.Node. Defaults to None.

    Returns:
        tensorops.Node: An n-dimensional list of tensorops.node.Node.
    """

    def _recurse(data, handle):
        if isinstance(data, list):
            return [_recurse(sub, handle) for sub in data]
        node = Node(data, requires_grad=False)
        if handle:
            node.save(handle)
        return node

    return _recurse(tensor.tolist(), optional_open_handle)


def save_nodes_from_pytorch(data, file_path, save):
    """
    Helper function to save nodes to a file if save is True.

    Args:
        data (pytorch.Tensor): PyTorch data to be processed
        file_path (Optional[string]): Path to save processed data, if `save=True`.
        save (bool): Whether to save the processed data.
    Returns:
        nodes (list[tensorops.nodeNode]): Processed `tensorops.nodeNode` corresponding to original PyTorch data.
    """
    handle = open(file_path, "ab+") if save else None
    nodes = tensor_to_node(data, handle)
    if handle:
        handle.close()
    return nodes


def prepare_mnist_dataset(
    num_samples_train, num_samples_test, tensorops_format=True, save=True
):
    """
    Download and process the MNIST dataset.

    Args:
        num_samples_train (Union[int, Callable]): The number of training samples to process. Alternatively, pass in `len()` to sample the entire dataset.
        num_samples_test (Union[int, Callable]): The number of test samples to process. Alternatively, pass in `len()` to sample the entire dataset.
        tensorops_format (bool, optional): If True, convert the dataset into tensorops.node.Node(). Defaults to True.
        save (bool, optional): If True, save the dataset to individual files to "data/MNIST/processed_node" or "data/MNIST/processed_pytorch" depending on tensorops_format. Defaults to True.

    Returns:
        tuple: Processed data and labels in either node format or tensor format.
    """
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False
    )

    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False
    )

    train_data, train_labels = next(iter(trainloader))
    test_data, test_labels = next(iter(testloader))

    num_samples_train, num_samples_test = process_data_count(
        num_samples_train, num_samples_test, train_data, test_data
    )

    verify_dataset(
        num_samples_train,
        num_samples_test,
        train_data,
        train_labels,
        test_data,
        test_labels,
    )

    print("Downloaded MNIST dataset.")

    print(
        f"Preparing to save {num_samples_train} training samples and {num_samples_test} test samples"
    )

    train_data = train_data.reshape(len(train_data), -1)[:num_samples_train]
    train_labels = train_labels.reshape(len(train_labels), -1)[:num_samples_train]
    test_data = test_data.reshape(len(test_data), -1)[:num_samples_test]
    test_labels = test_labels.reshape(len(test_labels), -1)[:num_samples_test]

    if tensorops_format:
        print("Preparing dataset in TensorOps format")
        os.makedirs("data/MNIST/processed_node/", exist_ok=True)
        train_data_nodes = save_nodes_from_pytorch(
            train_data, "data/MNIST/processed_node/train_data_nodes.pkl", save
        )
        train_labels_nodes = save_nodes_from_pytorch(
            train_labels, "data/MNIST/processed_node/train_labels_nodes.pkl", save
        )
        test_data_nodes = save_nodes_from_pytorch(
            test_data, "data/MNIST/processed_node/test_data_nodes.pkl", save
        )
        test_labels_nodes = save_nodes_from_pytorch(
            test_labels, "data/MNIST/processed_node/test_labels_nodes.pkl", save
        )
        if save:
            print("Saved tensorops data")
        return train_data_nodes, train_labels_nodes, test_data_nodes, test_labels_nodes

    print("Preparing dataset in PyTorch format")

    if save:
        os.makedirs("data/MNIST/processed_pytorch/", exist_ok=True)
        torch.save(train_data, "data/MNIST/processed_pytorch/train_data.pt")
        torch.save(train_labels, "data/MNIST/processed_pytorch/train_labels.pt")
        torch.save(test_data, "data/MNIST/processed_pytorch/test_data.pt")
        torch.save(test_labels, "data/MNIST/processed_pytorch/test_labels.pt")
        print("Saved PyTorch data")

    return train_data, train_labels, test_data, test_labels


def process_data_count(num_samples_train, num_samples_test, train_data, test_data):
    if num_samples_train is len:
        num_samples_train = len(train_data)
    elif isinstance(num_samples_train, int):
        pass
    else:
        raise ValueError(
            f"num_samples_train must be of type int or len(), got {type(num_samples_train)}"
        )
    if num_samples_test is len:
        num_samples_test = len(test_data)
    elif isinstance(num_samples_test, int):
        pass
    else:
        raise ValueError(
            f"num_samples_test must be of type int or len(), got {type(num_samples_test)}"
        )
    return num_samples_train, num_samples_test


def verify_dataset(
    num_samples_train,
    num_samples_test,
    train_data,
    train_labels,
    test_data,
    test_labels,
):
    assert len(train_data) == len(train_labels)

    assert len(test_data) == len(test_labels)

    assert num_samples_train > 0 and num_samples_train <= len(train_data)

    assert num_samples_test > 0 and num_samples_test <= len(test_data)

if __name__ == "__main__":
    prepare_mnist_dataset(60000, 10000)