import torchvision
import torchvision.transforms as transforms
import torch
from tensorops.node import Node


def tensor_to_node(tensor, optional_open_handle):
    """
    Converts an n-dimensional `torch.Tensor` object to a corresponding n-dimensional `tensorops.Node`.

    Args:
        tensor (torch.Tensor): An n-dimensional `torch.Tensor`
        optional_open_handle (Optional[_io.TextIOWrapper]): An optional binary open handle with write access for saving the `tensorops.node.Node`.
    Returns:
        node: an n-dimensional list of `tensorops.node.Node`
    """
    tensor_list = tensor.tolist()

    def recurse(data, optional_open_handle):
        if isinstance(data, list):
            return [recurse(sub, optional_open_handle) for sub in data]
        data_in_node = Node(data, requires_grad=False)
        data_in_node.save(optional_open_handle)
        return data_in_node

    return recurse(tensor_list, optional_open_handle)


def prepare_mnist_dataset(num_samples_train, num_samples_test, tensorops_format=True):
    """ """
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

    train_data, train_labels = next(
        iter(trainloader)
    )  # torch.Size([60000, 1, 28, 28]) torch.Size([60000])
    test_data, test_labels = next(
        iter(testloader)
    )  # torch.Size([10000, 1, 28, 28]) torch.Size([10000])

    train_data, train_labels = train_data.reshape(
        len(train_data), -1
    ), train_labels.reshape(len(train_labels), -1)
    test_data, test_labels = test_data.reshape(len(test_data), -1), test_labels.reshape(
        len(test_labels), -1
    )

    print(
        train_data.shape, train_labels.shape
    )  # torch.Size([60000, 784]), torch.Size([60000, 1])
    print(
        test_data.shape, test_labels.shape
    )  # torch.Size([10000, 784]), torch.Size([10000, 1])

    assert len(train_data) == len(train_labels)

    assert len(test_data) == len(test_labels)

    assert num_samples_train > 0 and num_samples_train <= len(train_data)

    assert num_samples_test > 0 and num_samples_test <= len(test_data)

    if tensorops_format:
        handle = open("./data/MNIST/processed_node/train_data_nodes.pkl", "ab")
        train_data_nodes = tensor_to_node(train_data[:num_samples_train], handle)
        handle = open("./data/MNIST/processed_node/train_labels_nodes.pkl", "ab")
        train_labels_nodes = tensor_to_node(train_labels[:num_samples_train], handle)
        handle = open("./data/MNIST/processed_node/test_data_nodes.pkl", "ab")
        test_data_nodes = tensor_to_node(test_data[:num_samples_test], handle)
        handle = open("./data/MNIST/processed_node/test_labels_nodes.pkl", "ab")
        test_labels_nodes = tensor_to_node(test_labels[:num_samples_test], handle)
        return train_data_nodes, train_labels_nodes, test_data_nodes, test_labels_nodes

    return train_data, train_labels, test_data, test_labels


if __name__ == "__main__":
    prepare_mnist_dataset(60000, 10000)
