import torchvision
import torchvision.transforms as transforms
import torch
from tensorops.node import Node

def tensor_to_node(tensor):
    tensor_list = tensor.tolist()
    
    def recurse(data):
        if isinstance(data, list):
            return [recurse(sub) for sub in data]
        return Node(data, requires_grad=False)
    
    return recurse(tensor_list)


def prepare_mnist_dataset(num_samples_train, num_samples_test, tensorops_format=True):
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
        train_data_nodes = tensor_to_node(train_data[:num_samples_train])
        train_labels_nodes = tensor_to_node(train_labels[:num_samples_train])
        test_data_nodes = tensor_to_node(test_data[:num_samples_test])
        test_labels_nodes = tensor_to_node(test_labels[:num_samples_test])
        return train_data_nodes, train_labels_nodes, test_data_nodes, test_labels_nodes

    return train_data, train_labels, test_data, test_labels


if __name__ == "__main__":
    prepare_mnist_dataset(60000,10000)
