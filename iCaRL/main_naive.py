import torch
from torchvision.datasets.cifar import CIFAR100
from cl_dataset_tools import NCProtocol, NCProtocolIterator
from torchvision.models.resnet import resnet18
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from cl_metrics_tools import get_accuracy

tasks = 10
n_epochs = 1
top_k_accuracies = [1, 5]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def main():
    protocol = NCProtocol(CIFAR100('./data/cifar100', train=True, download=True, transform=transform),
                          CIFAR100('./data/cifar100', train=False, download=True, transform=transform_test),
                          n_tasks=tasks, shuffle=True, seed=1234)

    model = resnet18(pretrained=False, num_classes=100).to(device)

    train_dataset: Dataset
    task_info: NCProtocolIterator
    for task_idx, (train_dataset, task_info) in enumerate(protocol):
        print('Classes in this batch:', task_info.classes_in_this_task)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        # Sets model in train mode
        model.train()
        for epoch in range(n_epochs):
            for patterns, labels in train_loader:
                # Clear grad
                model.zero_grad()

                # Send data to device
                patterns = patterns.to(device)
                labels = labels.to(device)

                # Forward
                output = model(patterns)

                # Loss
                loss = criterion(output, labels)

                # Backward
                loss.backward()

                # Update step
                optimizer.step()
        print('Task', task_idx, 'ended')

        top_train_accuracies, _, _ = get_accuracy(model,
                                                  task_info.swap_transformations().get_current_training_set(),
                                                  device=device, required_top_k=top_k_accuracies, batch_size=128)

        for top_k_idx, top_k_acc in enumerate(top_k_accuracies):
            print('Top', top_k_acc, 'train accuracy', top_train_accuracies[top_k_idx].item())

        top_test_accuracies, _, _ = get_accuracy(model, task_info.get_cumulative_test_set(), device=device,
                                                 required_top_k=top_k_accuracies, batch_size=128)

        for top_k_idx, top_k_acc in enumerate(top_k_accuracies):
            print('Top', top_k_acc, 'test accuracy', top_test_accuracies[top_k_idx].item())


if __name__ == '__main__':
    main()
