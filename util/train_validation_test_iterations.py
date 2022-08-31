import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(train_loader, num_epochs, model, criterion, optimizer):
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')


def evaluation(validation_loader, model):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            y_predicted = [1 if each > 0.5 else 0 for each in outputs]
            n_samples += labels.size(0)
            for i in range(len(y_predicted)):
                if y_predicted[i] == labels[i]:
                    n_correct += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

#def test_data_prediction():