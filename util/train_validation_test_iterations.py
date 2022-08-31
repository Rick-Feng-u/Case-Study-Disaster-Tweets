import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(train_loader, num_epochs, model, criterion, optimizer):
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (sequence, labels) in enumerate(train_loader):
            sequence = sequence.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(sequence)
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
        for sequence, labels in validation_loader:
            sequence = sequence.to(device)
            labels = labels.to(device)
            outputs = model(sequence)
            y_predicted = [1 if each > 0.5 else 0 for each in outputs]
            n_samples += labels.size(0)
            for i in range(len(y_predicted)):
                if y_predicted[i] == labels[i]:
                    n_correct += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')


def test_data_prediction(test_loader, model):
    submission_df = pd.DataFrame(columns=['id', 'target'])
    with torch.no_grad():
        for idx, sequence in enumerate(test_loader):
            sequence = sequence.to(device)
            predicted = model(sequence)
            id_with_target = [idx, predicted]
            submission_df.loc[len(submission_df)] = id_with_target

    return submission_df
