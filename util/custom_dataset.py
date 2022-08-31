from torch.utils.data import Dataset, DataLoader


class dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


def load_data(dataset, batch_size=64, shuffle=True):
    return DataLoader(dataset, batch_size, shuffle)
