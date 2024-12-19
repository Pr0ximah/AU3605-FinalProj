import sys
sys.path.append('./')
from models.unet import UNet
import torch
from torch.utils.data import DataLoader
from dataset.DRIVE.DRIVE import DRIVE_Dataset
from torch.optim.adam import Adam
from tqdm import trange


if __name__ == '__main__':
    # parameters
    lr = 5e-5
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" ** Using device: {device}")

    dataset = DRIVE_Dataset()
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    model = UNet(3)
    optimizer = Adam(model.parameters(), lr=lr)
    loss = torch.nn.MSELoss()
    model.to(device)

    print(" ** Start training...")
    for epoch in trange(epochs):
        for data, target in dataloader:
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_value = loss(output, target)
            loss_value.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f" ** Epoch {epoch + 1}: {loss_value.item()}")
    print(" ** Training finished.")
    torch.save(model.state_dict(), 'model.pth')