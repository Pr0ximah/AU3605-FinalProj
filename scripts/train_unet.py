import sys

sys.path.append("./")
from models.unet import UNet
import torch
from torch.utils.data import DataLoader
from dataset.DRIVE.DRIVE import DRIVE_Dataset
from torch.optim.adam import Adam
from tqdm import trange


if __name__ == "__main__":
    # parameters
    lr = 1e-3
    epochs = 1000
    load_pretrained = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" ** Using device: {device}")

    dataset = DRIVE_Dataset()
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    model = UNet(3)
    optimizer = Adam(model.parameters(), lr=lr)
    loss = torch.nn.MSELoss()
    model = model.to(device)

    if load_pretrained:
        pretrained_model_dir = "models/logs/unet_model_500.pth"
        model.load_state_dict(torch.load(pretrained_model_dir, weights_only=False))
        init_epoch = int(pretrained_model_dir.split("_")[-1].split(".")[0])
        print(
            f" ** Loaded model from {pretrained_model_dir}, init epoch {init_epoch + 1}"
        )

    print(" ** Start training...")
    for epoch in trange(init_epoch, epochs, initial=init_epoch, total=epochs):
        for data, target in dataloader:
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_value = loss(output, target)
            loss_value.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f" ** Epoch {epoch + 1}: {loss_value.item()}")
        if (epoch + 1) % 100 == 0:
            model_path = f"models/logs/unet_model_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f" ** Model saved to {model_path}")
    print(" ** Training finished.")
    model_path = "models/logs/unet_model_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f" ** Model saved to {model_path}")
