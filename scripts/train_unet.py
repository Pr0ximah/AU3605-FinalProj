import sys

sys.path.append("./")
from models.unet import UNet
import torch
from torch.utils.data import DataLoader, random_split
from dataset.DRIVE.DRIVE import DRIVE_Dataset
from torch.optim.adam import Adam
from tqdm import trange
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net model.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs.")
    parser.add_argument(
        "--train_data_ratio", type=float, default=0.8, help="Ratio of training data."
    )
    parser.add_argument(
        "-l",
        "--load_pretrained",
        action="store_true",
        help="Setting to load pretrained model.",
    )
    parser.add_argument(
        "-m",
        "--pretrained_model_epoch",
        type=str,
        default="",
        help="Pretrained model epoch to load.",
    )
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    # parameters
    train_args = parse_args()
    lr = train_args.lr
    epochs = train_args.epochs
    train_data_ratio = train_args.train_data_ratio
    load_pretrained = train_args.load_pretrained
    pretrained_model_dir = (
        f"models/logs/unet_model_{train_args.pretrained_model_epoch}.pth"
    )
    batch_size = train_args.batch_size
    seed = train_args.seed

    assert (
        not load_pretrained or train_args.pretrained_model_epoch != ""
    ), "Pretrained model dir needed."

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" ** Using device: {device}")

    dataset = DRIVE_Dataset()
    train_size = int(len(dataset) * train_data_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    model = UNet(3)
    optimizer = Adam(model.parameters(), lr=lr)
    loss = torch.nn.BCELoss()
    model = model.to(device)
    init_epoch = 0

    if load_pretrained:
        model.load_state_dict(torch.load(pretrained_model_dir, weights_only=False))
        init_epoch = int(pretrained_model_dir.split("_")[-1].split(".")[0])
        print(
            f" ** Loaded model from {pretrained_model_dir}, init epoch {init_epoch + 1}"
        )

    print(" ** Start training...")
    for epoch in trange(init_epoch, epochs, initial=init_epoch, total=epochs):
        # train
        model.train()
        train_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_value = loss(output, target)
            train_loss += loss_value.item()
            loss_value.backward()
            optimizer.step()
        train_loss /= len(train_loader)

        # eval
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss_value = loss(output, target)
                test_loss += loss_value.item()
        test_loss /= len(test_loader)

        print(
            f"\n ** Epoch {epoch + 1}, train loss: {train_loss}, test loss: {test_loss}"
        )

        if (epoch + 1) % 4 == 0:
            model_path = f"models/logs/unet_model_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f" ** Model saved to {model_path}")

    print("\n ** Training finished.")
    model_path = "models/logs/unet_model_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f" ** Model saved to {model_path}")
