import sys
sys.path.append('./')
import torch.optim as optim
from models.DISK_net import DiskMaculaNet
from torch import nn
from dataset.DISK.disk_OD import DISK_Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import trange

if __name__ == '__main__':

    # Example setup
    num_epochs = 300
    model = DiskMaculaNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" ** Using device: {device}")
    images_dir = r"dataset/DISK/shipanbiaozhu/ab/ab2"
    target_csv_path = r"dataset/DISK/shipanbiaozhu/OD.csv"
    dataset = DISK_Dataset(images_dir, target_csv_path)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model.to(device)
    print(" ** Start training...")
    # Training loop
    load_pretrained = True
    if load_pretrained:
        pretrained_model_dir = "models/logs/model_OD_seed_1.pth"
        model.load_state_dict(torch.load(pretrained_model_dir, weights_only=False))
    
    for epoch in trange(num_epochs):
        # Training loop
        train_loss = 0.0
        model.train()
        for images, labels in train_loader:  # labels: [batch_size, 4]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Validation loop
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

        # Save model
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'models/logs/model_OD_{epoch}.pth')

    torch.save(model.state_dict(), 'models/logs/model_OD_new.pth')

