import torch.optim as optim
import sys
sys.path.append('./')
from models.p1net import DiskMaculaNet
from torch import nn
from dataset.DISK.disk import DISK_Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import trange

if __name__ == '__main__':

    # Example setup
    num_epochs = 50
    model = DiskMaculaNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" ** Using device: {device}")
    images_dir = r"dataset/DISK/shipanbiaozhu/ab/ab2"
    target_csv_path = r"dataset/DISK/shipanbiaozhu/OD.csv"
    dataset = DISK_Dataset(images_dir, target_csv_path)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    loss = torch.nn.MSELoss()
    model.to(device)
    print(" ** Start training...")
    # Training loop
    load_pretrained = True
    if load_pretrained:
        pretrained_model_dir = "/models/logs/"
        model.load_state_dict(torch.load(pretrained_model_dir, weights_only=False))
    
    for epoch in trange(num_epochs):
        for images, labels in dataloader:  # labels: [batch_size, 4]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), '/models/logs/model_OD.pth')

