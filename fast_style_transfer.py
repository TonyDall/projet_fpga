import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# Chargez les images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
content_dataset = datasets.ImageFolder("path_to_content_images", transform=transform)
content_loader = torch.utils.data.DataLoader(content_dataset, batch_size=4, shuffle=True)

# Modèle de base (exemple simplifié)
class StyleTransferNet(nn.Module):
    def __init__(self):
        super(StyleTransferNet, self).__init__()
        # Exemple : plusieurs convolutions et activations
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

model = StyleTransferNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Entraînez le modèle
for epoch in range(10):
    for images, _ in content_loader:
        optimizer.zero_grad()
        output = model(images)  # Sortie stylisée
        loss = loss_fn(output, images)  # Ex. : perte simple (vous pouvez ajouter la perte de style)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")