import torch
import torch.nn as nn
import torchinfo

class LeNet5(nn.Module):
  def __init__(self, in_channel, num_classes):
    super(LeNet5, self).__init__()
    # === 1. BAGIAN EKSTRAKTOR FITUR ===
    self.feature_extractor = nn.Sequential(
        # Lapisan C1: Konvolusi pertama
        nn.Conv2d(in_channels=in_channel, out_channels=6, kernel_size=5, stride=1),
        nn.Tanh(),
        # Lapisan S2: Subsampling (Average Pooling)
        nn.AvgPool2d(kernel_size=2, stride=2),
        # Lapisan C3: Konvolusi kedua
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
        nn.Tanh(),
        # Lapisan S4: Subsampling (Average Pooling)
        nn.AvgPool2d(kernel_size=2, stride=2)
        )

    # === 2. BAGIAN KLASIFIKATOR ===
    self.classifier = nn.Sequential(
        # Lapisan C5 (sebagai Fully Connected pertama)
        nn.Linear(in_features=16 * 5 * 5, out_features=120),
        nn.Tanh(),
        # Lapisan F6 (Fully Connected kedua)
        nn.Linear(in_features=120, out_features=84),
        nn.Tanh(),
        # Lapisan Output
        nn.Linear(in_features=84, out_features=num_classes),
        nn.Softmax(dim=1)
        )

  # Mendefinisikan alur maju (forward pass) dari data melalui model.
  def forward(self, x):
    # Lewatkan input melalui ekstraktor fitur
    x = self.feature_extractor(x)
    # Flatten output dari 2D menjadi 1D sebelum masuk ke classifier
    # Dimensi 0 (batch) dipertahankan
    x = torch.flatten(x, 1)
    # Lewatkan data yang sudah diratakan melalui classifier
    logits = self.classifier(x)
    return logits

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model instance and move it to device
    model = LeNet5(in_channel=3, num_classes=5)
    model = model.to(device)
    
    # Print model summary
    print("\nLeNet-5 Model Architecture:")
    print("=" * 50)
    print(f"Input Shape: (batch_size, 3, 32, 32)")
    print("=" * 50)
    
    # Use torchinfo to show detailed layer information
    torchinfo.summary(model, input_size=(1, 3, 32, 32), verbose=1,
                     device=device,
                     col_names=["input_size", "output_size", "num_params", "trainable"])
    
    try:
        # Test with sample input - make sure it's on the same device as model
        sample_input = torch.randn(1, 3, 32, 32).to(device)
        output = model(sample_input)
        print(f"\nInput shape: {sample_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Sample output: {output}")
    except RuntimeError as e:
        print(f"\nError occurred: {e}")
