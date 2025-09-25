import torch
import torch.nn as nn
import timm

class ViTSmall(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ViTSmall, self).__init__()
        # Load pretrained ViT-Small model
        self.model = timm.create_model('vit_small_patch16_224', 
                                     pretrained=pretrained,
                                     num_classes=num_classes)
        
        # Mengubah head classifier jika diperlukan
        if num_classes != 1000:  # 1000 adalah default ImageNet classes
            # Dapatkan dimensi feature dari model
            n_features = self.model.head.in_features
            # Ganti classifier head
            self.model.head = nn.Sequential(
                nn.Linear(n_features, num_classes),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        # Forward pass melalui model ViT
        return self.model(x)

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test model dengan 5 kelas (gado_gado, rendang, bakso, soto_ayam, nasi_goreng)
    model = ViTSmall(num_classes=5).to(device)
    # Print model summary
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224).to(device)  # ViT expects 224x224 images
    print(model(x).shape)  # Should print torch.Size([1, 5])