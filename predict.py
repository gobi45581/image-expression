import torch
from torchvision import transforms, models
from PIL import Image
import argparse

def predict(model_path, image_path, topk=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 3)  # Adjust to number of classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs, indices = torch.topk(torch.softmax(outputs, dim=1), topk)
    print("Predictions:")
    for i in range(topk):
        print(f"Class {indices[0][i].item()} - Prob: {probs[0][i].item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    predict(args.model, args.image, args.topk)
