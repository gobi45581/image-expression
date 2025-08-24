from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import torch
from torchvision import transforms, models
from PIL import Image
import io

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # Change according to your classes
model.load_state_dict(torch.load('models/best.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
    return {"prediction": int(predicted)}

@app.get("/")
async def main():
    return HTMLResponse(open("web/index.html").read())
