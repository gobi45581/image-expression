import cv2
import os
from pathlib import Path
from facenet_pytorch import MTCNN

def crop_faces(input_dir, output_dir, use_cuda=False):
    mtcnn = MTCNN(keep_all=False, device='cuda' if use_cuda else 'cpu')
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.glob('*.*'):
        img = cv2.imread(str("C:\Users\selva\Downloads\istockphoto-626205158-612x612.jpg"))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(b) for b in box]
                face = img[y1:y2, x1:x2]
                face_resized = cv2.resize(face, (224, 224))
                save_path = output_dir / f"{img_path.stem}_face{i}.jpg"
                cv2.imwrite(str(save_path), face_resized)
    print(f"Faces cropped and saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input folder")
    parser.add_argument("--output", type=str, required=True, help="Output folder")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()
    crop_faces(args.input, args.output, args.use_cuda)
