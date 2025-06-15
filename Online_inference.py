import os
import cv2
import torch
import argparse
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import transforms
import torch.nn as nn

# --- 1. LSTM MODEL CLASS ---
class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(input_size=32 * 16 * 16, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, -1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 2. ATTENTION MODEL CLASS ---
class AttentionClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_embed = nn.Linear(32 * 16 * 16, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 60, embed_dim))
        self.attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=1
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x).view(B, T, -1)
        x = self.fc_embed(x) + self.pos_embedding[:, :T, :]
        x = self.attention(x)
        return self.classifier(x.mean(dim=1))

# --- 3. LOAD MODEL FUNCTION ---
def load_model(model_type, path, device):
    if model_type == "lstm":
        model = LSTMClassifier(hidden_size=128, num_classes=8)
    else:
        model = AttentionClassifier(embed_dim=256, num_heads=4, num_classes=8)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)

# --- 4. PREPROCESS FRAME ---
def preprocess(face_img):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    return transform(face_img)

# --- 5. CLASSIFY 20-FRAME WINDOW ---
def classify_sequence(frames, model, device):
    tensors = [preprocess(frame) for frame in frames]
    stacked = torch.stack(tensors).unsqueeze(0).to(device)  # (1, T, C, H, W)
    with torch.no_grad():
        output = model(stacked)
        pred = torch.argmax(output, dim=1).item()
    return pred

# --- 6. MAIN FUNCTION ---
def main(video_path, model_type, model_path):
    print("[INFO] Setting up device and loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_type, model_path, device)
    print(f"[INFO] Model '{model_type}' loaded successfully from: {model_path}")

    print("[INFO] Initializing face detection (MTCNN)...")
    mtcnn = MTCNN(keep_all=False, device=device)

    print(f"[INFO] Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total number of frames in video: {total_frames}")


    out_path = "annotated_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    frames_window = []
    labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

    frame_count = 0
    print("[INFO] Starting frame-by-frame processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video reached.")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[DEBUG] Processed {frame_count} frames...")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = rgb_frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_tensor = torch.tensor(face).permute(2, 0, 1).float() / 255
                face_pil = transforms.ToPILImage()(face_tensor)

                frames_window.append(face_pil)
                if len(frames_window) > 60:
                    frames_window.pop(0)

                if len(frames_window) == 60:
                    emotion_id = classify_sequence(frames_window, model, device)
                    emotion = labels[emotion_id]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(out_path, fourcc, 25.0, (w, h))
            print(f"[INFO] Output video will be saved as: {out_path}")

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("[INFO] Inference complete. Annotated video saved as:", out_path)

# --- 7. ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["lstm", "attention"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    main(args.video, args.model_type, args.model_path)
