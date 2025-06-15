import os
import zipfile
import requests
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision
from facenet_pytorch import MTCNN
from PIL import Image

# ------------------------
# 1. Download and unzip all actor video archives
# ------------------------

base_url = "https://zenodo.org/records/1188976/files/Video_Speech_Actor_{:02d}.zip?download=1"
os.makedirs("video_data", exist_ok=True)

for actor_id in range(1, 11):
    zip_name = f"Video_Speech_Actor_{actor_id:02d}.zip"
    zip_path = os.path.join(".", zip_name)

    if not os.path.exists(zip_path):
        print(f"[Téléchargement] {zip_name} en cours...")
        url = base_url.format(actor_id)
        with requests.get(url, stream=True) as r:
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"[OK] {zip_name} téléchargé.")
    else:
        print(f"[Info] {zip_name} déjà présent, on passe.")

    print(f"[Décompression] de {zip_name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("Projet_EADIR/video_data")
    print(f"[OK] {zip_name} extrait.\n")


# ------------------------
# 2. Custom Dataset for video reading
# ------------------------
class MyVideoDataset(Dataset):
    def __init__(self, root_dir, extensions=('.mp4', '.avi', '.mov'), transform=None):
        self.root_dir = root_dir
        self.extensions = extensions
        self.transform = transform
        self.samples = []

        print("[Indexation] des vidéos...")
        for root, dirs, files in os.walk(self.root_dir):
            for filename in files:
                if filename.lower().endswith(self.extensions):
                    name_parts = os.path.splitext(filename)[0].split('-')
                    if name_parts[0] == '02':
                        continue
                    category = int(name_parts[2]) - 1
                    filepath = os.path.join(root, filename)
                    self.samples.append((filepath, category))
        print(f"[OK] {len(self.samples)} vidéos trouvées.\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video_frames, audio_frames, info = torchvision.io.read_video(video_path, pts_unit='sec')
        png_frames = [T.ToPILImage()(video_frames[i].permute(2, 0, 1)) for i in range(video_frames.shape[0])]

        actor_folder = os.path.basename(os.path.dirname(video_path))    # e.g. "Actor_01"
        video_name = os.path.splitext(os.path.basename(video_path))[0]  # e.g. "01-01-06-02-01-02-01"
        subfolder = os.path.join(actor_folder, video_name)
        return png_frames, audio_frames, label, subfolder

# ------------------------
# 3. Face extraction function
# ------------------------
def process_video_frames(model, video_frames, root, actor_dir, video_id):
    save_path = os.path.join(root, actor_dir, video_id)
    os.makedirs(save_path, exist_ok=True)

    to_pil = T.ToPILImage()

    for idx, frame in enumerate(video_frames):
        with torch.no_grad():
            face = model([frame])[0]

        if face is not None:
            # Rescale [-1, 1] → [0, 1] before converting to image
            face = (face + 1) / 2
            face = torch.clamp(face, 0, 1)
            face_img = to_pil(face)
            face_img.save(os.path.join(save_path, f"{idx}.png"))


# ------------------------
# 4. Process all videos
# ------------------------
print("\[Initialization] of the dataset and the MTCNN model...")
dataset = MyVideoDataset('Projet_EADIR/video_data')
mtcnn = MTCNN()
print("\[Start] of video processing...\n")

for video_data in tqdm(dataset, desc="Processing videos"):
    video_frames, _, label, subfolder = video_data
    actor_dir = subfolder.split('/')[0]
    video_id = subfolder.split('/')[1]
    print(f"\n[Video] Processing of {actor_dir}/{video_id}...")
    process_video_frames(mtcnn, video_frames, 'Projet_EADIR/img_data', actor_dir, video_id)

print("\nAll faces have been successfully extracted!")