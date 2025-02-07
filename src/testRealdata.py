import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from learning.networks import FasterRCNN
from PIL import Image
import glob
import cv2 as cv

LABEL_PATH = "data/annotations/"
MODEL_PATH = "models/RCNN/MLP-best-train_loss2.9348-epoch=33-lr=0.002-wd=0.0001.pt" 
IMAGES_PATH = "data/testing/"
VIDEO_PATH = "data/videotest/"
# โหลดโมเดล
def load_model(device, model_path, num_classes=2):
    model = FasterRCNN(num_classes)
    model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model

# ทำนายผลลัพธ์
def predict(device, model, image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    return predictions

def crop_license_plate(image, predictions, threshold=0.5):
    for i in range(len(predictions[0]['boxes'])):
        score = predictions[0]['scores'][i].item()
        if score > threshold:
            box = predictions[0]['boxes'][i].tolist()
            box = [int(b) for b in box]
            license_plate = image.crop(box)
            return license_plate
        
def save_license_plate(image, predictions, threshold=0.5):
    license_plate = crop_license_plate(image, predictions, threshold)
    license_plate.save("license_plate.jpg")

def plot_predictions(image, predictions, threshold=0.5):
    fig, ax = plt.subplots(1, figsize=(10, 7))
    ax.imshow(image, cmap='gray')
    
    for i in range(len(predictions[0]['boxes'])):
        score = predictions[0]['scores'][i].item()
        if score > threshold:
            box = predictions[0]['boxes'][i].tolist()
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1] - 5, f"{score:.2f}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.show()

def img_test(device,model, image):
    predictions = predict(device,model, image)
    plot_predictions(image, predictions, threshold=0.9)

def video_test(device,model, video_path):
    cap = cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        predictions = predict(device,model, image)
        for i in range(len(predictions[0]['boxes'])):
            score = predictions[0]['scores'][i].item()
            if score > 0.5:
                box = predictions[0]['boxes'][i].tolist()
                cv.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv.putText(frame, f"{score:.2f}", (int(box[0]), int(box[1]) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.imshow("Frame", cv.resize(frame, (1280, 720)))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(device,MODEL_PATH)

    test_img = glob.glob(IMAGES_PATH + "/*.jpg") + glob.glob(IMAGES_PATH + "/*.png") + glob.glob(IMAGES_PATH + "/*.jpeg")

    # Test on images
    for image_file in test_img:
        image = Image.open(image_file)
        img_test(device, model, image)

    # Test on video
    # video_test(device,model, VIDEO_PATH + "sample.mp4")