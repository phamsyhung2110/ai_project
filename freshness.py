import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
import cv2
import base64
from net import Net

ML_MODEL = None
ML_MODEL_FILE = "model.pt"
TORCH_DEVICE = "cpu"

def get_model():
    """Loading the ML model once and returning the ML model"""
    global ML_MODEL
    if not ML_MODEL:
        ML_MODEL = Net()
        ML_MODEL.load_state_dict(
            torch.load(ML_MODEL_FILE, map_location=torch.device(TORCH_DEVICE))
        )

    return ML_MODEL

def freshness_label(freshness_percentage):
    # if freshness_percentage > 90:
    #     return "Fresh"
    # elif freshness_percentage > 65:
    #     return "Baik"
    # elif freshness_percentage > 50:
    #     return "Cukup Baik"
    # elif freshness_percentage > 0:
    #     return "Tidak Baik"
    # else:
    #     return "Busuk"
    return freshness_percentage

def price_to_text(price):
    if price == 0:
        return "Gratis"

    return str(price)

def price_by_freshness_percentage(freshness_percentage):
    return int(freshness_percentage/100*10000)

def freshness_percentage_by_cv_image(cv_image):
    mean = (0.7369, 0.6360, 0.5318)
    std = (0.3281, 0.3417, 0.3704)
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))
    image_tensor = transformation(image)
    batch = image_tensor.unsqueeze(0)
    out = get_model()(batch)
    s = nn.Softmax(dim=1)
    result = s(out)
    return int(result[0][0].item()*100)

def imdecode_image(image_file):
    return cv2.imdecode(
        np.frombuffer(image_file.read(), np.uint8),
        cv2.IMREAD_UNCHANGED
    )

def recognize_fruit_by_cv_image(cv_image):
    freshness_percentage = freshness_percentage_by_cv_image(cv_image)
    return {
        "freshness_level": freshness_percentage,
        "price": price_by_freshness_percentage(freshness_percentage)
    }


if __name__ == "__main__":
    # Gọi hàm main nếu tập lệnh được thực thi là file hiện tại
    main()
