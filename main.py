import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=False)
num_ftrs = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.7),
    torch.nn.Linear(num_ftrs, 50),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(50, 4)
)
model.load_state_dict(torch.load("skin_tone.pth", map_location=device))
model = model.to(device)
model.eval()

# Define the transformation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define class names
class_names = ['Deep', 'Fair', 'Medium', 'Olive']  # Replace with actual class names

# Streamlit app
st.title("Skin Tone Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = data_transforms(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds[0]]

    st.write(f'Predicted Class: {predicted_class}')
