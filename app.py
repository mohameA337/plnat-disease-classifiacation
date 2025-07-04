import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

with open(r'M:\GitHub\plnat-disease-classifiacation\notebooks\classes.json', 'r') as f:
    classes = json.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(r"M:\GitHub\plnat-disease-classifiacation\notebooks\resnet18_plant_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title(" Plant Disease Detection")
st.write("Upload an image of a plant leaf to predict its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
        predicted_class = classes[predicted.item()]

    st.write(f"#Predicted Disease: **{predicted_class}**")
