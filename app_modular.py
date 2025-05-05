import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import io

# Classifier Model (ResNet18)
def get_classifier_model(num_classes=2):
    model = resnet18(weights=None) 
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# De-raining/De-hazing Model (SimpleRestorationNet)
class SimpleRestorationNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SimpleRestorationNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh() 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# Function to load models
@st.cache_resource 
def load_models(classifier_path, derain_path, dehaze_path, device):
    # Load Classifier
    classifier = get_classifier_model()
    try:
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        classifier.to(device)
        # st.success(f"Classifier loaded from {classifier_path}")
    except FileNotFoundError:
        st.error(f"Classifier weights not found at {classifier_path}. Please check the path.")
        classifier = None
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        classifier = None

    # Load De-raining Model
    de_raining_model = SimpleRestorationNet()
    try:
        de_raining_model.load_state_dict(torch.load(derain_path, map_location=device))
        de_raining_model.eval()
        de_raining_model.to(device)
        # st.success(f"De-raining model loaded from {derain_path}")
    except FileNotFoundError:
        st.error(f"De-raining weights not found at {derain_path}. Please check the path.")
        de_raining_model = None
    except Exception as e:
        st.error(f"Error loading de-raining model: {e}")
        de_raining_model = None

    # Load De-hazing Model
    de_hazing_model = SimpleRestorationNet()
    try:
        de_hazing_model.load_state_dict(torch.load(dehaze_path, map_location=device))
        de_hazing_model.eval()
        de_hazing_model.to(device)
        # st.success(f"De-hazing model loaded from {dehaze_path}")
    except FileNotFoundError:
        st.error(f"De-hazing weights not found at {dehaze_path}. Please check the path.")
        de_hazing_model = None
    except Exception as e:
        st.error(f"Error loading de-hazing model: {e}")
        de_hazing_model = None
    return classifier, de_raining_model, de_hazing_model

# Preprocessing for Classifier 
classifier_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

# Preprocessing for SimpleRestoration models 
simple_restore_preprocess = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),
])

# Postprocessing to display image
postprocess = transforms.ToPILImage()
def denormalize_tensor(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

# Streamlit App
st.set_page_config(layout="wide")
st.title("CLEAR")

st.sidebar.header("About")
st.sidebar.info("Modular Climate-affected Loss Enhancement and Adaptive Restoration (Modular CLEAR) uses a SimpleRestorationNet architecture for removing rain or haze from input pictures.")

# Load Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")
classifier, de_raining_model, de_hazing_model = load_models(
    "classifier.pth" , "de-raining.pth", "de-hazing.pth", device
)
models_loaded = classifier and de_raining_model and de_hazing_model

# Image Upload 
uploaded_file = st.file_uploader("Choose an image file (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and models_loaded:
    # Read image
    image_bytes = uploaded_file.getvalue()
    input_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)

    # Preprocessing
    input_tensor_clf = classifier_preprocess(input_image).unsqueeze(0).to(device)
    input_tensor_sr = simple_restore_preprocess(input_image).unsqueeze(0).to(device) 

    # Inference
    with torch.no_grad():
        # 1. Classify
        prediction_logits = classifier(input_tensor_clf)
        prediction = prediction_logits.argmax(dim=1).item() # 0 = rain, 1 = haze

        # 2. Apply appropriate model
        if prediction == 0:
            output_tensor = de_raining_model(input_tensor_sr) 
        else:
            output_tensor = de_hazing_model(input_tensor_sr) 

        # Postprocessing
        # Squeeze batch dimension, detach from graph, move to CPU
        output_tensor = output_tensor.squeeze(0).detach().cpu()
        output_tensor_denorm = denormalize_tensor(output_tensor)
        output_image = postprocess(output_tensor_denorm)

    with col2:
        st.subheader("Cleaned Image")
        st.image(output_image, use_container_width=True)