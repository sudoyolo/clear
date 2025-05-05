import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io

# Model Definition
class TripletAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(TripletAttentionBlock, self).__init__()
        # Local Pixel-wise Attention (LPA)
        self.lpa = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # Global Strip-wise Attention (GSA)
        self.gsa_h = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1))
        self.gsa_v = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0))
        # Global Distribution Attention (GDA)
        self.gda_conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.gda_in = nn.InstanceNorm2d(in_channels // 2) 
        self.gda_conv2 = nn.Conv2d(in_channels // 2 * 2, in_channels, kernel_size=1)

    def forward(self, x):
        # LPA
        lpa_map = self.lpa(x)
        # GSA
        gsa_h = self.gsa_h(x)
        gsa_v = self.gsa_v(x)
        gsa_map = torch.sigmoid(gsa_h + gsa_v)
        # Combine LPA and GSA
        spatial_attended = x * lpa_map * gsa_map
        # GDA
        gda1 = self.gda_in(self.gda_conv1(x))
        gda2 = self.gda_conv1(x)
        gda_cat = torch.cat([gda1, gda2], dim=1)
        gda_out = self.gda_conv2(gda_cat)
        return spatial_attended + gda_out + x

class TANet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(TANet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            TripletAttentionBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            TripletAttentionBlock(128),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            TripletAttentionBlock(64),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh() 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Streamlit App
st.set_page_config(layout="wide")
st.title("CLEAR")

st.sidebar.header("About")
st.sidebar.info(
    "Triple Attention Net Climate-affected Loss Enhancement and Adaptive Restoration (TANet CLEAR) uses a TANet architecture for removing rain or haze from input pictures. "
    "TANet Paper: https://arxiv.org/abs/2410.08177."
)

# Load Model
@st.cache_resource 
def load_model(model_path="TANet.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TANet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        # st.success(f"Model loaded successfully from {model_path} onto {device}.")
        return model, device
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}. Please place TANet.pth in the same directory or provide the correct path.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Image Processing
def preprocess_image(image, img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def postprocess_image(tensor):
    """Converts the output tensor back to a PIL image."""
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    image = transforms.ToPILImage()(tensor)
    return image

# Main App Logic
model, device = load_model() 
uploaded_file = st.file_uploader("Choose an image file (JPG, PNG)", type=["jpg", "jpeg", "png"])

if model and uploaded_file is not None:
    # Read the uploaded image
    bytes_data = uploaded_file.getvalue()
    input_image = Image.open(io.BytesIO(bytes_data)).convert('RGB')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)

    with torch.no_grad(): 
        input_tensor = preprocess_image(input_image).to(device)
        output_tensor = model(input_tensor)
        cleaned_image = postprocess_image(output_tensor)

    with col2:
        st.subheader("Cleaned Image")
        st.image(cleaned_image, use_container_width=True)
