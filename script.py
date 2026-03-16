import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import cv2
from ultralytics import YOLO
import os
import tempfile
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Waste Sorting Dashboard", layout="wide", page_icon="♻️")

# Define model architectures
class EfficientNetClassifier(nn.Module):
    """EfficientNet - Excellent for waste classification"""
    def __init__(self, num_classes=6):
        super(EfficientNetClassifier, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class DenseNetClassifier(nn.Module):
    """DenseNet - Great feature reuse"""
    def __init__(self, num_classes=6):
        super(DenseNetClassifier, self).__init__()
        self.model = models.densenet121(pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class VGG16Classifier(nn.Module):
    """VGG16 - Robust and reliable"""
    def __init__(self, num_classes=6):
        super(VGG16Classifier, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class MobileNetClassifier(nn.Module):
    """MobileNet - Fast and lightweight"""
    def __init__(self, num_classes=6):
        super(MobileNetClassifier, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class YOLOv10Classifier(nn.Module):
    """YOLOv10 - Custom trained classifier"""
    def __init__(self, num_classes=6, dropout=0.08):
        super(YOLOv10Classifier, self).__init__()
        
        try:
            yolo_model = YOLO('yolov10n.pt')
            
            backbone_modules = []
            for i in range(min(10, len(yolo_model.model.model))):
                layer = yolo_model.model.model[i]
                backbone_modules.append(layer)
            
            self.backbone = nn.Sequential(*backbone_modules)
            del yolo_model
            
            total_layers = len(self.backbone)
            for i in range(max(0, total_layers - 4)):
                for param in self.backbone[i].parameters():
                    param.requires_grad = False
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )
        except:
            self.backbone = nn.Identity()
            self.classifier = nn.Linear(3, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_all_models(model_dir, device):
    """Load all available models"""
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    models_dict = {}
    
    st.sidebar.markdown("### 🤖 Loading Models...")
    
    # Custom trained models
    custom_models = {
        'best_yolov10_classifier.pt': ('YOLOv10 Classifier (Custom)', YOLOv10Classifier),
    }
    
    for model_file, (display_name, model_class) in custom_models.items():
        model_path = os.path.join(model_dir, model_file)
        
        if os.path.exists(model_path):
            try:
                model = model_class(num_classes=len(class_names))
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                models_dict[display_name] = model
                st.sidebar.success(f"✅ Loaded: {display_name}")
            except Exception as e:
                st.sidebar.warning(f"⚠️ {display_name}: {str(e)[:40]}")
    
    # Pretrained models
    pretrained_models = {
        'EfficientNet-B0 (Pretrained)': EfficientNetClassifier,
        'DenseNet-121 (Pretrained)': DenseNetClassifier,
        'VGG16 (Pretrained)': VGG16Classifier,
        'MobileNet-V2 (Pretrained)': MobileNetClassifier,
    }
    
    for display_name, model_class in pretrained_models.items():
        try:
            model = model_class(num_classes=len(class_names))
            model.to(device)
            model.eval()
            models_dict[display_name] = model
            st.sidebar.success(f"✅ Loaded: {display_name}")
        except Exception as e:
            st.sidebar.warning(f"⚠️ {display_name}: {str(e)[:40]}")
    
    # Load YOLO detection model
    try:
        if os.path.exists(os.path.join(model_dir, 'yolov10n.pt')):
            yolo_detect = YOLO(os.path.join(model_dir, 'yolov10n.pt'))
        else:
            yolo_detect = YOLO('yolov10n.pt')
    except:
        try:
            yolo_detect = YOLO('yolov8n.pt')
        except:
            yolo_detect = None
    
    return models_dict, yolo_detect, device, class_names

def get_recyclability(waste_type):
    recyclable_map = {
        'cardboard': 'Recyclable',
        'glass': 'Recyclable',
        'metal': 'Recyclable',
        'paper': 'Recyclable',
        'plastic': 'Recyclable',
        'trash': 'Non-Recyclable'
    }
    return recyclable_map.get(waste_type, 'Unknown')

def predict_with_ensemble(image, models_dict, device, class_names):
    """Enhanced ensemble prediction using all available models"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    predictions = {}
    all_probs = []
    
    with torch.no_grad():
        for model_name, model in models_dict.items():
            try:
                if isinstance(model, YOLO):
                    continue
                
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                predictions[model_name] = {
                    'class': class_names[predicted.item()],
                    'confidence': confidence.item() * 100,
                    'probs': probs.cpu().numpy()[0]
                }
                all_probs.append(probs.cpu().numpy()[0])
            except Exception as e:
                continue
    
    if not all_probs:
        return None, 0, 'Unknown', {}
    
    # Ensemble prediction (average probabilities)
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_pred = np.argmax(ensemble_probs)
    ensemble_conf = ensemble_probs[ensemble_pred] * 100
    
    waste_type = class_names[ensemble_pred]
    recyclability = get_recyclability(waste_type)
    
    return waste_type, ensemble_conf, recyclability, predictions

def detect_objects_with_classification(image, yolo_detect, models_dict, device, class_names, conf_threshold=0.10):
    """
    Detect objects and classify each detected region.
    If no objects detected, classify the entire image as fallback.
    """
    img_array = np.array(image)
    
    detections = []
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    objects_detected = False
    
    if yolo_detect:
        try:
            results = yolo_detect(img_array, conf=conf_threshold)
            
            if results:
                for result in results:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        objects_detected = True
                    
                    for box in boxes:
                        try:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            
                            cropped = image.crop((int(x1), int(y1), int(x2), int(y2)))
                            
                            waste_type, confidence, recyclability, _ = predict_with_ensemble(
                                cropped, models_dict, device, class_names
                            )
                            
                            if waste_type:
                                detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'waste_type': waste_type,
                                    'confidence': confidence,
                                    'recyclability': recyclability,
                                    'type': 'detected'
                                })
                                
                                color = 'green' if recyclability == 'Recyclable' else 'red'
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                                
                                label = f"{waste_type}: {confidence:.1f}%"
                                draw.text((int(x1), int(y1) - 25), label, fill=color, font=font)
                        except:
                            continue
        except:
            pass
    
    # FALLBACK: If no objects detected, classify the entire image!
    if not objects_detected:
        st.info("ℹ️ No specific objects detected. Classifying entire image...")
        
        waste_type, confidence, recyclability, _ = predict_with_ensemble(
            image, models_dict, device, class_names
        )
        
        if waste_type:
            detections.append({
                'bbox': (0, 0, image.width, image.height),
                'waste_type': waste_type,
                'confidence': confidence,
                'recyclability': recyclability,
                'type': 'full_image'
            })
            
            color = 'green' if recyclability == 'Recyclable' else 'red'
            draw.rectangle([0, 0, image.width, image.height], outline=color, width=3)
            
            label = f"Image: {waste_type} ({confidence:.1f}%)"
            draw.text((10, 10), label, fill=color, font=font)
    
    return annotated_image, detections

def process_video_frames(video_path, yolo_detect, models_dict, device, class_names, frame_skip=10):
    """Process video and detect waste every N frames"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detections_history = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            annotated_img, detections = detect_objects_with_classification(
                pil_image, yolo_detect, models_dict, device, class_names
            )
            
            if detections:
                for det in detections:
                    detections_history.append({
                        'frame': frame_count,
                        'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                        'waste_type': det['waste_type'],
                        'confidence': det['confidence'],
                        'recyclability': det['recyclability']
                    })
        
        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(detections_history)

def process_webcam(yolo_detect, models_dict, device, class_names, frame_skip=5, duration=10):
    """Process webcam feed for specified duration"""
    cap = cv2.VideoCapture(0)
    
    stframe = st.empty()
    stop_button = st.button("⏹️ Stop Detection")
    
    frame_count = 0
    detections_history = []
    start_time = datetime.now()
    
    while (datetime.now() - start_time).seconds < duration and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            annotated_img, detections = detect_objects_with_classification(
                pil_image, yolo_detect, models_dict, device, class_names
            )
            
            stframe.image(annotated_img, channels="RGB", use_container_width=True)
            
            if detections:
                for det in detections:
                    detections_history.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'waste_type': det['waste_type'],
                        'confidence': det['confidence'],
                        'recyclability': det['recyclability']
                    })
        
        frame_count += 1
    
    cap.release()
    return pd.DataFrame(detections_history)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = pd.DataFrame(columns=[
        'Timestamp', 'Waste Type', 'Confidence (%)', 'Material Recyclable', 'Source'
    ])

st.title("♻️ AI Powered Waste Sorting Dashboard")
st.markdown("### Multi-Model Ensemble with Enhanced Detection")
st.markdown("---")

# Model directory
MODEL_DIR = r"C:\Users\ANIL KUMAR\Downloads\AI Powered waste sorting\Garbage classification"

# Load models
st.sidebar.title("🔧 Model Loading")

try:
    with st.spinner("Loading ensemble models..."):
        models_dict, yolo_detect, device, class_names = load_all_models(
            MODEL_DIR, 
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
    
    if models_dict:
        st.sidebar.success(f"\n✅ {len(models_dict)} models ready!")
        st.sidebar.info(f"📱 Device: {device}")
        st.sidebar.markdown("**Active Models:**")
        for i, model_name in enumerate(models_dict.keys(), 1):
            st.sidebar.text(f"{i}. {model_name}")
    else:
        st.error("❌ No models loaded!")
        st.stop()
except Exception as e:
    st.sidebar.error(f"Error: {str(e)}")
    st.stop()

# Sidebar - Input Method Selection
st.sidebar.markdown("---")
st.sidebar.title("📥 Input Method")
input_method = st.sidebar.radio(
    "Choose input source:",
    ["Image Upload", "Webcam (Live)", "Video Upload"]
)

st.sidebar.markdown("---")

# Image Upload Mode
if input_method == "Image Upload":
    st.sidebar.title("📤 Upload & Classify")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    st.sidebar.markdown("---")
    st.sidebar.title("⚙️ Detection Settings")
    
    sensitivity = st.sidebar.slider(
        "🎯 Detection Sensitivity",
        min_value=0.05,
        max_value=0.50,
        value=0.10,
        step=0.05,
        help="Lower = More Sensitive (detects smaller objects)"
    )
    
    detect_multiple = st.sidebar.checkbox("🎯 Detect Multiple Objects", value=True)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.sidebar.button("🔍 Classify Image", type="primary"):
            with st.spinner("Analyzing image with ensemble models..."):
                if detect_multiple:
                    annotated_img, detections = detect_objects_with_classification(
                        image, yolo_detect, models_dict, device, class_names, conf_threshold=sensitivity
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(annotated_img, caption="Detected Objects", use_container_width=True)
                    
                    if detections:
                        st.success(f"✅ Detected {len(detections)} object(s)")
                        
                        for i, det in enumerate(detections, 1):
                            new_detection = pd.DataFrame([{
                                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'Waste Type': det['waste_type'],
                                'Confidence (%)': round(det['confidence'], 2),
                                'Material Recyclable': det['recyclability'],
                                'Source': 'Image Upload'
                            }])
                            
                            st.session_state.detection_history = pd.concat([
                                new_detection, 
                                st.session_state.detection_history
                            ], ignore_index=True)
                            
                            det_type = det.get('type', 'detected')
                            if det_type == 'full_image':
                                st.info(f"📸 **Full Image Classification** (No specific objects detected)")
                            
                            with st.expander(f"Object {i}: {det['waste_type']}"):
                                col1, col2 = st.columns(2)
                                col1.metric("Confidence", f"{det['confidence']:.2f}%")
                                col2.metric("Recyclability", det['recyclability'])
                    else:
                        st.warning("No waste objects detected in the image")
                
                else:
                    waste_type, confidence, recyclability, model_preds = predict_with_ensemble(
                        image, models_dict, device, class_names
                    )
                    
                    if waste_type is None:
                        st.error("Could not classify image. Please try another image.")
                    else:
                        new_detection = pd.DataFrame([{
                            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'Waste Type': waste_type,
                            'Confidence (%)': round(confidence, 2),
                            'Material Recyclable': recyclability,
                            'Source': 'Image Upload'
                        }])
                        
                        st.session_state.detection_history = pd.concat([
                            new_detection, 
                            st.session_state.detection_history
                        ], ignore_index=True)
                        
                        st.sidebar.success(f"**Detected:** {waste_type}")
                        st.sidebar.info(f"**Ensemble Confidence:** {confidence:.2f}%")
                        
                        if recyclability == 'Recyclable':
                            st.sidebar.success(f"♻️ **{recyclability}**")
                        else:
                            st.sidebar.warning(f"🗑️ **{recyclability}**")
                        
                        # Show individual model predictions
                        st.subheader("🔬 Individual Model Predictions")
                        if model_preds:
                            pred_df = pd.DataFrame([
                                {'Model': name, 'Prediction': pred['class'], 'Confidence': f"{pred['confidence']:.2f}%"}
                                for name, pred in model_preds.items()
                            ])
                            st.dataframe(pred_df, use_container_width=True)
                            
                            st.info(f"📊 **Ensemble Decision:** {waste_type} ({confidence:.2f}%)")
                        else:
                            st.warning("No individual model predictions available")

# Webcam Mode
elif input_method == "Webcam (Live)":
    st.sidebar.title("📷 Webcam Detection")
    
    st.sidebar.markdown("---")
    st.sidebar.title("⚙️ Detection Settings")
    
    sensitivity = st.sidebar.slider(
        "🎯 Detection Sensitivity",
        min_value=0.05,
        max_value=0.50,
        value=0.10,
        step=0.05
    )
    
    duration = st.sidebar.slider("Duration (seconds)", 5, 60, 10)
    frame_skip = st.sidebar.slider("Process every N frames", 1, 20, 5)
    
    if st.sidebar.button("🎥 Start Webcam Detection", type="primary"):
        st.info(f"🎥 Starting webcam detection for {duration} seconds...")
        
        detections_df = process_webcam(yolo_detect, models_dict, device, class_names, frame_skip, duration)
        
        if not detections_df.empty:
            detections_df['Source'] = 'Webcam'
            detections_df.rename(columns={'waste_type': 'Waste Type', 
                                         'confidence': 'Confidence (%)', 
                                         'recyclability': 'Material Recyclable'}, 
                                inplace=True)
            
            if 'timestamp' in detections_df.columns:
                detections_df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
            else:
                detections_df['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            st.session_state.detection_history = pd.concat([
                detections_df, 
                st.session_state.detection_history
            ], ignore_index=True)
            
            st.success(f"✅ Webcam detection completed! Found {len(detections_df)} detections")
        else:
            st.warning("No objects detected during webcam session")

# Video Upload Mode
elif input_method == "Video Upload":
    st.sidebar.title("🎬 Video Processing")
    
    st.sidebar.markdown("---")
    st.sidebar.title("⚙️ Detection Settings")
    
    sensitivity = st.sidebar.slider(
        "🎯 Detection Sensitivity",
        min_value=0.05,
        max_value=0.50,
        value=0.10,
        step=0.05
    )
    
    uploaded_video = st.sidebar.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
    frame_skip = st.sidebar.slider("Process every N frames", 1, 30, 10)
    
    if uploaded_video is not None:
        if st.sidebar.button("🎬 Process Video", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                tmp_path = tmp_file.name
            
            st.info("🎬 Processing video... This may take a while.")
            
            detections_df = process_video_frames(
                tmp_path, yolo_detect, models_dict, device, class_names, frame_skip
            )
            
            os.unlink(tmp_path)
            
            if not detections_df.empty:
                detections_df['Source'] = 'Video Upload'
                detections_df['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                detections_df.rename(columns={'waste_type': 'Waste Type', 
                                             'confidence': 'Confidence (%)', 
                                             'recyclability': 'Material Recyclable'}, 
                                    inplace=True)
                
                st.session_state.detection_history = pd.concat([
                    detections_df, 
                    st.session_state.detection_history
                ], ignore_index=True)
                
                st.success(f"✅ Video processing completed! Found {len(detections_df)} detections")
                
                st.subheader("📊 Video Analysis")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Detections", len(detections_df))
                col2.metric("Avg Confidence", f"{detections_df['Confidence (%)'].mean():.2f}%")
                col3.metric("Unique Types", detections_df['Waste Type'].nunique())
            else:
                st.warning("No objects detected in the video")

# Sidebar Controls
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All History"):
    st.session_state.detection_history = pd.DataFrame(columns=[
        'Timestamp', 'Waste Type', 'Confidence (%)', 'Material Recyclable', 'Source'
    ])
    st.sidebar.success("History cleared!")
    st.rerun()

if st.sidebar.button("📥 Download History CSV"):
    if not st.session_state.detection_history.empty:
        csv = st.session_state.detection_history.to_csv(index=False)
        st.sidebar.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Main Dashboard Content
if st.session_state.detection_history.empty:
    st.info("📊 No detections yet. Use the sidebar to start analyzing!")
    
    st.markdown("""
    ## 🎯 How to Use This App
    
    ### Waste Types Recognized:
    - 📄 **Paper** - Newspapers, documents, paper bags
    - 📦 **Cardboard** - Boxes, packaging, corrugated materials
    - 🍾 **Glass** - Bottles, jars, glass containers
    - 🥫 **Metal** - Cans, aluminum, steel items
    - 🛍️ **Plastic** - Bottles, bags, plastic containers
    - 🗑️ **Trash** - Mixed waste, non-recyclable items
    
    ### Steps:
    1. **Upload Image** - Use sidebar to upload a photo
    2. **Adjust Sensitivity** - Lower = detects smaller objects
    3. **Click Classify** - Models will analyze the image
    4. **View Results** - See predictions and recyclability
    
    ### Tips:
    - ✅ Good lighting helps accuracy
    - ✅ Clear, close-up images work best
    - ✅ Can detect single items or multiple objects
    - ✅ If YOLO misses it, full image classification kicks in!
    - ✅ Lower sensitivity slider for small items
    """)
else:
    df = st.session_state.detection_history
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Detections", len(df))
    
    with col2:
        avg_confidence = df['Confidence (%)'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
    
    with col3:
        recyclable_count = len(df[df['Material Recyclable'] == 'Recyclable'])
        st.metric("Recyclable Items", recyclable_count)
    
    with col4:
        non_recyclable_count = len(df[df['Material Recyclable'] == 'Non-Recyclable'])
        st.metric("Non-Recyclable", non_recyclable_count)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📈 Trends", "🎯 Confidence", "📋 Data Table", "🔍 Insights"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Count by Waste Category")
            waste_counts = df['Waste Type'].value_counts().reset_index()
            waste_counts.columns = ['Waste Type', 'Count']
            
            fig_bar = px.bar(
                waste_counts,
                x='Waste Type',
                y='Count',
                color='Waste Type',
                title="Waste Detection Count",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_bar.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader("Composition Breakdown")
            fig_pie = px.pie(
                waste_counts,
                values='Count',
                names='Waste Type',
                title="Waste Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Recyclability Status")
            recyclability_counts = df['Material Recyclable'].value_counts().reset_index()
            recyclability_counts.columns = ['Status', 'Count']
            
            fig_recycle = px.pie(
                recyclability_counts,
                values='Count',
                names='Status',
                title="Recyclable vs Non-Recyclable",
                color='Status',
                color_discrete_map={'Recyclable': '#28a745', 'Non-Recyclable': '#dc3545'}
            )
            fig_recycle.update_layout(height=400)
            st.plotly_chart(fig_recycle, use_container_width=True)
        
        with col4:
            st.subheader("Detection by Source")
            if 'Source' in df.columns:
                source_counts = df['Source'].value_counts().reset_index()
                source_counts.columns = ['Source', 'Count']
                
                fig_source = px.bar(
                    source_counts,
                    x='Source',
                    y='Count',
                    color='Source',
                    title="Detections by Input Source"
                )
                fig_source.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_source, use_container_width=True)
    
    with tab2:
        st.subheader("Detection Trends Over Time")
        
        df_trends = df.copy()
        df_trends['Timestamp'] = pd.to_datetime(df_trends['Timestamp'])
        df_trends['Date'] = df_trends['Timestamp'].dt.date
        df_trends['Hour'] = df_trends['Timestamp'].dt.hour
        
        col1, col2 = st.columns(2)
        
        with col1:
            daily_counts = df_trends.groupby('Date').size().reset_index(name='Count')
            daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])
            
            fig_daily = px.line(
                daily_counts,
                x='Date',
                y='Count',
                title="Daily Detection Count",
                markers=True
            )
            fig_daily.update_layout(height=400)
            st.plotly_chart(fig_daily, use_container_width=True)
        
        with col2:
            hourly_counts = df_trends.groupby('Hour').size().reset_index(name='Count')
            
            fig_hourly = px.bar(
                hourly_counts,
                x='Hour',
                y='Count',
                title="Hourly Detection Pattern",
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig_hourly.update_layout(height=400)
            st.plotly_chart(fig_hourly, use_container_width=True)
    
    with tab3:
        st.subheader("Model Confidence Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_conf_dist = px.histogram(
                df,
                x='Confidence (%)',
                nbins=20,
                title="Confidence Distribution",
                color_discrete_sequence=['#636EFA']
            )
            fig_conf_dist.update_layout(height=400)
            st.plotly_chart(fig_conf_dist, use_container_width=True)
        
        with col2:
            fig_conf_box = px.box(
                df,
                x='Waste Type',
                y='Confidence (%)',
                title="Confidence by Category",
                color='Waste Type'
            )
            fig_conf_box.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_conf_box, use_container_width=True)
    
    with tab4:
        st.subheader("Detection History Table")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_waste = st.multiselect(
                "Filter by Waste Type",
                options=df['Waste Type'].unique(),
                default=list(df['Waste Type'].unique())
            )
        
        with col2:
            filter_recyclable = st.multiselect(
                "Filter by Recyclability",
                options=df['Material Recyclable'].unique(),
                default=list(df['Material Recyclable'].unique())
            )
        
        with col3:
            min_confidence = st.slider(
                "Minimum Confidence (%)",
                min_value=0,
                max_value=100,
                value=0
            )
        
        filtered_df = df[
            (df['Waste Type'].isin(filter_waste)) &
            (df['Material Recyclable'].isin(filter_recyclable)) &
            (df['Confidence (%)'] >= min_confidence)
        ]
        
        st.dataframe(
            filtered_df.style.background_gradient(subset=['Confidence (%)'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")
    
    with tab5:
        st.subheader("🔍 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📌 Top Findings")
            
            most_common = df['Waste Type'].mode()[0]
            most_common_count = df['Waste Type'].value_counts().iloc[0]
            st.info(f"**Most Detected:** {most_common} ({most_common_count} times)")
            
            highest_conf_idx = df['Confidence (%)'].idxmax()
            highest_conf_waste = df.loc[highest_conf_idx, 'Waste Type']
            highest_conf_val = df.loc[highest_conf_idx, 'Confidence (%)']
            st.success(f"**Highest Confidence:** {highest_conf_waste} ({highest_conf_val:.2f}%)")
            
            recyclable_pct = (len(df[df['Material Recyclable'] == 'Recyclable']) / len(df)) * 100
            st.metric("Recyclable Percentage", f"{recyclable_pct:.1f}%")
        
        with col2:
            st.markdown("### 📊 Category Analysis")
            
            category_stats = df.groupby('Waste Type').agg({
                'Confidence (%)': ['mean', 'count']
            }).round(2)
            category_stats.columns = ['Avg Confidence', 'Count']
            category_stats = category_stats.sort_values('Count', ascending=False)
            
            st.dataframe(category_stats, use_container_width=True)
        
        st.markdown("### 💡 Recommendations")
        
        if df['Confidence (%)'].mean() < 80:
            st.warning("⚠️ Average confidence is below 80%. Try lowering the sensitivity slider.")
        else:
            st.success("✅ Excellent average confidence! Models are performing well.")
        
        if len(df[df['Material Recyclable'] == 'Recyclable']) > len(df) * 0.7:
            st.success("♻️ Great job! Most detected items are recyclable.")
        
        if len(df['Waste Type'].unique()) < 3:
            st.info("ℹ️ Limited variety detected. Try analyzing different waste types.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>♻️ AI Powered Waste Sorting Dashboard | Multi-Model Ensemble</p>
        <p>EfficientNet + DenseNet + VGG16 + MobileNet + YOLOv10 Detection</p>
        <p>Built with Streamlit | Real-time Detection Enabled | Enhanced Sensitivity</p>
    </div>
    """,
    unsafe_allow_html=True
)