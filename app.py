import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import time
import os
import gradio as gr
from PIL import Image

# --- 1. Main Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "Image"  # The folder containing 'Bleeding' and 'Non-Bleeding' subfolders
MODEL_SAVE_PATH = "bleeding_model.pth"
NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_EPOCHS = 5

print(f"Running on device: {DEVICE}")

# --- 2. Data Transformation and Loading ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 3. Model Training Function ---
def train_model():
    """Loads data, trains the model, and saves the trained weights."""
    print("--- Starting Model Training ---")
    
    # Load and split dataset
    full_dataset = datasets.ImageFolder(DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = full_dataset.classes
    print(f"Class names found: {class_names}")

    # Define model using transfer learning
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Training loop
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print(f'--- Epoch {epoch+1}/{NUM_EPOCHS} ---')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    # Save the trained model
    print(f"Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    return model, class_names

# --- 4. Gradio App Function ---
def launch_demo(model, class_names):
    """Loads the trained model and launches the Gradio interface."""
    print("\n--- Launching Interactive Demo ---")
    
    model.to(DEVICE)
    model.eval()

    # Define the prediction function for Gradio
    def predict(image):
        transform = data_transforms['val']
        image = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        return confidences

    # Create and launch the Gradio interface
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload Endoscopic Image"),
        outputs=gr.Label(num_top_classes=2, label="Prediction"),
        title="🩸 Endoscopic Bleeding Detector",
        description="An AI model to detect active bleeding events in endoscopy. Trains and launches in a single script.",
    )
    interface.launch()

# --- 5. Main Execution ---
if __name__ == '__main__':
    # Step 1: Train the model and get the class names
    trained_model, class_names_from_training = train_model()
    
    # Step 2: Launch the Gradio demo with the trained model
    launch_demo(trained_model, class_names_from_training)