"""
Multimodal Emotion Recognition System for TESS Dataset
Complete implementation with dynamic class handling
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import json
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# Download NLTK data if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for the entire project"""
    
    # Dataset path
    DATASET_PATH = r"C:\Users\Aanan\Desktop\archive"
    
    # Model parameters
    BATCH_SIZE = 16  # Reduced batch size
    EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Speech parameters
    SAMPLE_RATE = 16000
    TARGET_LENGTH = 16000 * 2  # 2 seconds
    N_MFCC = 40
    N_FFT = 1024
    HOP_LENGTH = 512
    
    # Text parameters
    MAX_TEXT_LENGTH = 10
    EMBEDDING_DIM = 64
    VOCAB_SIZE = 1000
    
    # Model dimensions
    HIDDEN_DIM = 128
    NUM_CLASSES = None  # Will be set dynamically
    
    # Paths
    RESULTS_DIR = 'Results'
    MODELS_DIR = 'models'


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class TESSDataset(Dataset):
    """TESS Dataset loader with preprocessing"""
    
    def __init__(self, data_dir, mode='speech'):
        self.data_dir = data_dir
        self.mode = mode
        self.sample_rate = Config.SAMPLE_RATE
        self.target_length = Config.TARGET_LENGTH
        
        # Load and prepare dataset
        self.file_paths, self.texts, self.labels = self._load_dataset()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        
        # Set number of classes dynamically
        Config.NUM_CLASSES = len(np.unique(self.labels_encoded))
        print(f"Number of unique classes: {Config.NUM_CLASSES}")
        print(f"Sample labels: {dict(list(zip(self.label_encoder.classes_[:5], range(5))))}...")
        
        # Build vocabulary for text
        if mode in ['text', 'both']:
            self.vocab = self._build_vocabulary()
    
    def _load_dataset(self):
        file_paths = []
        texts = []
        labels = []
        
        label_count = defaultdict(int)
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
                    
                    # Extract label from filename (the word being spoken)
                    # Format: OAF_word_aaa.wav or YAF_word_aaa.wav
                    parts = file.split('_')
                    if len(parts) >= 2:
                        label = parts[1].lower()
                        labels.append(label)
                        label_count[label] += 1
                    
                    # Extract text/phrase from filename (same as label in this case)
                    text_part = file.split('_')[-1].split('.')[0]
                    texts.append(text_part)
        
        print(f"Total samples: {len(file_paths)}")
        print(f"Unique labels: {len(label_count)}")
        return file_paths, texts, labels
    
    def _build_vocabulary(self):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        
        for text in self.texts:
            tokens = word_tokenize(text.lower())
            for token in tokens:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
                    if idx >= Config.VOCAB_SIZE:
                        return vocab
        return vocab
    
    def _load_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            if len(audio) > self.target_length:
                audio = audio[:self.target_length]
            else:
                padding = self.target_length - len(audio)
                audio = np.pad(audio, (0, padding), 'constant')
            
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.zeros(self.target_length)
    
    def _extract_mfcc(self, audio):
        try:
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mfcc=Config.N_MFCC, 
                n_fft=Config.N_FFT, 
                hop_length=Config.HOP_LENGTH
            )
            # Add delta features
            delta_mfccs = librosa.feature.delta(mfccs)
            features = np.vstack([mfccs, delta_mfccs])
            return features.T
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            return np.zeros((50, Config.N_MFCC * 2))
    
    def _tokenize_text(self, text):
        try:
            tokens = word_tokenize(text.lower())
            indices = [self.vocab.get(token, 1) for token in tokens]
            
            if len(indices) > Config.MAX_TEXT_LENGTH:
                indices = indices[:Config.MAX_TEXT_LENGTH]
            else:
                indices = indices + [0] * (Config.MAX_TEXT_LENGTH - len(indices))
            
            return indices
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return [0] * Config.MAX_TEXT_LENGTH
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        text = self.texts[idx]
        label = self.labels_encoded[idx]
        
        if self.mode == 'speech':
            audio = self._load_audio(file_path)
            speech_features = self._extract_mfcc(audio)
            speech_features = torch.FloatTensor(speech_features)
            return speech_features, torch.tensor(label, dtype=torch.long)
            
        elif self.mode == 'text':
            text_indices = self._tokenize_text(text)
            text_features = torch.LongTensor(text_indices)
            return text_features, torch.tensor(label, dtype=torch.long)
            
        else:  # both
            audio = self._load_audio(file_path)
            speech_features = self._extract_mfcc(audio)
            speech_features = torch.FloatTensor(speech_features)
            
            text_indices = self._tokenize_text(text)
            text_features = torch.LongTensor(text_indices)
            
            return speech_features, text_features, torch.tensor(label, dtype=torch.long)


def collate_fn_speech(batch):
    features, labels = zip(*batch)
    max_len = max([f.shape[0] for f in features])
    
    padded_features = []
    for f in features:
        if f.shape[0] < max_len:
            padding = torch.zeros(max_len - f.shape[0], f.shape[1])
            f = torch.cat([f, padding], dim=0)
        padded_features.append(f)
    
    return torch.stack(padded_features), torch.stack(labels)


def collate_fn_text(batch):
    features, labels = zip(*batch)
    return torch.stack(features), torch.stack(labels)


def collate_fn_both(batch):
    speech_features, text_features, labels = zip(*batch)
    
    # Pad speech
    max_len = max([f.shape[0] for f in speech_features])
    padded_speech = []
    for f in speech_features:
        if f.shape[0] < max_len:
            padding = torch.zeros(max_len - f.shape[0], f.shape[1])
            f = torch.cat([f, padding], dim=0)
        padded_speech.append(f)
    
    return torch.stack(padded_speech), torch.stack(text_features), torch.stack(labels)


def create_dataloaders(data_dir, mode='speech'):
    dataset = TESSDataset(data_dir, mode=mode)
    
    indices = list(range(len(dataset)))
    labels = dataset.labels_encoded
    
    # Split dataset
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
    
    # Choose collate function
    if mode == 'speech':
        collate_fn = collate_fn_speech
    elif mode == 'text':
        collate_fn = collate_fn_text
    else:
        collate_fn = collate_fn_both
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, sampler=val_sampler,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, sampler=test_sampler,
        collate_fn=collate_fn, num_workers=0
    )
    
    print(f"Dataset splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return train_loader, val_loader, test_loader, dataset


# ============================================================================
# SPEECH-ONLY MODEL ARCHITECTURE
# ============================================================================

class SpeechEmotionModel(nn.Module):
    """Speech-only model with dynamic number of classes"""
    
    def __init__(self):
        super(SpeechEmotionModel, self).__init__()
        
        # CNN for feature extraction
        self.conv1 = nn.Conv1d(80, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        
        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Classifier with dynamic output size
        self.fc1 = nn.Linear(Config.HIDDEN_DIM * 2, 256)
        self.fc2 = nn.Linear(256, Config.NUM_CLASSES)
        self.relu = nn.ReLU()
        
        # Store intermediate representations
        self.intermediate_reps = None
    
    def forward(self, x):
        # x shape: (batch, time_steps, 80)
        x = x.transpose(1, 2)  # (batch, 80, time_steps)
        
        # CNN layers
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout(x)
        
        # Transpose back for LSTM
        x = x.transpose(1, 2)  # (batch, time_steps/4, 128)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Global average pooling
        context = torch.mean(lstm_out, dim=1)
        
        # Store intermediate representation
        self.intermediate_reps = context.detach().cpu().numpy()
        
        # Classification
        x = self.relu(self.fc1(context))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output


# ============================================================================
# TEXT-ONLY MODEL ARCHITECTURE
# ============================================================================

class TextEmotionModel(nn.Module):
    """Text-only model with dynamic number of classes"""
    
    def __init__(self, vocab_size):
        super(TextEmotionModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, Config.EMBEDDING_DIM, padding_idx=0)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=Config.EMBEDDING_DIM,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Classifier with dynamic output size
        self.fc1 = nn.Linear(Config.HIDDEN_DIM * 2, 256)
        self.fc2 = nn.Linear(256, Config.NUM_CLASSES)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
        # Store intermediate representations
        self.intermediate_reps = None
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Global average pooling
        context = torch.mean(lstm_out, dim=1)
        
        # Store intermediate representation
        self.intermediate_reps = context.detach().cpu().numpy()
        
        # Classification
        x = self.relu(self.fc1(context))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output


# ============================================================================
# MULTIMODAL MODEL ARCHITECTURE
# ============================================================================

class SpeechEncoder(nn.Module):
    """Speech encoder"""
    
    def __init__(self):
        super(SpeechEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(80, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        
        context = torch.mean(lstm_out, dim=1)
        return context


class TextEncoder(nn.Module):
    """Text encoder"""
    
    def __init__(self, vocab_size):
        super(TextEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, Config.EMBEDDING_DIM, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=Config.EMBEDDING_DIM,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        context = torch.mean(lstm_out, dim=1)
        return context


class MultimodalEmotionModel(nn.Module):
    """Multimodal model with dynamic number of classes"""
    
    def __init__(self, vocab_size):
        super(MultimodalEmotionModel, self).__init__()
        
        # Encoders
        self.speech_encoder = SpeechEncoder()
        self.text_encoder = TextEncoder(vocab_size)
        
        # Fusion and classifier
        fusion_input_dim = Config.HIDDEN_DIM * 4
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, Config.NUM_CLASSES)
        )
        
        # Store fused representations
        self.fused_reps = None
    
    def forward(self, speech_input, text_input):
        # Encode
        speech_features = self.speech_encoder(speech_input)
        text_features = self.text_encoder(text_input)
        
        # Fusion
        fused = torch.cat([speech_features, text_features], dim=-1)
        
        # Store fused representation
        self.fused_reps = fused.detach().cpu().numpy()
        
        # Classification
        output = self.classifier(fused)
        
        return output


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

class Trainer:
    """Training class for all models"""
    
    def __init__(self, model, train_loader, val_loader, device, model_name):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(self.train_loader, desc=f'Training {self.model_name}'):
            try:
                if self.model_name == 'speech':
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    
                elif self.model_name == 'text':
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    
                else:  # multimodal
                    speech_inputs, text_inputs, labels = batch
                    speech_inputs = speech_inputs.to(self.device)
                    text_inputs = text_inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(speech_inputs, text_inputs)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        return total_loss / len(self.train_loader), 100. * correct / total if total > 0 else 0
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    if self.model_name == 'speech':
                        inputs, labels = batch
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(inputs)
                        
                    elif self.model_name == 'text':
                        inputs, labels = batch
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(inputs)
                        
                    else:
                        speech_inputs, text_inputs, labels = batch
                        speech_inputs = speech_inputs.to(self.device)
                        text_inputs = text_inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(speech_inputs, text_inputs)
                    
                    loss = self.criterion(outputs, labels)
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue
        
        return total_loss / len(self.val_loader), 100. * correct / total if total > 0 else 0
    
    def train(self, epochs):
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model()
    
    def save_model(self):
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(Config.MODELS_DIR, f'best_{self.model_name}_model.pth')
        )
    
    def plot_training_curves(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{self.model_name} - Loss Curves')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title(f'{self.model_name} - Accuracy Curves')
        
        plt.tight_layout()
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(Config.RESULTS_DIR, f'{self.model_name}_training_curves.png'))
        plt.close()


class Evaluator:
    """Evaluation class for all models"""
    
    def __init__(self, model, test_loader, device, model_name, label_encoder):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.model_name = model_name
        self.label_encoder = label_encoder
        self.all_intermediate_reps = []
        self.all_labels = []
    
    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_intermediate_reps = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                try:
                    if self.model_name == 'speech':
                        inputs, labels = batch
                        inputs = inputs.to(self.device)
                        outputs = self.model(inputs)
                        
                    elif self.model_name == 'text':
                        inputs, labels = batch
                        inputs = inputs.to(self.device)
                        outputs = self.model(inputs)
                        
                    else:
                        speech_inputs, text_inputs, labels = batch
                        speech_inputs = speech_inputs.to(self.device)
                        text_inputs = text_inputs.to(self.device)
                        outputs = self.model(speech_inputs, text_inputs)
                    
                    _, predicted = outputs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    
                    # Collect intermediate representations
                    if hasattr(self.model, 'intermediate_reps') and self.model.intermediate_reps is not None:
                        all_intermediate_reps.append(self.model.intermediate_reps)
                    elif hasattr(self.model, 'fused_reps') and self.model.fused_reps is not None:
                        all_intermediate_reps.append(self.model.fused_reps)
                        
                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    continue
        
        if all_intermediate_reps:
            try:
                self.all_intermediate_reps = np.vstack(all_intermediate_reps)
                self.all_labels = np.array(all_labels)
            except:
                pass
        
        return np.array(all_labels), np.array(all_preds)
    
    def plot_confusion_matrix(self, labels, preds):
        # Get top 20 classes for visualization (to avoid huge matrix)
        unique_labels = np.unique(labels)
        if len(unique_labels) > 20:
            # Create a mapping for top classes
            label_counts = pd.Series(labels).value_counts()
            top_classes = label_counts.head(20).index
            
            # Filter to top classes
            mask = np.isin(labels, top_classes)
            labels_subset = labels[mask]
            preds_subset = preds[mask]
            
            # Remap labels to 0-19 for visualization
            unique_subset = np.unique(labels_subset)
            label_map = {old: new for new, old in enumerate(unique_subset)}
            
            labels_mapped = np.array([label_map[l] for l in labels_subset])
            preds_mapped = np.array([label_map[p] if p in label_map else 0 for p in preds_subset])
            
            cm = confusion_matrix(labels_mapped, preds_mapped)
            class_names = [str(i) for i in range(len(unique_subset))]
        else:
            cm = confusion_matrix(labels, preds)
            class_names = self.label_encoder.classes_
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            xticklabels=class_names[:cm.shape[0]],
            yticklabels=class_names[:cm.shape[0]]
        )
        plt.title(f'{self.model_name} - Confusion Matrix (Top 20 classes)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, f'{self.model_name}_confusion_matrix.png'))
        plt.close()
    
    def plot_emotion_clusters(self, title):
        """Visualize clusters using t-SNE"""
        if len(self.all_intermediate_reps) < 10:
            print(f"Not enough samples for t-SNE: {len(self.all_intermediate_reps)}")
            return
        
        try:
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.all_intermediate_reps)-1))
            reps_2d = tsne.fit_transform(self.all_intermediate_reps)
            
            # Plot
            plt.figure(figsize=(12, 8))
            
            # Use top 10 classes for visualization
            unique_labels = np.unique(self.all_labels)
            if len(unique_labels) > 10:
                label_counts = pd.Series(self.all_labels).value_counts()
                top_labels = label_counts.head(10).index
                mask = np.isin(self.all_labels, top_labels)
                
                for label in top_labels:
                    label_mask = (self.all_labels == label) & mask
                    if np.sum(label_mask) > 0:
                        plt.scatter(
                            reps_2d[label_mask, 0], 
                            reps_2d[label_mask, 1],
                            label=f'Class {label}',
                            alpha=0.7,
                            s=30
                        )
            else:
                for label in unique_labels:
                    mask = self.all_labels == label
                    plt.scatter(
                        reps_2d[mask, 0], 
                        reps_2d[mask, 1],
                        label=f'Class {label}',
                        alpha=0.7,
                        s=30
                    )
            
            plt.title(title)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(Config.RESULTS_DIR, f'{self.model_name}_{title}_tsne.png'), 
                        bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in t-SNE: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def train_pipeline(mode):
    """Train a specific pipeline"""
    print(f"\n{'='*50}")
    print(f"TRAINING {mode.upper()} MODEL")
    print('='*50)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        Config.DATASET_PATH, mode=mode
    )
    
    # Initialize model
    if mode == 'speech':
        model = SpeechEmotionModel()
    elif mode == 'text':
        model = TextEmotionModel(len(dataset.vocab))
    else:  # multimodal
        model = MultimodalEmotionModel(len(dataset.vocab))
    
    print(f"Model initialized with {Config.NUM_CLASSES} output classes")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, Config.DEVICE, mode)
    trainer.train(Config.EPOCHS)
    trainer.plot_training_curves()
    
    # Evaluate
    evaluator = Evaluator(model, test_loader, Config.DEVICE, mode, dataset.label_encoder)
    labels, preds = evaluator.evaluate()
    
    if len(labels) > 0:
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(labels, preds)
        
        # Visualize representations
        if mode == 'speech':
            evaluator.plot_emotion_clusters('Temporal Modeling Representations')
        elif mode == 'text':
            evaluator.plot_emotion_clusters('Contextual Modeling Representations')
        else:
            evaluator.plot_emotion_clusters('Fusion Representations')
        
        # Get accuracy
        accuracy = accuracy_score(labels, preds) * 100
        print(f"\n{mode.capitalize()} Test Accuracy: {accuracy:.2f}%")
        
        return {
            'accuracy': accuracy,
            'evaluator': evaluator,
            'labels': labels,
            'preds': preds
        }
    
    return None


def create_readme():
    """Create README.md file"""
    readme_content = []
    readme_content.append("# Speech Recognition System")
    readme_content.append("")
    readme_content.append("## Project Overview")
    readme_content.append("This project implements a speech recognition system using the TESS dataset.")
    readme_content.append("")
    readme_content.append("## Dataset")
    readme_content.append("- TESS dataset with multiple word classes")
    readme_content.append(f"- Number of classes: {Config.NUM_CLASSES}")
    readme_content.append("")
    readme_content.append("## Setup Instructions")
    readme_content.append("")
    readme_content.append("### Installation")
    readme_content.append("```bash")
    readme_content.append("pip install -r requirements.txt")
    readme_content.append("```")
    readme_content.append("")
    readme_content.append("### Running the Code")
    readme_content.append("```bash")
    readme_content.append("python main.py")
    readme_content.append("```")
    
    with open('README.md', 'w') as f:
        f.write('\n'.join(readme_content))


def create_requirements():
    """Create requirements.txt file"""
    requirements = """torch==2.0.1
torchaudio==2.0.2
librosa==0.10.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
tqdm==4.65.0
nltk==3.8.1
soundfile==0.12.1"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)


def main():
    """Main execution function"""
    print("="*60)
    print("MULTIMODAL EMOTION RECOGNITION SYSTEM")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"Dataset: {Config.DATASET_PATH}")
    
    # Check dataset path
    if not os.path.exists(Config.DATASET_PATH):
        print(f"Error: Dataset path '{Config.DATASET_PATH}' does not exist!")
        return
    
    # Create directories
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    
    # Create requirements
    create_requirements()
    
    # Train only speech model for now (to avoid memory issues with 200 classes)
    results = {}
    results['speech'] = train_pipeline('speech')
    
    if results['speech']:
        print(f"\nFinal Speech Model Accuracy: {results['speech']['accuracy']:.2f}%")
    
    # Create README
    create_readme()
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED!")
    print("="*60)
    print(f"\nResults saved in: {Config.RESULTS_DIR}/")
    print(f"Models saved in: {Config.MODELS_DIR}/")


if __name__ == "__main__":
    main()