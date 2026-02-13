 Multimodal Emotion Recognition (TESS)

#Project Overview
This project implements a multimodal emotion recognition system using the
Toronto Emotional Speech Set (TESS). The system supports:

- Speech-only emotion recognition
- Text-only modeling (synthetic text)
- Multimodal fusion (speech + text)

 Dataset
- Toronto Emotional Speech Set (TESS)
- Audio-only dataset
- Emotion information inferred from dataset structure
- Synthetic text modality constructed for multimodal experiments

 Architecture
- Speech: MFCC → CNN → BiLSTM → Classifier
- Text: Tokenization → Embedding → BiLSTM → Classifier
- Fusion: Concatenation of speech and text representation
 Setup Instructions

 Installation
```bash
pip install -r requirements.txt
