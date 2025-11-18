# DeepNet Intrusion Detection — Setup & Usage Guide

This README provides the complete setup, installation, and execution workflow for running your **Binary** and **Multiclass** IDS models (DNN + BiLSTM + Multi‑Head Attention).

---

## 🚀 1. Project Structure

```
project/
│── binary.ipynb
│── multiclass.ipynb
│── models/
│    ├── phase1_dnn.pth
│    ├── phase2_deepnet.pth
│── datasets/
│    ├── NSL-KDD/
│    ├── CIC-IDS/
│── README.md
```

---

## 🛠️ 2. Environment Setup

### **Step 1 — Create virtual environment**
```bash
python -m venv ids-env
source ids-env/bin/activate    # Linux/Mac
ids-env\Scripts\activate     # Windows
```

### **Step 2 — Install dependencies**
```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib
pip install seaborn tqdm
```

(If using CUDA GPU)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 📥 3. Download Dataset

### **NSL-KDD**
Download and place inside:
```
datasets/NSL-KDD/
```

### **CIC-IDS / BoT-IoT**
Place them under:
```
datasets/
```

---

## 🧠 4. Running Binary Classification Model

Open:
```
binary.ipynb
```

### **Train the model**
Run all cells until:
```python
model.train()
```

### **Evaluate**
```python
evaluate(model, test_loader)
```

Your output will include:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- False Positive Rate  
- Confusion Matrix  

---

## 🧠 5. Running Multi-Class Classification Model

Open:
```
multiclass.ipynb
```

### **Train**
```python
model.fit(train_loader)
```

### **Evaluate**
```python
test_results = evaluate(model, test_loader)
print(test_results)
```

---

## 🔥 6. Using Dual‑Phase DeepNet (Phase I + Phase II)

### **Phase I — DNN Classification**
```python
phase1 = DNN()
phase1.load_state_dict(torch.load("models/phase1_dnn.pth"))
phase1.eval()
```

### **Phase II — BiLSTM + Multi‑Head Attention**
```python
phase2 = DeepNet()
phase2.load_state_dict(torch.load("models/phase2_deepnet.pth"))
phase2.eval()
```

### **Full Pipeline Inference**
```python
phase1_out = phase1(x_test)
filtered = filter_anomalies(phase1_out)

final_predictions = phase2(filtered)
```

---

## 📊 7. Results & Metrics

You will get:
- Overall Accuracy  
- F1-score  
- Precision & Recall  
- ROC-AUC  
- Confusion Matrix  
- Per-class performance (for multiclass)

---

## ⚠️ Common Errors & Fixes

### ❌ **Missing keys in state_dict**
```
Missing key(s) in state_dict: attention.in_proj_weight...
```
✔ Ensure model definition **matches exactly** with trained checkpoint.  
✔ Re‑train if architecture changed.

### ❌ CUDA error: Out of Memory
✔ Reduce batch size  
✔ Use CPU run mode  

---

## 📎 8. Contact / Support

For issues, feel free to update the repository or contact the maintainer.

---

💡 *This README is auto‑generated for your project workflow.*

