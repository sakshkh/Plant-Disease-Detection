🌱 Plant Disease Detection using Deep Learning
This project is an AI-powered system for detecting plant diseases from leaf images. It uses a Convolutional Neural Network (CNN) trained on the PlantVillage Dataset to classify various plant diseases with high accuracy.

📌 Features
✅ Trained using TensorFlow/Keras
✅ Uses CNN-based image classification
✅ Supports multiple plant species & diseases
✅ Real-time prediction using uploaded images
✅ Model saved in .keras format

📂 Project Structure
bash
Copy
Edit
📁 Plant-Disease-Detection  
│── 📁 PlantVillage-Dataset/   # Dataset for training  
│── 📄 train.py                # Model training script  
│── 📄 predict.py              # Image prediction script  
│── 📄 plant_disease_model.keras  # Trained model  
│── 📄 requirements.txt        # Dependencies  
│── 📄 README.md               # Project documentation 

🛠️ Installation
1️⃣ Clone the Repository:
git clone https://github.com/sakshkh/Plant-Disease-Detection.git
cd Plant-Disease-Detection

2️⃣ Install Dependencies:
pip install -r requirements.txt

3️⃣ Run the Model Training (Optional):
python train.py

4️⃣ Run the Prediction Script:
python predict.py --image_path path/to/image.jpg

📊 Dataset
I have used the PlantVillage Dataset, which contains color images of plant leaves labeled with diseases.
🔗 Dataset Source: PlantVillage on GitHub

🤝 Contributing
Contributions are welcome! Please open an issue or pull request for suggestions or improvements.
📧 Contact: sakshhamk24@gmail.com
