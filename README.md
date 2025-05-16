**Dog Breed Prediction Using CNN**  
🐶 A deep learning project for predicting dog breeds from images using a Convolutional Neural Network (CNN). This repository contains a Python script that classifies dog breeds (`scottish_deerhound`, `maltese_dog`, `bernese_mountain_dog`) from the Kaggle "Dog Breed Identification from Competition" dataset, leveraging Keras with TensorFlow backend.

### Features
- **Data Preprocessing**: Loads and preprocesses images (224x224x3) with normalization (dividing by 255) and creates a one-hot encoded target matrix.
- **Model**: Custom CNN architecture with:
  - Two Conv2D layers (32 and 16 filters) with ReLU activation and MaxPool2D.
  - Flatten layer followed by Dense layers (64 units and 3 units with softmax for 3 classes).
- **Dataset**: Utilizes a subset of the [Kaggle Dog Breed Identification dataset](https://www.kaggle.com/c/dog-breed-identification) with 3 selected breeds.
- **Training**: Trains the model with a batch size of 32 over 10 epochs, using Adam optimizer and categorical cross-entropy loss.
- **Evaluation**: Displays training/validation accuracy plots and reports test set accuracy.
- **Prediction**: Includes a sample prediction on a test image with visualization.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/[MahirHossain12]/[Dog-Breed-prediction].git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Kaggle API (required for dataset download):
   - Upload your `kaggle.json` file (obtained from Kaggle account settings).
   - Run the setup commands in the script or manually:
     ```bash
     mkdir -p ~/.kaggle
     cp kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```
4. Download and prepare the dataset:
   ```bash
   pip install kaggle
   mkdir dog_dataset
   kaggle datasets download catherinehorng/dogbreedidfromcomp
   unzip dog_dataset/dogbreedidfromcomp.zip -d dog_dataset
   rm dog_dataset/dogbreedidfromcomp.zip dog_dataset/sample_submission.csv
   ```

### Usage
1. Run the script to preprocess data, train the model, and evaluate:
   ```bash
   python dog_breed_prediction.py
   ```
2. View the generated accuracy plot and test prediction output.

### Results
- **Accuracy**: Achieves [X]% accuracy on the test set (update with your actual result after running the script).
- **Sample Prediction**: Displays an example test image with the original and predicted breed labels.

### Dependencies
- Python 3.8+
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- tqdm
- scikit-learn
- kaggle

### Dataset
This project uses a subset of the [Kaggle Dog Breed Identification dataset](https://www.kaggle.com/c/dog-breed-identification) by Catherine Horng, focusing on `scottish_deerhound`, `maltese_dog`, and `bernese_mountain_dog`. The dataset is downloaded and processed within the script.

### Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, enhancements, or additional breed support.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Customization Notes:
- **Accuracy**: The current code doesn’t display the final test accuracy in your snippet. After running the script, update the "Results" section with the actual accuracy (e.g., `round((score[1]*100), 2)` from the `model.evaluate` output).
- **Model**: The description reflects the CNN architecture in your code. If you plan to experiment with pre-trained models (e.g., ResNet50), update the "Model" section accordingly.
- **Dataset**: I specified the Kaggle dataset and the three breeds you’re using. If you expand to more breeds, adjust the `CLASS_NAMES` and description.
- **Requirements.txt**: Create a `requirements.txt` file with the listed dependencies (e.g., `tensorflow`, `numpy`, `pandas`, etc.) for users to install easily.
- **File Name**: The description uses `dog_breed_prediction.py` as the main script name, matching your `<DOCUMENT filename="dog_breed_prediction.py">`. Ensure this matches your actual file name.

### Next Steps:
1. **Run the Script**: Execute the code to get the test accuracy and update the "Results" section.
2. **Add Visuals**: If you generate additional plots or save model weights, mention them in the "Usage" or "Results" sections.
3. **Expand Features**: If you add data augmentation, cross-validation, or a prediction API, update the "Features" section.
4. **Share Output**: If you encounter issues or want to refine the description further, share the script’s output (e.g., accuracy, errors) or additional code.

