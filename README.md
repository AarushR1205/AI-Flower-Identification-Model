# 🌸 AI Flower Identification Model

A web application that identifies flowers from images using a deep learning model (DenseNet121) trained on five flower species. Built with PyTorch and Streamlit.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- Upload or capture flower images for instant identification
- DenseNet121 CNN model fine-tuned for 5 flower classes: Daisy, Dandelion, Rose, Sunflower, Tulip
- Adjustable confidence threshold for predictions
- Interactive UI built with Streamlit

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/ai-flower-identification-model.git
   cd ai-flower-identification-model
   ```

2. **Create a virtual environment (recommended):**
   ```sh
   python -m venv venv
   venv\Scripts\activate   # On Windows
   # Or
   source venv/bin/activate   # On macOS/Linux
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Download the model weights:**
   - Place `densenet121_flower.pth` in the project root directory.

## Usage

Start the Streamlit app:
```sh
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

- Upload an image or use your webcam to capture a flower photo.
- Adjust the confidence threshold in the sidebar.
- View the predicted flower type and confidence score.

## Project Structure

```
.
├── app.py                   # Main Streamlit application
├── densenet121_flower.pth   # Trained model weights
├── requirements.txt         # Python dependencies
├── LICENSE.txt              # License information
└── ...
```

## Model Details

- **Architecture:** DenseNet121 (PyTorch)
- **Input Size:** 224x224 pixels
- **Classes:** Daisy, Dandelion, Rose, Sunflower, Tulip
- **Training:** Transfer learning on public flower datasets

## License

See [LICENSE.txt](LICENSE.txt) for details.  
Flower images are licensed under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- Flower dataset contributors

---
## User Interface
<br><br>
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/b9fe9c44-35f1-467e-88bc-bafe9092ff5c" />


**Made with ❤️ for flower lovers and AI enthusiasts!**
