# Face Recognition with Pretrained Models

This project demonstrates the use of pretrained models (AlexNet, ResNet, and VGG) for face recognition using the LFW (Labeled Faces in the Wild) dataset.

## Project Structure

- `main.py`: Main Python script containing the implementation of the face recognition task with different pretrained models.

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Scikit-learn
- PIL

You can install the necessary libraries using the following command:
```bash
pip install torch torchvision scikit-learn pillow

```

## Usage
1.Clone the repository:
```bash
git clone https://github.com/munibakar/Face-Recognition-with-Pretrained-Models.git
cd face-recognition
```
2.Run the script:
```bash
python main.py --model [MODEL_NAME] --scenario [SCENARIO] [--bypass_train]

```

## Arguments
- `--model`: Choose from `AlexNet`, `ResNet`, or `VGG` (default: `AlexNet`).
- `--scenario`: Scenario number (1, 2, 3, or 4) (default: 1).
- `--bypass_train`: Skip training and use pre-trained weights if available.

## Example
```bash
python main.py --model ResNet --scenario 2
```
## Result
- The script trains the selected model on the LFW dataset, evaluates its performance, and prints training and test losses along with accuracy.
- Model weights are saved in the weight directory.
