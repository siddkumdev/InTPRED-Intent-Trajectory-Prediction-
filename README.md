Here is the updated README, reorganized and expanded to strictly match the required sections outlined in your image. 

# Multi-Modal Trajectory Prediction

## Project overview
Welcome! This project is about predicting where a person or object (like a pedestrian or car) is going to move in the future. Imagine a self-driving car trying to guess where a person on the sidewalk will walk next so it can drive safely. That's what this code does! 

Because the future is uncertain, the model guesses **multiple possible paths** (called "multi-modal" predictions) and tells us how confident it is in each path. It looks at the person's past movements and a map of their surroundings to make these intelligent guesses.

## Model architecture
The core of this project relies on a multi-modal trajectory prediction architecture. Instead of outputting a single, deterministic future path (which is often inaccurate in complex real-world scenarios), the architecture is designed to:
* Take historical movement data (past trajectories) and environmental context (maps) as inputs.
* Process these inputs to understand the agent's behavior and the physical constraints of their surroundings.
* Output **multiple potential future paths** alongside confidence scores for each path, effectively mapping out the distribution of likely future movements.

## Dataset used
This project learns how to predict movements using real-world data from self-driving cars. We use a famous dataset called the **nuScenes mini** dataset (a small, easy-to-use version).

1.  **Download**: You can find the dataset [here](https://drive.google.com/drive/folders/1g5KgxG0p8-MmTiXkNtCpoYSIkdBQprEm).
2.  **Create Folder**: Create a folder named `data` in your main project directory (where your scripts are located).
3.  **Extract with 7-Zip**: 
    * Right-click the downloaded dataset file and select **7-Zip** > **Extract files...** * Choose your new `data` folder as the destination. *(Note: Using 7-Zip ensures all nested files are extracted correctly without errors.)*
4.  **Verify Folder Structure**: Inside your `data` folder, you must have this exact structure:
    * 📁 `maps`
    * 📁 `samples`
    * 📁 `sweeps`
    * 📁 `v1.0-mini`
    * 📄 `.v1.0-mini.txt`
    * 📄 `LICENSE`

> **Pro Tip:** If you see another folder inside `data` (like `data/v1.0-mini/v1.0-mini`), move the contents up one level so they match the list above!

## Setup & installation instructions
Before you can run the code, you need a few tools installed on your computer.

1.  **Python**: You need Python installed. If you don't have it, download it from [python.org](https://www.python.org/).
2.  **Required Libraries**: This code uses several external packages (like PyTorch for AI, and Matplotlib for drawing graphs). 
    
    Open your terminal or command prompt and run the following command:
    ```bash
    pip install torch torchvision torchaudio nuscenes-devkit matplotlib numpy
    ```

## How to run the code

**1. Training the Model**
To "teach" the AI, it needs to look at thousands of examples of past movements to learn patterns.
* Open your terminal and navigate to the project folder.
* Run the training script:
  ```bash
  python train.py
  ```
* You will see the training progress (e.g., "Epoch 1/70"). As epochs increase, the "Train Loss" should decrease. This process creates an `outputs` folder containing a `final_model.pth` file (your trained AI).

**2. Evaluating the Model**
Once training is complete, you can test how smart the AI has become on new data.
* Run the evaluation script:
  ```bash
  python eval.py
  ```
* The script will load your trained model, test it, print benchmark scores, and generate visual results in the `outputs` folder.

## Example outputs / results

When you run the evaluation script, you will receive both quantitative scores and visual outputs:

**1. Quantitative Scores (Printed in terminal):**
* **ADE (Average Displacement Error):** On average, how far off was the AI's predicted path from the real path? Lower is better (measured in meters).
* **FDE (Final Displacement Error):** How far off was the AI's guess at the *very last* moment of the prediction? Lower is better (measured in meters).
* *Note:* The code shows the error for its "Best Guess" (Top-1) and the error for the path that ended up being the closest to reality.

**2. Visual Outputs (Saved in `outputs/` folder):**
* **PNG images** (e.g., `eval_sample_1.png`): Graphs showing the past path (blue), the actual future path (green), and the AI's guesses (orange, purple, red).
* **GIF animations** (e.g., `radar_sample_1.gif`): Animated "radar" views showing the prediction unfolding over time.

---
*Note: Ensure this repository is publicly accessible during the evaluation period as per the submission guidelines.*