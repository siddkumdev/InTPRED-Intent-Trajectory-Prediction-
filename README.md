# Multi-Modal Trajectory Prediction for Beginners

Welcome! This project is about predicting where a person or object (like a pedestrian or car) is going to move in the future. Imagine a self-driving car trying to guess where a person on the sidewalk will walk next so it can drive safely. That's what this code does!

Because the future is uncertain, the model guesses **multiple possible paths** (called "multi-modal" predictions) and tells us how confident it is in each path. It looks at the person's past movements and a map of their surroundings to make these guesses.

This guide will walk you through exactly how to set up, run, and understand this project, even if you are new to programming.

---

## 🛠️ Step 1: Getting Ready (Prerequisites)

Before you can run the code, you need a few tools installed on your computer.

1.  **Python**: The programming language this project is written in. You need Python installed. If you don't have it, download it from [python.org](https://www.python.org/).
2.  **Required Libraries**: This code uses some popular external packages (like PyTorch for AI, and Matplotlib for drawing graphs). 
    
    Open your computer's terminal (or command prompt) and type this command, then press Enter:
    ```bash
    pip install torch torchvision torchaudio nuscenes-devkit matplotlib numpy
    ```
    *Wait for the installation to finish.*

## 📂 Step 2: Getting the Data

This project learns how to predict movements using real-world data from self-driving cars. We use a famous dataset called **nuScenes**.

1.  **Download**: You need the **nuScenes mini** dataset (a small, easy-to-use version). You can find it [here](https://drive.google.com/drive/folders/1g5KgxG0p8-MmTiXkNtCpoYSIkdBQprEm).
2.  **Create Folder**: Create a folder named `data` in your main project directory (where `train.py` and `eval.py` are located).
3.  **Extract with 7-Zip**: 
    * Right-click the downloaded dataset file.
    * Select **7-Zip** > **Extract files...** * Choose your new `data` folder as the destination. 
    * *Note: Using 7-Zip ensures all nested files are extracted correctly without errors.*
4.  **Verify Folder Structure**: It is crucial that your folders match the structure shown in the image below. Inside your `data` folder, you should see these specific items:
    * 📁 `maps`
    * 📁 `samples`
    * 📁 `sweeps`
    * 📁 `v1.0-mini`
    * 📄 `.v1.0-mini.txt`
    * 📄 `LICENSE`

> **Pro Tip:** If you see another folder inside `data` (like `data/v1.0-mini/v1.0-mini`), move the contents up one level so they match the list above!

### 🧠 Step 3: Teaching the AI (Training)

Now that you have the code and the data, it's time to "teach" the artificial intelligence. This is called **training**. The AI will look at thousands of examples of past movements and try to learn the patterns.

1.  Open your terminal.
2.  Make sure you are in the folder where the project is saved.
3.  Run the training script by typing:
    ```bash
    python train.py
    ```
4.  **What to expect:** You will see text appearing on the screen showing "Epoch 1/70", "Epoch 2/70", etc. An "Epoch" is one full pass through the dataset. As the epochs go up, the "Train Loss" number should generally go down, meaning the AI is getting better!
5.  This process will create a new folder called `outputs`. Inside, it will save images showing its progress and, finally, a file called `final_model.pth`. This file is the "brain" of your trained AI.

## 🧪 Step 4: Testing the AI (Evaluation)

Once training is complete (or if you already have a `final_model.pth` file in your `outputs` folder), you can test how smart the AI has become.

1.  In your terminal, run the evaluation script by typing:
    ```bash
    python eval.py
    ```
2.  **What to expect:** The script will load your trained model and test it on new data it hasn't learned from yet.
3.  It will print out scores (called "Benchmarks").
4.  **It will also generate pictures!** Look inside your `outputs` folder. You will find:
    *   **PNG images** (e.g., `eval_sample_1.png`): These are graphs showing the past path (blue), the actual future path (green), and the AI's guesses (orange, purple, red).
    *   **GIF animations** (e.g., `radar_sample_1.gif`): These are cool animated "radar" views showing the prediction unfolding over time!

## 📊 Understanding the Scores

When you run `eval.py`, you'll see two main types of scores:

*   **ADE (Average Displacement Error):** On average, how far off was the AI's predicted path from the real path? Lower is better! (Measured in meters).
*   **FDE (Final Displacement Error):** How far off was the AI's guess at the *very last* moment of the prediction? This tells us if it got the final destination right. Lower is better! (Measured in meters).

The code will show you the error for its "Best Guess" (Top-1) and also the error for the path that ended up being the closest to reality (Geometric Limits).

---
