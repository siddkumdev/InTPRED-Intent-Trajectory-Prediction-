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

If you are new to running code from a terminal, don't worry! Just follow these exact steps:

### Phase 1: Opening your Terminal and Finding the Folder
Before you can run the code, you need to tell your computer's terminal *where* the code is saved.

1.  **Open your Terminal:**
    * **Windows:** Press the Start button, type `cmd`, and press Enter to open the Command Prompt.
    * **Mac:** Press Command + Space, type `Terminal`, and press Enter.
2.  **Navigate to the project folder:** Use the `cd` (change directory) command to go to the folder where you saved this project. For example, if you saved it on your Desktop in a folder called "TrajectoryProject", you would type:
    ```bash
    cd Desktop/TrajectoryProject
    ```
    *(Press Enter after typing the command. Your terminal should now show that you are inside that folder).*

### Phase 2: Training the AI Model
Now we need to "teach" the AI. It will look at thousands of examples of past movements to learn patterns.

1.  In your terminal, type the following command and press Enter:
    ```bash
    python train.py
    ```
2.  **What happens next?** You will see text start scrolling on your screen (e.g., "Epoch 1/70", "Epoch 2/70"). An "Epoch" is one full practice round for the AI. This process might take a while depending on your computer.
3.  As the epochs increase, look at the "Train Loss" number. It should slowly get smaller—this means the AI is learning and making fewer mistakes!
4.  Once it reaches the final epoch, it will automatically create a new folder called `outputs` and save its "brain" inside a file named `final_model.pth`.

### Phase 3: Evaluating the Model
Once training is 100% complete, it's time to test the AI with data it has never seen before to see how smart it really is.

1.  In the exact same terminal window, type this command and press Enter:
    ```bash
    python eval.py
    ```
2.  The script will load the "brain" (`final_model.pth`) it just created and start making predictions. 
3.  It will print out its final scores (benchmarks) in the terminal, and it will save visual graphs and animations into the `outputs` folder so you can see its guesses with your own eyes!

## Example outputs / results

When you run the evaluation script, you will receive both quantitative scores and visual outputs:

### 1. Quantitative Scores (Printed in terminal)
* **ADE (Average Displacement Error):** On average, how far off was the AI's predicted path from the real path? Lower is better (measured in meters).
* **FDE (Final Displacement Error):** How far off was the AI's guess at the *very last* moment of the prediction? Lower is better (measured in meters).
* *Note:* The code shows the error for its "Best Guess" (Top-1) and the error for the path that ended up being the closest to reality.

### 2. Visual Outputs
The code generates images showing the **Past Path (Blue)**, the **Actual Future Path (Green)**, and the **AI's Multiple Guesses (Red, Orange, Purple)**. Here are some examples of what the AI generates:

**Sample 3 Output:**
![Evaluation Sample 3](https://raw.githubusercontent.com/siddkumdev/InTPRED-Intent-Trajectory-Prediction-/main/outputs/eval_sample_3.png)

**Sample 7 Output:**
![Evaluation Sample 7](https://raw.githubusercontent.com/siddkumdev/InTPRED-Intent-Trajectory-Prediction-/main/outputs/eval_sample_7.png)

**Sample 9 Output:**
![Evaluation Sample 9](https://raw.githubusercontent.com/siddkumdev/InTPRED-Intent-Trajectory-Prediction-/main/outputs/eval_sample_9.png)

*(You can also find GIF animations like `radar_sample.gif` in your `outputs` folder showing these predictions unfolding dynamically over time!)*

---
*Note: Ensure this repository is publicly accessible during the evaluation period as per the submission guidelines.*
