# CLEAR
CLEAR stands for Climate-affected Loss Enhancement and Adaptive Restoration.     
You can install the requirements used to setup the project using:     

```bash
pip install -r requirements.txt
```

This project demonstrates model inference using Streamlit. You can run the apps using:

```bash
streamlit run app_modular.py
# or
streamlit run app_tanet.py
```
Note: You can train the models on your own using *.ipynb files to get your own .pth files. 

---

## Project Structure

### 📁 `Model/`

* Contains code, model files (`.pth`), plots, etc., organized in separate folders for each model.
* **Note**: Model training was done in sessions (paused/resumed/saved across devices), so file naming may be inconsistent. However, if a `.pth` file is inside the `TANet/` folder, it belongs to the TANet model, and so on.

---

## Comparisons

* **`comparison-combined/`**: Comparison between Modular and TANet on the **mixed** dataset.
* **`comparison-haze/`**: Comparison on the **haze** dataset.
* **`comparison-rain/`**: Comparison on the **rain** dataset.
* **`modular-vs-unified/`**: Code for comparing Modular vs. Unified model performance.

---

## Demos & Presentations

* 🎥 `Video Demonstration.mp4`: Video Demonstration for the presentation. Link: https://drive.google.com/file/d/1OqC-4BTfL92Bllm7ZHmv-w5FMCpucj7B/view?usp=sharing     
* 🎥 `demo.mp4`: Streamlit app demo for `app_modular.py` and `app_tanet.py`. Link: https://drive.google.com/file/d/1S-RrodTNXMILLEoecDs90Tu5eXkbFkrg/view?usp=sharing     
* 🗄️ `Rain100 Dataset`: Link: https://www.kaggle.com/datasets/bshaurya/rain-dataset      
* 🗄️ `RESIDE-6K Dataset`: Link: https://www.kaggle.com/datasets/kmljts/reside-6k      
