# CLEAR

This project demonstrates model inference using Streamlit. You can run the apps using:

```bash
streamlit run app_modular.py
# or
streamlit run app_tanet.py
```

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

* 🎥 `ATDL_Video_720p.zip`: Contains the video recording of the presentation.
* 📹 `demo.mp4`: Streamlit app demo for `app_modular.py` and `app_tanet.py`.
