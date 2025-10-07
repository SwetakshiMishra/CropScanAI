
# Contribution Guide

Welcome! Here’s how you can contribute and set up the project locally.


## 1️⃣ Fork and Clone the Repository

1. Fork this repository on GitHub.
2. Clone it to your local machine:

```bash
git clone <your-forked-repo-url>
cd <repo-folder>
````



## 2️⃣ Create a Python Virtual Environment

Open a terminal in the project folder and run:

```bash
python -m venv venv
```

Activate it:

* **Windows:**

```bash
venv\Scripts\activate
```

* **Mac/Linux:**

```bash
source venv/bin/activate
```



## 3️⃣ Install Required Libraries

With the virtual environment activated, install dependencies:

```bash
pip install tensorflow flask flask-babel opencv-python matplotlib
```

> ⚠️ Optional: `numpy` and `werkzeug` may be required, but they’re usually installed automatically.



## 4️⃣ Train the Model (First-Time Setup Only)

If you don’t already have `pearl_millet_ergot_model.h5`, train the model:

```bash
python train.py
```

This will:

* Train the CNN on the dataset
* Save the trained model as `pearl_millet_ergot_model.h5`

> If you already have this file, you can skip this step.



## 5️⃣ Test the Model (Optional)

To verify the model works:

```bash
python test.py
```

> Ensure there’s a `test.jpg` image in your folder.
> Output will be something like:

```
Diseased: Ergot
```

or

```
Healthy
```


## 6️⃣ Run the Flask App

Start the Flask server:

```bash
python app.py
```

You should see:

```
* Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```


## 7️⃣ Open in Browser

Go to:

👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

You’ll see the homepage (`index.html`). Navigate to `/identify` to upload an image and get results.


## Download Dataset (Drive Links)

- [Train Zip](https://drive.google.com/file/d/1oUbvcYsj4cKbeSTVbizR-p_mToBralFH/view?usp=drive_link)
- [Val Zip](https://drive.google.com/file/d/1-k6XOZZcV5HjZOHTM0G-o32sbRn6v5Vn/view?usp=drive_link)


