# ML_AT_SCALE
# Setup Guide

Follow these steps to set up your Python environment and install the necessary packages.

## Prerequisites

Make sure you have Python installed on your system. You can download and install Python from the [official Python website](https://www.python.org/downloads/).

## Steps

1. **Clone the repository** (if applicable):

    ```sh
    git clone https://github.com/your-repo/your-project.git
    cd your-project
    ```

2. **Create a virtual environment** (optional but recommended):

    It's a good practice to use a virtual environment to manage dependencies. You can create a virtual environment using the `venv` module:

    ```sh
    python -m venv venv
    ```

    Activate the virtual environment:

    - **On Windows:**

        ```sh
        venv\Scripts\activate
        ```

    - **On macOS and Linux:**

        ```sh
        source venv/bin/activate
        ```

3. **Install the required packages**:

    Make sure you have a `requirements.txt` file in your project directory with the following content:

    Install the packages using pip:

    ```sh
    pip install -r requirements.txt
    ```

## Train 
To train model use

`
python3 main.py
`

## Predict
To Predict Recommendation

`Python3 model.py`