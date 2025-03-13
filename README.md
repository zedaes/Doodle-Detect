# Doodle Detect

## Setup
You most likely would want to set up a Python virtual environment.

1. **Clone the repository:**
    ```sh
    git clone https://github.com/zedaes/Doodle-Detect.git
    cd Doodle-Detect
    ```

2. **Create a virtual environment:**
    ```sh
    python3 -m venv venv
    ```

3. **Activate the virtual environment:**
    - On macOS and Linux:
        ```sh
        source venv/bin/activate
        ```
    - On Windows:
        ```sh
        .\venv\Scripts\activate
        ```

4. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage
### Download Data
First you need to decide what data you want to train your model on, to do this first edit the `doodles_to_download.txt` to the list of doodles you want to download. 

Then run the `download_doodles.py` with `python download_doodles.py`. This will create a `/data` folder with each of your doodles in a `.ndjson` format.

### Train Model
To train the model you need to edit the `doodles_to_train.txt` to what doodles ou want the model to be trained on. Then run  `train_model.py` with `python train_model.py`. This will generate a `doodle_model.h5` model.

## Contributing
If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
If you have any questions or suggestions, feel free to open an issue or contact me at [zedaes@proton.me](mailto:zedaes@proton.me).
