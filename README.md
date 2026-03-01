<img src="Data/logo/python.png" height="128">
<img src="Data/logo/UofT.png" height="128">


# Airfare Structure Analysis


# Contribution
Instructions for running and contributing to this project are shown here.
## Setup ⚙️
This section will take you through the procedure to take your development environment from zero to hero.
1. Install python from the official [website](https://www.python.org/downloads/).

    The project runs on python `3.12`.


2. Install [git](https://git-scm.com/).

3. Install [pip](https://pip.pypa.io/en/stable/installation/)

3. Clone the repository.

    It is recommended that you use [Github Desktop](https://desktop.github.com/) to clone the project repository.


4. Install project dependencies

    It's recommended to virtual environment to ensure proper project dependencies:
    ```
    python -m venv .venv
   .venv/Scripts/activate
    ```
   
    Project dependecies are listed on requirements.txt. To install on Terminal:
    ```
    python -m pip install -r requirements.txt
    ```


You're now ready to run the project!

## Adding Packages 📦
To add a new package to the virtual environment, install it via:
```
python -m pip install <package>
```
Remember to update your requirements.txt to ensure your additional feature producibility:
```
pip freeze > requirements.txt
```


## Branches 🌿
Branches are organized as follow:

1. `main`: the branch containing the most recent working release. All code in this branch should run perfectly without any known errors.

1. `dev`: branched off of `main`; the most updated version of the project with the newest features and bug fixes.

1. `<feature>`: branched off of `dev`; a feature branch. Features must be tested thoroughly before being merged into dev.

## Taking on Tickets 🎫
Check out the issues tab to see all open tickets.

## Running The Project
To ensure Path consistency, it's recommended to run all the script using terminal from root
```
python path/to/file
```

All images produced by the scripts are stored in `Data/docs/image`

### Data Preprocessing and Exploratory
- The executeable file is marked by `s*.py`. It's recommended for user to run the file sequentially
- Data exploratory done with `s01_EDA.ipynb`
- All data preprocessing from Raw into Processed are stored inside `scripts/`. Data preprocess will store the new data 
in `Data\Interim\adjusted_airline_tickets.csv`


### Model Building
All the model used for model comparison are stored in `scripts/`. To add model for comparison:
- Make a new function that takes (X_train, X_valid, y_train, y_valid) and return the `trained model`. 
- To add the model for evaluation, add the function to `model_to_test` at #2  on `evaluation.py` . We need the `trained model` to be able use `model.predict()` for evaluation.

### Streamlit Dashboard
User can directly access deployed [dashboard](https://fair-fare.streamlit.app/) <br>
To deploy the website locally:
```
streamlit run app.py
```




