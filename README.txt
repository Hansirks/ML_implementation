Power Optimization for Factory Machines
1.Project Overview

This project focuses on two main objectives:

    Modeling Power Usage: Using machine learning to model power consumption as a function of three inputs for two types of machines.
    Optimization: Minimizing the total power usage of 20 machines while achieving a target total Goods Per Hour (GPH) for the factory.

The solution is designed to allow flexibility in the number of machines, machine inputs, and the target GPH, while optimizing power consumption efficiently.

Setup Instructions
2.Prerequisites:

    Python 3.x

3.Required packages (can be installed via pip):
        pandas
        numpy
        scikit-learn
        scipy
        matplotlib (for optional plots)
        pickle (for saving/loading models)

    You can install these dependencies using the following command:

    bash

    pip install pandas numpy scikit-learn scipy matplotlib

4.Dataset Setup:

    Place the dataset CSV files (machine1.csv and machine2.csv) in the same directory as the Python script.
        These files should have columns: input_1, input_2, input_3, power, and check.
        The script filters data based on the check column (values between 90 and 110).

5.Running the Solution:

    Training the Models:
        The models are trained automatically when the script is run. The script applies multiple machine learning models (Linear Regression and Random Forest) to predict power usage for each machine type.

    Optimization:
        The optimization process uses the trained models to minimize total power usage for a set number of machines and a target GPH.
        The optimization parameters (such as number of machines, input_1, input_2, and GPH bounds) can be modified directly in the script.

    How to Run:
        Ensure the datasets are in place, and run the script with Python:

    bash

    python modelling.py

    Outputs:
        The script will output the optimal GPH values for each machine and the total power consumption.
        It also includes optional comparison metrics for different models (Mean Squared Error) for evaluation purposes.

6.Reusable Model
The models are saved using Python's pickle library. This allows them to be reused in another program without retraining. The models are stored in a file called trained_models.pkl and can be loaded with the provided load_models() function.

7.Assumptions & Limitations
  Assumptions:

    The data provided has been cleaned and contains no missing or invalid values except those filtered by the check column.
    The relationship between inputs and power usage is linear or can be approximated by the models provided (Linear Regression or Random Forest).

8.Limitations:

    The optimization method assumes the power usage of machines is continuous and smooth, which may not hold for all machines in a real-world setting.
    The models are trained on historical data and may not generalize well to machines operating outside the range of the dataset provided.
    Only two types of models (Linear Regression and Random Forest) are compared, though other models could potentially perform better.

9.Configuration

The solution supports the following configurable parameters:

    input_1, input_2: These are static values for each machine, defaulting to 25 and 6 respectively.
    num_machines1, num_machines2: These represent the number of instances of each machine type. Default is 10 for each.
    target_gph: The total GPH target for all machines combined. Default is 9,000.
    bounds1, bounds2: These represent the minimum and maximum GPH that each machine type can produce.

    You can modify these parameters in the optimize_power() function to customize the setup.

10.Model Comparison

    The script automatically trains two models:

    Linear Regression: Simple linear model with coefficients representing the contribution of each input to power consumption.
    Random Forest: An ensemble model that builds multiple decision trees for better predictive performance, especially with non-linear data.

    The script compares these models using the Mean Squared Error (MSE) and outputs the results.
