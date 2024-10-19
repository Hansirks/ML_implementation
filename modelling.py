'''Power Optimization for Factory Machines: 
Problem Overview:

The goal of this project was to minimize the total power consumption of 20 machines in a factory while achieving a target total Goods Per Hour (GPH) of 9,000. 
The solution needed to:

    Learn a model for each machine type that predicts power usage based on machine inputs.
    Optimize the GPH for each machine to minimize total power consumption while meeting the target.'''


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import pickle

# Training the models for multiple approaches

def train_linear_model(data):
    X = data[['input_1', 'input_2', 'input_3']]
    y = data['power']
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_random_forest_model(data):
    X = data[['input_1', 'input_2', 'input_3']]
    y = data['power']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

'''Modeling:

    Two machine learning models were trained:
        Linear Regression: Simple model with coefficients for each input.
        Random Forest: A more complex, non-linear model.
    Models were trained separately for each machine type, using input_1, input_2, and input_3 (GPH) as predictors of power consumption.'''
def train_models(machine1_data, machine2_data):
    # Linear Regression
    model1_lr = train_linear_model(machine1_data)
    model2_lr = train_linear_model(machine2_data)

    # Random Forest
    model1_rf = train_random_forest_model(machine1_data)
    model2_rf = train_random_forest_model(machine2_data)

    return {
        'linear_regression': (model1_lr, model2_lr),
        'random_forest': (model1_rf, model2_rf)
    }

# Function to export the models so they can be reused without retraining
def save_models(models_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(models_dict, f)

# Function to load the models
def load_models(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def power_machine(gph, model, input_1, input_2):
    input_3 = gph  # Variable GPH
    return model.predict([[input_1, input_2, input_3]])[0]

def total_power(gph_values, models, input_1, input_2, num_machines1, num_machines2):
    total_power_usage = 0
    model1, model2 = models

    for i in range(num_machines1):
        total_power_usage += power_machine(gph_values[i], model1, input_1, input_2)

    for i in range(num_machines1, num_machines1 + num_machines2):
        total_power_usage += power_machine(gph_values[i], model2, input_1, input_2)

    return total_power_usage

def gph_constraint(gph_values, target_gph):
    return np.sum(gph_values) - target_gph

'''Optimization:

    The optimization process used the models to minimize total power usage, subject to the constraint that the sum of the GPH values for all machines must equal 9,000.
    I  used the Sequential Least Squares Programming (SLSQP) method for efficient optimization.'''

def optimize_power(models, input_1=25, input_2=6, num_machines1=10, num_machines2=10, target_gph=9000, bounds1=(180, 600), bounds2=(300, 1000)):
    # Bounds for GPH (Input 3) for each machine type
    bounds = [bounds1] * num_machines1 + [bounds2] * num_machines2

    #Initial guess for GPH value
    initial_guess = [sum(bounds1)/2] * num_machines1 + [sum(bounds2)/2] * num_machines2

    # Defning constraints  
    constraints = {'type': 'eq', 'fun': lambda gph_values: gph_constraint(gph_values, target_gph)}

    result = minimize(total_power, initial_guess, method='SLSQP', bounds=bounds,
                      constraints=constraints, args=(models, input_1, input_2, num_machines1, num_machines2))

    optimal_gph_values = result.x
    total_power_usage = total_power(optimal_gph_values, models, input_1, input_2, num_machines1, num_machines2)

    return optimal_gph_values, total_power_usage

'''Model Comparison:

    Linear Regression and Random Forest were compared using the Mean Squared Error (MSE) to assess their performance on the training data.
    Random Forest typically showed lower MSE, indicating better predictive accuracy.
'''

def compare_models(machine1_data, machine2_data, models_dict):
    comparisons = {}
    X1 = machine1_data[['input_1', 'input_2', 'input_3']]
    y1 = machine1_data['power']
    
    X2 = machine2_data[['input_1', 'input_2', 'input_3']]
    y2 = machine2_data['power']

    for model_name, (model1, model2) in models_dict.items():
        # Predict and compute the Mean Squared Error
        y1_pred = model1.predict(X1)
        y2_pred = model2.predict(X2)

        mse1 = mean_squared_error(y1, y1_pred)
        mse2 = mean_squared_error(y2, y2_pred)

        comparisons[model_name] = {'mse_machine_1': mse1, 'mse_machine_2': mse2}
    
    return comparisons


# Main 
# Load the datasets
machine1_path = './machine1.csv'
machine2_path = './machine2.csv'

machine1_df = pd.read_csv(machine1_path)
machine2_df = pd.read_csv(machine2_path)

'''Data Preparation:

    I used provided datasets for two types of machines.
    The data was filtered to include only valid rows (where check value is between 90 and 110).'''

machine1_valid = machine1_df[(machine1_df['check'] >= 90) & (machine1_df['check'] <= 110)]
machine2_valid = machine2_df[(machine2_df['check'] >= 90) & (machine2_df['check'] <= 110)]

# Train models
models_dict = train_models(machine1_valid, machine2_valid)

save_models(models_dict, 'trained_models.pkl')

loaded_models = load_models('trained_models.pkl')

# Compare model performance
comparison = compare_models(machine1_valid, machine2_valid, loaded_models)
print("Model comparison: ", comparison)

optimal_gph_values, total_power_usage = optimize_power(loaded_models['linear_regression'], input_1=25, input_2=6, target_gph=9000)

# Output results
'''Optimal GPH Values:

    The solution provided the optimal GPH values for each machine, allowing the factory to produce exactly 9,000 GPH while minimizing power consumption.'''
print("Optimal GPH values for each machine:")
for i in range(len(optimal_gph_values)):
    print(f"Machine {i+1}: {optimal_gph_values[i]:.2f} GPH")



'''Total Power Usage:

    The total minimized power usage across all machines was calculated and displayed.'''
print(f"\nTotal power usage: {total_power_usage:.2f} units")




'''
    -Key Observations :

    Linear models are easy to interpret and provide quick solutions, but may lack the accuracy of more complex models like Random Forest.
    Efficient optimization techniques like SLSQP can quickly solve high-dimensional problems like minimizing power usage across many machines.
    
    -Possible Next Measures:

    Test with additional machine learning models to explore further improvements in predictive accuracy.
    Explore real-time optimization techniques for dynamic environments.
'''