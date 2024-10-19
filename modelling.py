import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import pickle

# Step 1: Training the models for multiple approaches

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

# Method to train and return different models for comparison
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

# Function to load the models back
def load_models(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Step 2: Optimization

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

# Constraint: Sum of GPH must equal the target
def gph_constraint(gph_values, target_gph):
    return np.sum(gph_values) - target_gph

# Main function to run optimization with configurable inputs
def optimize_power(models, input_1=25, input_2=6, num_machines1=10, num_machines2=10, target_gph=9000, bounds1=(180, 600), bounds2=(300, 1000)):
    # Bounds for GPH (Input 3) for each machine type
    bounds = [bounds1] * num_machines1 + [bounds2] * num_machines2

    # Initial guess for GPH values (mid-range values)
    initial_guess = [sum(bounds1)/2] * num_machines1 + [sum(bounds2)/2] * num_machines2

    # Constraints
    constraints = {'type': 'eq', 'fun': lambda gph_values: gph_constraint(gph_values, target_gph)}

    # Optimize
    result = minimize(total_power, initial_guess, method='SLSQP', bounds=bounds,
                      constraints=constraints, args=(models, input_1, input_2, num_machines1, num_machines2))

    optimal_gph_values = result.x
    total_power_usage = total_power(optimal_gph_values, models, input_1, input_2, num_machines1, num_machines2)

    return optimal_gph_values, total_power_usage

# Comparison function to evaluate different models
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

# Step 3: Putting it all together

# Load the datasets
machine1_path = './machine1.csv'
machine2_path = './machine2.csv'

machine1_df = pd.read_csv(machine1_path)
machine2_df = pd.read_csv(machine2_path)

# Filter rows based on 'check' value between 90 and 110 for both machines
machine1_valid = machine1_df[(machine1_df['check'] >= 90) & (machine1_df['check'] <= 110)]
machine2_valid = machine2_df[(machine2_df['check'] >= 90) & (machine2_df['check'] <= 110)]

# Train models
models_dict = train_models(machine1_valid, machine2_valid)

# Export the models to be used in another program without retraining
save_models(models_dict, 'trained_models.pkl')

# Load models (for reusability)
loaded_models = load_models('trained_models.pkl')

# Compare model performance
comparison = compare_models(machine1_valid, machine2_valid, loaded_models)
print("Model comparison: ", comparison)

# Optimize power usage with configurable parameters
optimal_gph_values, total_power_usage = optimize_power(loaded_models['linear_regression'], input_1=25, input_2=6, target_gph=9000)

# Output results
print("Optimal GPH values for each machine:")
for i in range(len(optimal_gph_values)):
    print(f"Machine {i+1}: {optimal_gph_values[i]:.2f} GPH")

print(f"\nTotal power usage: {total_power_usage:.2f} units")
