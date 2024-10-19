'''Power Optimization for Factory Machines: 
Problem Overview:

The goal of this project was to minimize the total power consumption of 20 machines in a factory while achieving a target total Goods Per Hour (GPH) of 9,000. 
The solution needed to:

    Learn a model for each machine type that predicts power usage based on machine inputs.
    Optimize the GPH for each machine to minimize total power consumption while meeting the target.'''


from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
import pickle

# Optional Flags for saving/loading models
save_model = False
load_model = False

# File paths for saving models
model_file_path = "trained_models.pkl"

# Function to train the models for both machine types
'''Modeling:

    Two machine learning models were trained:
        Linear Regression: Simple model with coefficients for each input.
        Random Forest: A more complex, non-linear model.
    Models were trained separately for each machine type, using input_1, input_2, and input_3 (GPH) as predictors of power consumption.'''

def train_models(data1, data2):
    # Preprocess the data
    '''Data Preparation:

    I used provided datasets for two types of machines.
    The data was filtered to include only valid rows (where check value is between 90 and 110).'''

    data1 = data1[(data1['check'] >= 90) & (data1['check'] <= 110)]
    data2 = data2[(data2['check'] >= 90) & (data2['check'] <= 110)]
    
    X1 = data1[['input_1', 'input_2', 'input_3']]
    y1 = data1['power']
    
    X2 = data2[['input_1', 'input_2', 'input_3']]
    y2 = data2['power']

    # Train the models
    model1_lr = LinearRegression().fit(X1, y1)
    model1_rf = RandomForestRegressor().fit(X1, y1)
    
    model2_lr = LinearRegression().fit(X2, y2)
    model2_rf = RandomForestRegressor().fit(X2, y2)
     # Calculate MSE for model comparison
    mse1_lr = mean_squared_error(y1, model1_lr.predict(X1))
    mse1_rf = mean_squared_error(y1, model1_rf.predict(X1))
    
    mse2_lr = mean_squared_error(y2, model2_lr.predict(X2))
    mse2_rf = mean_squared_error(y2, model2_rf.predict(X2))

    '''Model Comparison:

    Linear Regression and Random Forest were compared using the Mean Squared Error (MSE) to assess their performance on the training data.
    Random Forest typically showed lower MSE, indicating better predictive accuracy.
    '''


    if mse1_lr < mse1_rf:
        best_model1 = model1_lr
        best_model1_name = "Linear Regression"
    else:
        best_model1 = model1_rf
        best_model1_name = "Random Forest"

    # Choose the best model for machine 2 based on lower MSE
    if mse2_lr < mse2_rf:
        best_model2 = model2_lr
        best_model2_name = "Linear Regression"
    else:
        best_model2 = model2_rf
        best_model2_name = "Random Forest"

    # Optionally save models
    if save_model:
        with open(model_file_path, 'wb') as f:
            pickle.dump((best_model1, best_model2), f)

    # Output chosen models and their MSEs
    print(f"Best model for Machine Type 1: {best_model1_name} (MSE: {min(mse1_lr, mse1_rf)})")
    print(f"Best model for Machine Type 2: {best_model2_name} (MSE: {min(mse2_lr, mse2_rf)})")
    
    return best_model1, best_model2

# Function to load models if they are saved
def load_saved_models():
    with open(model_file_path, 'rb') as f:
        return pickle.load(f)

# Power usage functions for machine types 1 and 2
def power_machine_1(model, input_1, input_2, gph):
    inputs = pd.DataFrame([[input_1, input_2, gph]], columns=['input_1', 'input_2', 'input_3'])
    return model.predict(inputs)[0]

def power_machine_2(model, input_1, input_2, gph):
    inputs = pd.DataFrame([[input_1, input_2, gph]], columns=['input_1', 'input_2', 'input_3'])
    return model.predict(inputs)[0]

'''Optimization:

    The optimization process used the models to minimize total power usage, subject to the constraint that the sum of the GPH values for all machines must equal 9,000.
    I  used the Sequential Least Squares Programming (SLSQP) method for efficient optimization.'''


def optimize_power(models, target_gph=9000, num_machines1=10, num_machines2=10, bounds1=(180, 600), bounds2=(300, 1000)):
    model1, model2 = models  # Choose the model pair (could be LR or RF)
    
    # Objective function: total power usage
    def total_power_usage(x):
        total_power = 0
        for i in range(num_machines1):
            total_power += power_machine_1(model1, 25, 6, x[i])
        for i in range(num_machines2):
            total_power += power_machine_2(model2, 25, 6, x[i + num_machines1])
        return total_power
    
    # 
    def gph_constraint(x):
        return np.sum(x) - target_gph
    
    # Bounds for GPH values
    bounds = [(bounds1[0], bounds1[1])] * num_machines1 + [(bounds2[0], bounds2[1])] * num_machines2
    
    # Initial guess: mid-point of the GPH ranges
    initial_guess = [np.mean(bounds1)] * num_machines1 + [np.mean(bounds2)] * num_machines2
    
   
    constraints = {'type': 'eq', 'fun': gph_constraint}
    
    result = minimize(total_power_usage, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP')
    
    return result.x, total_power_usage(result.x)

# Main execution
if __name__ == "__main__":
    # Load datasets
    data1 = pd.read_csv('./machine1.csv')
    data2 = pd.read_csv('./machine2.csv')
    
    # Train or load models(OPTIONAL)
    # if load_model and model_file_path:
    #     model1, model2 = load_saved_models()
    # else:
    model1, model2 = train_models(data1, data2)
    
    
    models = (model1, model2)  
    
    # Run the optimization
    optimal_gph, total_power = optimize_power(models)
    

    '''Optimal GPH Values:

    The solution provided the optimal GPH values for each machine, allowing the factory to produce exactly 9,000 GPH while minimizing power consumption.'''

    print(f"Optimal GPH values for machines: {optimal_gph}")

    '''Total Power Usage:

    The total minimized power usage across all machines was calculated and displayed.'''
    print(f"Total power consumption: {total_power} kW")

'''
    -Key Observations :

    Linear models are easy to interpret and provide quick solutions, but may lack the accuracy of more complex models like Random Forest.
    Efficient optimization techniques like SLSQP can quickly solve high-dimensional problems like minimizing power usage across many machines.
    
    -Possible Next Measures:

    Test with additional machine learning models to explore further improvements in predictive accuracy.
    Explore real-time optimization techniques for dynamic environments.
'''
