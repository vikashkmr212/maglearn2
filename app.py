import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import resample
# If you don't want tqdm progress bars cluttering the interface, comment it out or handle differently
from tqdm import tqdm

##############################################
# Custom Ensemble for Residual Prediction
##############################################
class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, models, n_bootstrap=500):
        self.models = models
        self.n_bootstrap = n_bootstrap
    
    def fit(self, X, y):
        self.models_ = []
        # If you want to show a progress bar in the console, keep tqdm.
        # If you'd like a Streamlit progress bar, you'd handle that differently.
        for _ in tqdm(range(self.n_bootstrap), desc="Bootstrapping"):
            X_bootstrap, y_bootstrap = resample(X, y)
            models_bootstrap = [clone(model).fit(X_bootstrap, y_bootstrap) for model in self.models]
            self.models_.append(models_bootstrap)
        return self
    
    def predict(self, X):
        all_predictions = []
        for models_bootstrap in tqdm(self.models_, desc="Predicting"):
            predictions = np.column_stack([model.predict(X) for model in models_bootstrap])
            mean_prediction = np.mean(predictions, axis=1)
            all_predictions.append(mean_prediction)
        all_predictions = np.array(all_predictions)
        
        mean_prediction = np.mean(all_predictions, axis=0)
        std_deviation = np.std(all_predictions, axis=0)
        
        # 1 sigma envelope
        lower_bound = mean_prediction - std_deviation
        upper_bound = mean_prediction + std_deviation
        
        # Save all bootstrap predictions to a CSV file (optional)
        # bootstrap_df = pd.DataFrame(all_predictions.T, columns=[f'bootstrap_{i+1}' for i in range(self.n_bootstrap)])
        # bootstrap_df.to_csv('bootstrap_predictions.csv', index=False)
        
        return mean_prediction, std_deviation, lower_bound, upper_bound

def main():
    st.title("Custom Ensemble Streamlit App")

    st.sidebar.header("Configuration")

    # 1) Number of Monte Carlo draws (N_MC)
    N_MC = st.sidebar.number_input("Number of Monte Carlo Draws (N_MC):", 
                                   min_value=1, 
                                   value=1000000,
                                   step=1000)
    
    # 2) Number of bootstrap iterations for the ensemble
    n_bootstrap = st.sidebar.number_input("Number of Bootstrap Iterations:", 
                                          min_value=1, 
                                          value=500,
                                          step=10)

    # 3) CSV file uploader
    uploaded_file = st.file_uploader("Upload your CSV file (e.g., core_top_data.csv)", type=["csv"])

    # Button to trigger training
    if uploaded_file is not None:
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                df = pd.read_csv(uploaded_file)

                ##############################################
                # 1) Monte Carlo sampling for parametric T
                ##############################################
                # Known coefficients (mean ± 1σ)
                a_mean, a_std = 0.036, 0.006
                b_mean, b_std = 0.061, 0.005
                c_mean, c_std = -0.73, 0.07
                d_mean, d_std = 0.0, 0.0  # d=0, no stated uncertainty

                # Pre-generate random coefficients
                a_samples = np.random.normal(a_mean, a_std, N_MC)
                b_samples = np.random.normal(b_mean, b_std, N_MC)
                c_samples = np.random.normal(c_mean, c_std, N_MC)
                d_samples = np.random.normal(d_mean, d_std, N_MC)

                param_T_list = []
                param_T_std_list = []

                # For each row, run N_MC draws
                for idx, row in df.iterrows():
                    mgca = row['mgca']
                    S = row['s_annual']
                    pH = row['pH']
                    
                    # T_samples for each of the N_MC draws
                    T_samples = (np.log(mgca) 
                                - a_samples * (S - 35) 
                                - c_samples * (pH - 8)
                                - d_samples) / b_samples
                    
                    param_T_list.append(np.mean(T_samples))
                    param_T_std_list.append(np.std(T_samples))

                # Convert to numpy arrays
                param_T = np.array(param_T_list)
                param_T_uncertainty = np.array(param_T_std_list)

                ##############################################
                # 2) Residual = Actual T - Parametric T
                ##############################################
                target = 't_annual'
                residual = df[target] - param_T

                ##############################################
                # 3) Features for the residual model
                ##############################################
                numerical_features_residual = ['mgca', 'omega_inv_sq']  
                categorical_features_residual = ['species', 'clean_method']

                X = df[numerical_features_residual + categorical_features_residual]
                y = residual

                # Split data
                (X_train, X_test, 
                 y_train_residual, y_test_residual, 
                 param_T_train, param_T_test) = train_test_split(
                     X, y, param_T, test_size=0.3, random_state=42
                )

                # Preprocessor
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numerical_features_residual),
                        ('cat', OrdinalEncoder(), categorical_features_residual)
                    ]
                )

                # Define candidate models
                random_forest = RandomForestRegressor(
                    n_estimators=200,
                    max_features='sqrt',
                    max_depth=20,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42
                )

                gradient_boosting = GradientBoostingRegressor(
                    learning_rate=0.1,
                    max_depth=4,
                    max_features='sqrt',
                    min_samples_split=2,
                    min_samples_leaf=1,
                    n_estimators=200,
                    subsample=0.9,
                    random_state=42
                )

                svr = SVR(
                    C=100,
                    degree=2,
                    epsilon=0.1,
                    gamma='scale',
                    kernel='rbf'
                )

                # Custom ensemble model
                ensemble_model = CustomEnsembleRegressor(
                    models=[random_forest, gradient_boosting, svr], 
                    n_bootstrap=n_bootstrap
                )

                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('ensemble', ensemble_model)
                ])

                # Fit pipeline on the residual
                pipeline.fit(X_train, y_train_residual)

                # Predictions
                y_train_residual_pred, y_train_uncertainty, y_train_lower, y_train_upper = pipeline.predict(X_train)
                y_test_residual_pred, y_test_uncertainty, y_test_lower, y_test_upper = pipeline.predict(X_test)

                ##############################################
                # 4) Final predictions = Parametric T + residual
                ##############################################
                y_train_final = param_T_train + y_train_residual_pred
                y_test_final = param_T_test + y_test_residual_pred

                ##############################################
                # Evaluate performance
                ##############################################
                rmse_train = np.sqrt(mean_squared_error(param_T_train + y_train_residual, y_train_final))
                r2_train = r2_score(param_T_train + y_train_residual, y_train_final)

                rmse_test = np.sqrt(mean_squared_error(param_T_test + y_test_residual, y_test_final))
                r2_test = r2_score(param_T_test + y_test_residual, y_test_final)

                st.success("Training Completed!")
                st.write(f"**Training RMSE:** {rmse_train:.4f}")
                st.write(f"**Training R²:** {r2_train:.4f}")
                st.write(f"**Test RMSE:** {rmse_test:.4f}")
                st.write(f"**Test R²:** {r2_test:.4f}")

                # Optionally, save param_T and param_T_uncertainty
                # df['param_T'] = param_T
                # df['param_T_uncertainty'] = param_T_uncertainty
                # df.to_csv('core_top_data_with_paramT_uncertainty.csv', index=False)

if __name__ == "__main__":
    main()
