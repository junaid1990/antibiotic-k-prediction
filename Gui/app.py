from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the preprocessing objects and the model
try:
    with open("model/preprocessing.pkl", "rb") as f:
        preprocessing = pickle.load(f)

    le_dope = preprocessing['label_encoder_dope']
    le_gs = preprocessing['label_encoder_gs']
    target_encoder = preprocessing['target_encoder']
    power_transformer_X = preprocessing['power_transformer_X']
    minmax_scaler = preprocessing['minmax_scaler']
    power_transformer_y = preprocessing['power_transformer_y']
    numerical_features = preprocessing['numerical_features']

    with open("model/random_forest_model.pkl", "rb") as f:
        rf_model = pickle.load(f)

except FileNotFoundError as e:
    print(f"Error loading model or preprocessing: {e}")
    exit()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    actual_value = None
    absolute_error = None
    squared_error = None
    percentage_error = None
    raw_input_df = None
    encoded_df = None
    transformed_df = None
    scaled_df = None
    inverse_transformed_prediction = None
    error_message = None

    if request.method == "POST":
        try:
            dope = request.form["Dope"]
            gs = request.form["GS"]
            ssa = float(request.form["SSA"])
            aps = float(request.form["APS"])
            tpv = float(request.form["TPV"])
            idig = float(request.form["IdIg"])
            pt = int(request.form["PT"])  # Ensure integer conversion
            pfrs = request.form["PFRs"]
            if pfrs == "Value":
                _ = request.form.get("PFRs_value")  # ignored, just for UI
                pfrs = 0.559
            elif pfrs == "NaN":
                pfrs = np.nan
            c = float(request.form["C"])
            h = float(request.form["H"])
            o = float(request.form["O"])
            n = float(request.form["N"])
            ph = float(request.form["pH"])
            oxi_c = float(request.form["Oxi_C"])
            bc_c = float(request.form["Bc_C"])
            pol_c = float(request.form["Pol_C"])
            actual_k = request.form.get("actual_k", "").strip()

            # Step 1: Create DataFrame from user input
            input_data = {
                'Dope': [dope], 'GS': [gs], 'SSA': [ssa], 'APS': [aps], 'TPV': [tpv], 'IdIg': [idig], 'PT': [pt],
                'PFRs': [pfrs], 'C': [c], 'H': [h], 'O': [o], 'N': [n], 'pH': [ph], 'Oxi-C': [oxi_c],
                'Bc-C': [bc_c], 'Pol-C': [pol_c]
            }
            raw_input_df = pd.DataFrame(input_data)

            # Step 2: Label Encode 'Dope' and 'GS'
            encoded_df = raw_input_df.copy()
            encoded_df['Dope'] = le_dope.transform(encoded_df['Dope'])
            encoded_df['GS'] = le_gs.transform(encoded_df['GS'])

            # Step 3: Target Encode 'PFRs'
            encoded_df = target_encoder.transform(encoded_df)

            # Step 4: Yeo-Johnson Transform for numerical features
            transformed_df = encoded_df.copy()
            transformed_df[numerical_features] = power_transformer_X.transform(transformed_df[numerical_features])

            # Step 5: MinMax Scaling
            scaled = minmax_scaler.transform(transformed_df)
            scaled_df = pd.DataFrame(scaled, columns=transformed_df.columns)

            # Step 6: Prediction (in transformed space)
            transformed_prediction = rf_model.predict(scaled_df)

            # Step 7: Inverse Transform the prediction
            prediction = power_transformer_y.inverse_transform(transformed_prediction.reshape(-1, 1))[0][0]
            inverse_transformed_prediction = prediction # For display

            # Step 8: Calculate error metrics if actual value is provided
            if actual_k:
                try:
                    actual_value = float(actual_k)
                    absolute_error = abs(prediction - actual_value)
                    squared_error = (prediction - actual_value) ** 2
                    percentage_error = (absolute_error / actual_value) * 100 if actual_value != 0 else None
                except ValueError:
                    actual_value = "Invalid input"

        except ValueError as e:
            error_message = f"Error processing input: {e}"
        except KeyError as e:
            error_message = f"Missing input field: {e}. Please check your form."
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"

    return render_template("index.html",
                           prediction=inverse_transformed_prediction,
                           actual_value=actual_value,
                           absolute_error=absolute_error,
                           squared_error=squared_error,
                           percentage_error=percentage_error,
                           raw_input=raw_input_df,
                           encoded_input=encoded_df,
                           transformed_input=transformed_df,
                           scaled_input=scaled_df,
                           error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)