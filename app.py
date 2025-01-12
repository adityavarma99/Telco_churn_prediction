from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline


application = Flask(__name__)
app = application


class CustomInputData:
    def __init__(self, contract, dependents, device_protection, monthly_charges, online_backup,
                 online_security, paperless_billing, tech_support, tenure):
        self.contract = contract
        self.dependents = dependents
        self.device_protection = device_protection
        self.monthly_charges = monthly_charges
        self.online_backup = online_backup
        self.online_security = online_security
        self.paperless_billing = paperless_billing
        self.tech_support = tech_support
        self.tenure = tenure

    def get_data_as_dataframe(self):
        """Converts input data to a DataFrame."""
        data_dict = {
            "Contract": [self.contract],
            "Dependents": [self.dependents],
            "DeviceProtection": [self.device_protection],
            "MonthlyCharges": [self.monthly_charges],
            "OnlineBackup": [self.online_backup],
            "OnlineSecurity": [self.online_security],
            "PaperlessBilling": [self.paperless_billing],
            "TechSupport": [self.tech_support],
            "tenure": [self.tenure],
        }
        return pd.DataFrame(data_dict)


@app.route('/')
def index():
    """Render the index.html page."""
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Handle predictions based on user input."""
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Gather input data from the form
            input_data = CustomInputData(
                contract=request.form.get('contract'),
                dependents=request.form.get('dependents'),
                device_protection=request.form.get('deviceProtection'),
                monthly_charges=float(request.form.get('monthlyCharges')),
                online_backup=request.form.get('onlineBackup'),
                online_security=request.form.get('onlineSecurity'),
                paperless_billing=request.form.get('paperlessBilling'),
                tech_support=request.form.get('techSupport'),
                tenure=int(request.form.get('tenure'))
            )

            # Convert the input data to a DataFrame
            pred_df = input_data.get_data_as_dataframe()

            # Debugging: Print the DataFrame to console
            print("Input DataFrame:\n", pred_df)

            # Create an instance of the PredictPipeline and predict
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # Render the result on the home page
            return render_template('home.html', results=f"Churn Prediction: {results[0]}")

        except Exception as e:
            # Handle exceptions and display an error message
            return render_template('home.html', results=f"Error: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
