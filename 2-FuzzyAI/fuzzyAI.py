"""
A credit risk assessment application using fuzzy logic.

This script calculates the credit risk level (on a 0-100% scale) based on
financial data provided by the user. A fuzzy inference system
is used for the analysis.

Authors:
   - Daniel Bieliński (s27292)
   - Aleksander Stankowski (s27549)

---------------------------------------------------------------------
Environment Setup:

This script requires Python 3.10 or newer (due to type hinting).
It is recommended to use a virtual environment.

1. Create a virtual environment:
   python -m venv venv
2. Activate the environment:
   source venv/bin/activate  (on macOS/Linux)
   .\\venv\\Scripts\\activate   (on Windows)
3. Install required libraries:
   pip install numpy scikit-fuzzy matplotlib
---------------------------------------------------------------------

The module prompts the user to input the following 5 data points:

   1. Monthly net income (in PLN):
      The basis for creditworthiness assessment and DTI calculation.

   2. Sum of monthly liabilities/payments (in PLN):
      The total of all fixed payments (other loans, alimony, etc.).

   3. Average monthly net savings (in PLN):
      The amount remaining for the user after all expenses;
      an indicator of their "financial cushion".

   4. BIK Scoring (0-100):
      A credit history indicator, where 0 is the lowest score
      and 100 is the highest trustworthiness.

   5. Employment Contract Type:
      A categorical choice (e.g., UOP, JDG, Zlecenie) used to
      determine income stability.

Based on this data, the fuzzy system analyzes five indicators:
   - Monthly Income
   - Monthly Savings
   - BIK Score
   - DTI (Debt-to-Income) ratio, automatically calculated as:
   (Sum of payments / Monthly income) * 100%
   - Income Stability, automatically mapped from the contract type
   to a numerical scale (0-10).

The program's output is a single, quantified (crisp)
percentage value defining the credit risk level.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import sys

def create_credit_system():
   """
   Creates and configures the complete fuzzy logic system
   for credit risk assessment.
    
   Returns:
      ctrl.ControlSystemSimulation: A ready-to-use simulation object.
   """
   
   # --- Universes ---
   x_income = np.arange(0, 25001, 100)
   x_savings = np.arange(0, 7500, 50)
   x_bik = np.arange(0, 101, 1)
   x_dti = np.arange(0, 101, 1)
   x_stability = np.arange(0, 11, 1)
   x_risk = np.arange(0, 101, 1)

   # --- Antecedents ---
   income = ctrl.Antecedent(x_income, 'income')
   savings = ctrl.Antecedent(x_savings, 'savings')
   bik = ctrl.Antecedent(x_bik, 'bik')
   dti = ctrl.Antecedent(x_dti, 'dti')
   stability = ctrl.Antecedent(x_stability, 'stability')

   # 1. Income
   income['low'] = fuzz.trapmf(x_income, [0, 0, 3000, 5000])
   income['medium'] = fuzz.trimf(x_income, [4000, 8000, 12000])
   income['high'] = fuzz.trapmf(x_income, [10000, 15000, 25000, 25000])

   # 2. Savings
   savings['none'] = fuzz.trapmf(x_savings, [0, 0, 200, 500])
   savings['low'] = fuzz.trimf(x_savings, [300, 1000, 2000])
   savings['medium'] = fuzz.trimf(x_savings, [1500, 2000, 3000])
   savings['high'] = fuzz.trapmf(x_savings, [2500, 4000, 7500, 7500])

   # 3. BIK
   bik['low'] = fuzz.trapmf(x_bik, [0, 0, 50, 58])
   bik['moderate'] = fuzz.trimf(x_bik, [50, 63, 68])
   bik['good'] = fuzz.trimf(x_bik, [64, 71, 79])
   bik['excellent'] = fuzz.trapmf(x_bik, [75, 80, 100, 100])

   # 4. DTI
   dti['low'] = fuzz.trapmf(x_dti, [0, 0, 25, 35])
   dti['acceptable'] = fuzz.trimf(x_dti, [30, 40, 50])
   dti['high'] = fuzz.trapmf(x_dti, [45, 55, 100, 100])

   # 5. Stability
   stability['poor'] = fuzz.trapmf(x_stability, [0, 0, 2, 4])     # Zlecenie/Dzieło
   stability['average'] = fuzz.trimf(x_stability, [3, 5, 7])      # JDG
   stability['good'] = fuzz.trapmf(x_stability, [6, 8, 10, 10])   # UOP

   # --- Consequent ---
   risk = ctrl.Consequent(x_risk, 'risk')
   risk['low'] = fuzz.trapmf(x_risk, [0, 0, 15, 25])
   risk['medium'] = fuzz.trimf(x_risk, [20, 35, 50])
   risk['high'] = fuzz.trimf(x_risk, [45, 60, 75])
   risk['very_high'] = fuzz.trapmf(x_risk, [70, 85, 100, 100])

   # --- Rules ---

   # Very high risk
   rule1 = ctrl.Rule(bik['low'], risk['very_high'])
   rule2 = ctrl.Rule(dti['high'] & stability['poor'], risk['very_high'])
   rule3 = ctrl.Rule(stability['poor'] & savings['none'] & income['low'], risk['very_high'])

   # High risk
   rule4 = ctrl.Rule(dti['high'], risk['high'])
   rule5 = ctrl.Rule(stability['poor'] & savings['low'], risk['high'])
   rule6 = ctrl.Rule(bik['moderate'] & dti['acceptable'] & income['low'], risk['high'])
   rule7 = ctrl.Rule(stability['average'] & bik['moderate'], risk['high'])

   # Medium risk
   rule8 = ctrl.Rule(income['high'] & savings['none'] & stability['good'], risk['medium'])
   rule9 = ctrl.Rule(income['medium'] & bik['good'] & dti['acceptable'] & stability['good'], risk['medium'])
   rule10 = ctrl.Rule(stability['average'] & bik['excellent'] & dti['low'], risk['medium'])
   rule11 = ctrl.Rule(bik['moderate'] & dti['low'] & savings['high'], risk['medium'])
   rule12 = ctrl.Rule(income['high'] & savings['medium'] & dti['acceptable'], risk['medium'])

   # Low risk
   rule13 = ctrl.Rule(bik['excellent'] & dti['low'] & stability['good'], risk['low'])
   rule14 = ctrl.Rule(income['high'] & savings['high'] & dti['low'] & bik['good'], risk['low'])
   rule15 = ctrl.Rule(income['low'] & stability['good'] & dti['low'] & bik['excellent'] & savings['medium'], risk['low'])
   
   # More cases in case upper ones dont catch all
   rule16 = ctrl.Rule(bik['excellent'] & stability['good'] & dti['acceptable'] & income['medium'], risk['low'])
   rule17 = ctrl.Rule(bik['moderate'], risk['high'])
   rule18 = ctrl.Rule(bik['good'], risk['medium'])
   rule19 = ctrl.Rule(bik['excellent'], risk['low'])

   all_rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19]

   # --- Control system ---

   credit_system = ctrl.ControlSystem(all_rules)
   risk_simulation = ctrl.ControlSystemSimulation(credit_system)
   
   return risk_simulation

def _get_stability_score() -> float:
   """
   Helper function to map the user's choice to a numerical stability score.
   
   Returns:
       int: Representing user stability score on a scale from 1 to 10.
   """
   
   mapping = {
      '1': 10,  # 'good'
      '2': 8,   # 'good'
      '3': 8,   # 'good'
      '4': 5,   # 'average'
      '5': 4,   # 'average'/'poor'
      '6': 3,   # 'poor'
      '7': 1    # 'poor'
   }
   
   print("\n--- EMPLOYMENT TYPE ---")
   print("1. Employment Contract (UoP) - INDEFINITE period")
   print("2. Employment Contract (UoP) - DEFINITE period (> 12 months remaining)")
   print("3. Self-employment (JDG/B2B) - running > 24 months")
   print("4. Self-employment (JDG/B2B) - running 12-24 months")
   print("5. Employment Contract (UoP) - DEFINITE period (< 12 months remaining)")
   print("6. Mandate/Task Contract (Zlecenie/Dzieło) - regular, for > 12 months")
   print("7. Other (JDG < 12 months, trial period, irregular Mandate contracts)")

   while True:
      choice = input("Your choice (1-7): ")

      if choice in mapping:
         return float(mapping[choice])
      else:
         print("Invalid selection. Please enter a number from 1 to 7.")


def _validate_input(prompt: str, min_val: float | None = None, max_val: float | None = None) -> float:
   """
   Gets a float from the user, validating its type and
   optional range (min/max).
   Loops until the user provides a valid value.
   
   Args:
      prompt (str): Message to display to the user.
      min_val (float | None): Optional minimum allowed value.
      max_val (float | None): Optional maximum allowed value.

   Returns:
      float: The validated user input.
   """
   while True:
      try:
         value_str = input(prompt)
         value_float = float(value_str)
         
         if min_val is not None and value_float < min_val:
            print(f"ERROR: Value must be at least {min_val}.")
         elif max_val is not None and value_float > max_val:
            print(f"ERROR: Value cannot exceed {max_val}.")
         else:
            return value_float
         
      except ValueError:
         print("ERROR: Please enter a valid number")

def get_user_inputs() -> dict:
   """
   Collects all 5 input data points from the user.
   [income, liabilities, savings, bik, stability_score]
   
   Returns:
      dict: A dictionary with the "raw" input data.
   """
   print("--- PLEASE PROVIDE DATA FOR ANALYSIS ---")
   
   inputs = {}
   inputs['income'] = _validate_input("\nMonthly net income (in PLN): ", min_val=0)
   inputs['liabilities'] = _validate_input("\nSum of monthly liabilities/payments (in PLN): ", min_val=0)
   inputs['savings'] = _validate_input("\nAverage monthly net savings (in PLN): ", min_val=0)
   inputs['bik'] = _validate_input("\nBIK Scoring (0-100): ", min_val=0, max_val=100) 
   inputs['stability_score'] = _get_stability_score()
   
   return inputs

def calculate_risk(simulation: ctrl.ControlSystemSimulation, inputs: dict) -> float:
   """
   Calculates the credit risk based on the inputs and the simulation.

   Args:
      simulation (ctrl.ControlSystemSimulation): The fuzzy system simulation.
      inputs (dict): A dictionary of user-provided "raw" data.

   Returns:
      float: The final, crisp risk value (0-100).
   """
   
   if inputs['income'] > 0:
      dti_value = (inputs['liabilities'] / inputs['income']) * 100 
   else:
      dti_value = 100.00
      
   simulation.input['income'] = inputs['income']
   simulation.input['savings'] = inputs['savings']
   simulation.input['bik'] = inputs['bik']
   simulation.input['stability'] = inputs['stability_score']
   simulation.input['dti'] = dti_value
   
   simulation.compute()
   
   return simulation.output['risk']

def translate_interpreter_risk(risk_value: float) -> str:
   """
   Converts the numerical risk value to a human-readable category.

   Args:
      risk_value (float): risk value returned from simulation

   Returns:
      str: Category of risk
   """
   if risk_value <= 30:
      return f"Low risk ({risk_value:.2f}%)"
   elif risk_value <= 55:
      return f"Medium risk ({risk_value:.2f}%)"
   elif risk_value <= 80:
      return f"High risk ({risk_value:.2f}%)"
   else:
      return f"Very high risk ({risk_value:.2f}%)"
   
def main() -> None:
   """
   Main function to run the credit risk assessment.

   Orchestrates the process:
   1. Build the fuzzy system.
   2. Get validated user inputs.
   3. Calculate the risk.
   4. Print the final, interpreted result.
   """
   try:
      print("Building fuzzy logic system...")
      risk_sim = create_credit_system()
      print("System ready.")
      
      user_data = get_user_inputs()
      
      risk = calculate_risk(risk_sim, user_data)
      
      print("\n--- ANALYSIS RESULT ---")
      print(translate_interpreter_risk(risk))

   except ValueError:
      print("\nERROR: Invalid value entered. Please use numbers.", file=sys.stderr)
   except Exception as e:
      print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
   main()