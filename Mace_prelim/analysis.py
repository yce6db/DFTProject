import pandas as pd
experiment = pd.read_csv("Mace_prelim/exp.csv")
datasets = {"MACE-MP-0": pd.read_csv("Mace_prelim/results_mace_mp0.csv"),
           "MACE-MPA-0": pd.read_csv("Mace_prelim/results_mace_mpa.csv"),
           "MACE-MATPES-r2SCAN-0": pd.read_csv("Mace_prelim/results_mace_matpes.csv"),
           "MACE-OMAT-0": pd.read_csv("Mace_prelim/results_mace_omat.csv")}
properties = {"Lattice Constant (Å)": "Lattice Constant (Å)",
              "Bulk Modulus (GPa)": "Bulk Modulus (GPa)",
              "Cohesive Energy (eV/atom)": "Cohesive Energy (eV/atom)"}
def calculate_MAE(experimental, predicted):
    mae = 0
    for row in experimental.index:
        mae += abs(experimental[row] - predicted[row])
    mae /= len(experimental)
    return mae

def calculate_MARE(experimental, predicted):
    mare = 0
    for row in experimental.index:
        mare += (abs(experimental[row] - predicted[row]) / abs(experimental[row])) * 100
    mare /= len(experimental)
    return mare

def calculate_MAX(experimental, predicted):
    max_error = 0
    for row in experimental.index:
        error = abs(experimental[row] - predicted[row]) /abs(experimental[row]) * 100
        if error > max_error:
            max_error = error
    return max_error

summary_rows = []

for potential_name, data in datasets.items():
    for column_name, property_name in properties.items():

        mae = calculate_MAE(
            experiment[column_name],
            data[column_name]
        )

        mare = calculate_MARE(
            experiment[column_name],
            data[column_name]
        )

        maxare = calculate_MAX(
            experiment[column_name],
            data[column_name]
        )

        summary_rows.append({
            "Potential": potential_name,
            "Property": property_name,
            "MAE": round(mae, 4),
            "MARE (%)": round(mare, 2),
            "MAX (%)": round(maxare, 2),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv("analysis_summary.csv", index=False, encoding="utf-8")
