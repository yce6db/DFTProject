from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_FILES = {
    "Cohesive Energy": BASE_DIR / "cohes_comp.csv",
    "Lattice Constant": BASE_DIR / "lat_comp.csv",
    "Bulk Modulus": BASE_DIR / "bulk_comp.csv",
}


def calculate_MAE(experimental: pd.Series, predicted: pd.Series) -> float:
    mae = 0.0
    for row in experimental.index:
        mae += abs(experimental[row] - predicted[row])
    return mae / len(experimental)


def calculate_MARE(experimental: pd.Series, predicted: pd.Series) -> float:
    mare = 0.0
    for row in experimental.index:
        mare += (abs(experimental[row] - predicted[row]) / abs(experimental[row])) * 100
    return mare / len(experimental)


def calculate_MAX(experimental: pd.Series, predicted: pd.Series) -> float:
    max_error = 0.0
    for row in experimental.index:
        error = abs(experimental[row] - predicted[row]) / abs(experimental[row]) * 100
        if error > max_error:
            max_error = error
    return max_error


def calculate_RMSE(experimental: pd.Series, predicted: pd.Series) -> float:
    mse = 0.0
    for row in experimental.index:
        mse += (experimental[row] - predicted[row]) ** 2
    mse /= len(experimental)
    return mse ** 0.5


def analyze_file(file_path: Path, property_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    experiment_col = next((col for col in df.columns if col.lower() == "experiment"), None)
    if experiment_col is None:
        raise ValueError(f"Expected an 'Experiment' column in {file_path}")

    model_columns = [col for col in df.columns if col not in {"Solids", experiment_col}]
    summary_rows = []

    for potential_name in model_columns:
        experimental = pd.to_numeric(df[experiment_col], errors="coerce")
        predicted = pd.to_numeric(df[potential_name], errors="coerce")

        mae = calculate_MAE(experimental, predicted)
        mare = calculate_MARE(experimental, predicted)
        max_error = calculate_MAX(experimental, predicted)
        rmse = calculate_RMSE(experimental, predicted)

        summary_rows.append(
            {
                "Functional": potential_name,
                "Property": property_name,
                "MAE": round(mae, 4),
                "MARE (%)": round(mare, 2),
                "MAX (%)": round(max_error, 2),
                "RMSE": round(rmse, 4),
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    summary_frames = []
    for property_name, file_path in DATA_FILES.items():
        if not file_path.exists():
            print(f"Missing file: {file_path}")
            continue
        summary_frames.append(analyze_file(file_path, property_name))

    if not summary_frames:
        raise SystemExit("No comparison CSV files were found.")

    summary = pd.concat(summary_frames, ignore_index=True)
    summary.to_csv(BASE_DIR / "analysis_summary.csv", index=False, encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nSaved summary to {BASE_DIR / 'functional_analysis_summary.csv'}")


if __name__ == "__main__":
    main()
