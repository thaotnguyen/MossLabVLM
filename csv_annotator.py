import pandas as pd
import argparse
import re
from tqdm import tqdm
from ollama import chat

def classify_decision(decision_text):
    prompt = (
        "You are an expert medical image analysis assistant specializing in optic disc evaluation. "
        "Given a description of a disc swelling decision, output exactly one of the following four responses (all lowercase): "
        "'yes' if the disc is swollen (papilledema), 'no' if the disc is normal (not swollen), "
        "'refused to give an answer' if you decline to provide a diagnosis, or "
        "'ambiguous answer' if the description is unclear. "
        "Do not include any additional text, punctuation, or spaces. "
        f"Description: \"{decision_text}\""
    )
    try:
        response = chat(model='gemma2:27b', messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        output = response['message']['content']
        valid_responses = ["yes", "no", "refused to give an answer", "ambiguous answer"]
        if output in valid_responses:
            if output == "yes":
                return 1
            elif output == "no":
                return 0
            else:
                return output
        else:
            # Heuristic fallback if the output isn't an exact match:
            if "yes" in output:
                return 1
            elif "no" in output:
                return 0
            elif "refuse" in output:
                return "refused"
            elif "ambiguous" in output or "unclear" in output or "unable" in output:
                return "ambiguous"
            else:
                print(f"Warning: Unrecognized output: '{output}'. Defaulting to 'ambiguous'.")
                return "ambiguous"
    except Exception as e:
        print(f"Error calling Ollama API: {e}. Defaulting to 'ambiguous'.")
        return "ambiguous"

def main():
    parser = argparse.ArgumentParser(
        description="Process a VLM CSV file, optionally compute 'True' and 'Prediction' columns, and calculate sensitivity/specificity."
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("--recompute", "-r", action="store_true", 
                        help="Force recomputation of 'True' and 'Prediction' values even if they already exist")
 
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    df["True"] = None
    cols = df.drop(columns=['Image Name', 'Caption', 'True'], errors='ignore').columns
    cols = [col for col in cols if not col.startswith("Prediction")]

    columns_exist = ("True" in df.columns) and all([f"Prediction_{col}" in df.columns for col in cols])

    if not columns_exist or args.recompute:
        for col in cols:
            df[f"Prediction_{col}"] = None
        for idx, row in tqdm(df.iterrows()):
            image_name = row["Image Name"].split('/')[-1]
            # Only process rows with expected filename prefixes.
            if image_name.startswith("Normal_") or image_name.startswith("Papilledema_"):
                true_val = 0 if image_name.startswith("Normal_") else 1
                df.at[idx, "True"] = true_val
                for col in cols:
                    decision_text = row[col]
                    prediction_val = classify_decision(decision_text)
                    df.at[idx, f"Prediction_{col}"] = prediction_val

        # Save the updated CSV with recomputed columns.
        df.to_csv(args.csv_file, index=False)
        print("Recomputed 'True' and 'Prediction' columns and updated CSV file.")

    # Whether recomputed or not, calculate sensitivity and specificity using rows with valid predictions.
    for col in cols:
        print(f"\nResults for '{col}':")
        valid_df = df[df["True"].notnull() & df[f"Prediction_{col}"].notnull()].copy()
        total_valid = len(valid_df)

        # Count rates for refusal and ambiguous answers.
        refusal_count = valid_df[valid_df[f"Prediction_{col}"] == "refused"].shape[0]
        ambiguous_count = valid_df[valid_df[f"Prediction_{col}"] == "ambiguous"].shape[0]
        refusal_rate = refusal_count / total_valid if total_valid > 0 else 0
        ambiguous_rate = ambiguous_count / total_valid if total_valid > 0 else 0

        # Convert "True" to integer.
        valid_df["True"] = valid_df["True"].astype(int)
        # Keep only rows where Prediction is an integer (i.e. 0 or 1) and not "unable to assess"
        valid_df = valid_df[valid_df[f"Prediction_{col}"].apply(lambda x: isinstance(x, int))]
        valid_df[f"Prediction_{col}"] = valid_df[f"Prediction_{col}"].astype(int)

        TP = ((valid_df["True"] == 1) & (valid_df[f"Prediction_{col}"] == 1)).sum()
        FN = ((valid_df["True"] == 1) & (valid_df[f"Prediction_{col}"] == 0)).sum()
        TN = ((valid_df["True"] == 0) & (valid_df[f"Prediction_{col}"] == 0)).sum()
        FP = ((valid_df["True"] == 0) & (valid_df[f"Prediction_{col}"] == 1)).sum()

        print(f"TP: {TP}, FN: {FN}, TN: {TN}, FP: {FP}")
        print(f"Accuracy: {valid_df['True'].eq(valid_df[f'Prediction_{col}']).mean():.3f}")
        print(f"Precision: {TP / (TP + FP) if (TP + FP) > 0 else 0:.3f}")
        print(f"F1: {2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0:.3f}")
        print(f"Sensitivity: {TP / (TP + FN) if (TP + FN) > 0 else 0:.3f}")
        print(f"Specificity: {TN / (TN + FP) if (TN + FP) > 0 else 0:.3f}")
        print(f"Refusal Rate: {refusal_rate:.3f} ({refusal_count}/{total_valid})")
        print(f"Ambiguous Answer Rate: {ambiguous_rate:.3f} ({ambiguous_count}/{total_valid})")


if __name__ == "__main__":
    main()
