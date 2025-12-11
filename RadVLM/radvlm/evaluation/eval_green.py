import json
from green_score import GREEN
import os 
import argparse

parser = argparse.ArgumentParser(description="A script to evaluate reports with the GREEN metric.")
parser.add_argument('--results_path', type=str, default=None, help="The results path to eval")
args = parser.parse_args()

results_path = args.results_path

# Open and read the JSON file from the 'results' directory
with open(results_path, 'r') as file:
    output = json.load(file)


list_predictions = [item["output"] for item in output]
list_groundtruth = [item["txt"] for item in output]
items = [item for item in output]

#list_predictions = list_predictions[:300]
#list_groundtruth = list_groundtruth[:300]
#items = items[:300]

model_name = "StanfordAIMI/GREEN-radllama2-7b"

green_scorer = GREEN(model_name, output_dir=".")
mean, std, green_score_list, summary, result_df = green_scorer(list_groundtruth, list_predictions)

print(f"{results_path=}")
print(f"{mean}")
print(f"{summary}")

# Prepare output dictionary
results_data = {
    "original_path": results_path,
    "mean": mean,
    "std": std,
    "summary": summary,
    "green_scores": green_score_list,
    "list_predictions": list_predictions,
    "list_groundtruth": list_groundtruth,
    "items": items,
}

# Define output file path in the same directory
output_dir = os.path.dirname(results_path)
base_name = os.path.basename(results_path)

if base_name.endswith("_output.json"):
    output_filename = base_name.replace("_output.json", "_output_green.json")
else:
    output_filename = os.path.splitext(base_name)[0] + "_green.json"

output_path = os.path.join(output_dir, output_filename)

# Save JSON
with open(output_path, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"Saved results to {output_path}")