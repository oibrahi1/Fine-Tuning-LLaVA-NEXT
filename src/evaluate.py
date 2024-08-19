import os
import json
import subprocess
from GREEN.green_score.green import compute

def clone_green_repo():
    repo_url = "https://github.com/Stanford-AIMI/GREEN.git"
    repo_dir = "GREEN"
    
    if not os.path.exists(repo_dir):
        print(f"Cloning GREEN repository from {repo_url}...")
        subprocess.run(["git", "clone", repo_url])
    else:
        print("GREEN repository already exists.")

def load_data(generated_path, ground_truth_path):
    with open(generated_path, 'r') as gen_infile:
        generated_outputs = json.load(gen_infile)

    with open(ground_truth_path, 'r') as gt_infile:
        ground_truths = json.load(gt_infile)
    
    return generated_outputs, ground_truths

def extract_section(data, section_tag):
    extracted = []
    for report in data:
        if isinstance(report, dict):
            section_text = report.get(section_tag, '')
        else:
            start_tag = f"<s_{section_tag}>"
            end_tag = f"</s_{section_tag}>"
            start_idx = report.find(start_tag) + len(start_tag)
            end_idx = report.find(end_tag)
            section_text = report[start_idx:end_idx].strip()
        extracted.append(section_text)
    return extracted

def compute_green_scores(generated_outputs, ground_truths, model_name="StanfordAIMI/GREEN-radllama2-7b"):
    sections = ["mediastinal", "lung", "heart", "bone"]
    results = {}

    for section in sections:
        print(f"Processing {section} section...")
        generated_section = extract_section(generated_outputs, section)
        ground_truth_section = extract_section(ground_truths, section)
        
        # Compute the GREEN score for the section
        os.makedirs(f"results/{section}", exist_ok=True)
        green_scores = compute(model_name, ground_truth_section, generated_section, output_dir=f"results/{section}")
        results[section] = green_scores
    
    return results

def save_results(results, output_path):
    with open(output_path, 'w') as outfile:
        json.dump(results, outfile, indent=4)
    print(f"Results have been saved to {output_path}")

def main():
    # Clone the GREEN repo if not already present
    # clone_green_repo()

    # Paths to the generated and ground truth reports
    generated_path = "results/generated_reports_test.json"
    ground_truth_path = "results/ground_truths_test.json"
    output_path = "results/green_scores.json"

    # Load data
    generated_outputs, ground_truths = load_data(generated_path, ground_truth_path)

    # Compute GREEN scores
    results = compute_green_scores(generated_outputs, ground_truths)

    # Save the results
    # save_results(results, output_path)

if __name__ == "__main__":
    main()
