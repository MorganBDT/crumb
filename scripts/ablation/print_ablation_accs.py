import os
import pandas as pd

def get_avg_accuracy(csv_path):
    """
    Load the CSV file and return the mean of the final column.
    """
    data = pd.read_csv(csv_path, header=None)
    return data.iloc[:, -1].mean()

def main():
    base_dir = "ablation_study"
    conditions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    conditions.sort()

    for condition in conditions:

        if condition == "core50_1_memory_blocks":
            continue

        result_str = f"{condition}: "

        class_iid_path = os.path.join(base_dir, condition, "class_iid")
        class_instance_path = os.path.join(base_dir, condition, "class_instance")

        # Extracting accuracy from class_instance if it exists
        if os.path.exists(class_instance_path):
            network_dir = next(os.walk(class_instance_path))[1][0]  # get the first network directory
            csv_path = os.path.join(class_instance_path, network_dir, "top1_test_all_direct_all_runs.csv")
            result_str += f"class_instance: {get_avg_accuracy(csv_path):.2f}%"

        # Extracting accuracy from class_iid
        network_dir = next(os.walk(class_iid_path))[1][0]  # get the first network directory
        csv_path = os.path.join(class_iid_path, network_dir, "top1_test_all_direct_all_runs.csv")
        result_str += f", class_iid: {get_avg_accuracy(csv_path):.2f}%"

        print(result_str)

if __name__ == "__main__":
    main()