import json
import os

save_path = 'cpy_train_results'

results_list = []

for model_name in os.listdir(save_path):
    json_path = os.path.join(save_path, model_name, 'scores.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    f1_score = data['test']['f1']

    results_list.append((model_name, f1_score))

results_list.sort(key=lambda x: x[1])

print(results_list[-3:])
