import json
import wandb

with open('config_local.json') as f:
    data = json.load(f)

api_key = data['wandb_api_key']
wandb.login(key=api_key)
api = wandb.Api()

project = api.project('cx3', entity='tez4')
runs = api.runs("tez4/cx3")

best_val_mse = {}

for run in runs:
    if run.state == 'finished':
        group = run.group
        if group not in best_val_mse:
            best_val_mse[group] = {}

        name = run.name
        run_history = run.history()
        if name not in best_val_mse[group]:
            # run.summary.get("Acc/a_val_acc")
            if 'Image MSE/b_val_image_mse' in run_history:
                best_val_mse[group][name] = run_history['Image MSE/b_val_image_mse'].min()

print(best_val_mse)
