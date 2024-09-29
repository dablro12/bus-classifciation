import sys, os 
sys.path.append('../')
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


from lib.seed import set_seed
from lib.dataset import Custom_stratified_Dataset, Custom_bus_dataset, JointTransform
from lib.datasets.ds_tools import kfold_extract
from model.loader import model_Loader
from lib.metric.metrics import multi_classify_metrics, binary_classify_metrics
from tqdm import tqdm 
def save_dict_to_json(dic, path):
    import json
    print(f"Save to {path}")
    with open(path, 'w') as f:
        json.dump(dic, f, indent=4)
        
if torch.cuda.is_available():
    device = torch.device("cuda")
    
def tester(dataloader, model):
    all_labels, all_preds, all_probs = [], [], []
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            predicted = torch.round(probs)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
    metrics = binary_classify_metrics(all_labels, all_preds, all_probs, test_on = True)
    return metrics


def main(csv_path, fold_num, seed, checkpoint_root_dir, model_name, model_type, epoch, root_dir, outlayer_num, device, save_metric_dir):
    # K-fold extraction
    kfolds, _, _ = kfold_extract(
        csv_path=csv_path,
        n_splits=fold_num,
        random_state=seed,
        shuffle=True
    )

    fold_metrics = {}
    # Iterate through each fold
    for idx, fold in enumerate(tqdm(kfolds, desc="Processing Folds")):
        current_fold_num = idx + 1  # Adjust fold number
        
        # Define checkpoint path
        checkpoint_path = os.path.join(checkpoint_root_dir, model_name + '_' + model_type + '_fold_' + str(current_fold_num), f"model_fold_{current_fold_num}_epoch_{epoch}.pth")

        # Create dataset and data loader for the validation set of the current fold
        test_dataset = Custom_bus_dataset(
            df=fold['val'],
            root_dir=root_dir,
            transform=JointTransform(
                resize=(self.input_res[1], self.input_res[2]),
                horizontal_flip=False,
                vertical_flip=False,
                rotation=0,
                interpolation=False,
            )
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8
        )

        # Load the model for the current fold
        test_model = model_Loader(model_name=model_name, outlayer_num=outlayer_num, type=model_type).to(device)
        test_model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

        # Test the model and collect metrics
        metrics = tester(dataloader=test_loader, model=test_model)
        
        # Store the metrics for the current fold
        fold_metrics[f'fold_{current_fold_num}'] = metrics

    # Save the fold metrics as a JSON file
    save_dict_to_json(fold_metrics, os.path.join(save_metric_dir, model_name + '_' + model_type + '_fold_metrics.json'))