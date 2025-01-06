from library import * 
import scanpy as sc
from gears import PertData, GEARS
from gears.inference import evaluate
import pickle
from scgpt.utils import compute_perturbation_metrics

def run_gears(runs=1, mode="train"):
    for run_number in range(0, runs):
        for data_name in ["adam_corrected", "adam_corrected_upr", "adamson", "norman", "replogle_k562_essential"]:
            ##setup PertData object
            if data_name == "adam_corrected":
                pert_data = get_adam_corrected_dataset(split="simulation", batch_size=64, test_batch_size=64, generate_new=False, just_upr=False)
            if data_name == "adam_corrected_upr":
                pert_data = get_adam_corrected_dataset(split="simulation", batch_size=64, test_batch_size=64, generate_new=False, just_upr=True)
            if data_name == "replogle_k562_gwps":
                pert_data = get_replogle_gwps_pert_data(split='simulation', batch_size=64, test_batch_size=64, generate_new=False)
            if data_name in ["adamson", "norman", "replogle_k562_essential"]: 
                # get data
                pert_data = PertData('./data')
                # load dataset in paper: norman, adamson, replogle
                pert_data.load(data_name = data_name)
                # specify data split
                pert_data.prepare_split(split = 'simulation', seed = 1)
                # get dataloader with batch size
                pert_data.get_dataloader(batch_size = 64, test_batch_size = 64)
            if "replogle" in data_name:
                modify_pertdata_dataloaders(pert_data, logger=None)
            gears_model = GEARS(pert_data, device = 'cuda:0')
            if mode == "train":
                # set up and train a model
                gears_model.model_initialize(hidden_size = 64)
                gears_model.train(epochs = 20) #20 originally
                gears_model.save_model(f'gears_models/gears_trained_{data_name}_{run_number}')
            #load model
            gears_model.load_pretrained(f'gears_models/gears_trained_{data_name}_{run_number}')
            ##evaluate
            eval_results = evaluate(loader=pert_data.dataloader['test_loader'], model=gears_model.model, uncertainty=gears_model.config['uncertainty'], device=torch.device("cuda:0"))
            ##get rank score
            ranks = get_gears_rank(eval_results)
            print("avg rank: ", np.mean(list(ranks.values())), np.std(list(ranks.values())))
            pickle.dump(ranks, open(f"pickles/gears_results/gears_rank_metrics_{data_name}_{run_number}.pkl", "wb"))
            ##get pearson scores
            metrics, metrics_pert = compute_metrics(eval_results)
            test_metrics = compute_perturbation_metrics(eval_results, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"])
            print(f"metrics: {metrics}")
            print(f"metrics_pert: {metrics_pert}")
            print(f"test metrics: {test_metrics}")
            pickle.dump((metrics, metrics_pert), open(f"pickles/gears_results/gears_results_{data_name}_{run_number}.pkl", "wb"))
            pickle.dump(test_metrics, open(f"pickles/gears_results/gears_pert_delta_results_{data_name}_{run_number}.pkl", "wb"))

def get_gears_rank(eval_results):
    pert_map = {} ##key: condition, value: (actual avg truth vector, predicted avg vector)
    for condition in set(eval_results["pert_cat"]):
        indices = [i for i in range(0, len(eval_results["pert_cat"])) if eval_results["pert_cat"][i] == condition]
        truth_array = eval_results["truth"][indices, :]
        truth_mean = np.mean(truth_array, axis=0)
        pred_array = eval_results["pred"][indices, :]
        pred_mean = np.mean(pred_array, axis=0)
        pert_map[condition] = (truth_mean, pred_mean)
    ranks = compute_rank(pert_map)
    return ranks

print("running gears")
##mode = "train" for training or "eval" for just evaluating models
run_gears(runs=10, mode="train")

