"""
Main script for generating figures and doing one-off analyses of files generated from runner.py
"""
from library import * 
import matplotlib.pyplot as plt
import shutil
from matplotlib_venn import venn3
import venn
from scgpt.utils import get_gene_names
import re
import itertools
import glob
import sklearn.metrics as sklm
import seaborn as sns

# datasets = ["adam_corrected", "adam_corrected_upr", "adamson", "norman", "replogle_k562_essential"]
datasets = ["adam_corrected_upr",  "norman", "replogle_k562_essential"]

def get_pert_data(data_name):
    if data_name in ["adamson", "norman", "replogle_k562_essential"]: 
        pert_data = PertData("./data")
        pert_data.load(data_name=data_name)
        pert_data.prepare_split(split="simulation", seed=1)
        pert_data.get_dataloader(batch_size=64, test_batch_size=64)
    if data_name == "adam_corrected":
        pert_data = get_adam_corrected_dataset(split="simulation", batch_size=64, test_batch_size=64, generate_new=False, just_upr=False)
    if data_name == "adam_corrected_upr":
        pert_data = get_adam_corrected_dataset(split="simulation", batch_size=64, test_batch_size=64, generate_new=False, just_upr=True)
    if data_name == "replogle_k562_gwps":
        pert_data = get_replogle_gwps_pert_data(split="simulation", batch_size=64, test_batch_size=64, generate_new=False)
    if "replogle" in data_name:
        modify_pertdata_anndata(pert_data)
        modify_pertdata_dataloaders(pert_data, logger=None)
    return pert_data

def get_dataset(string):
    mapp = {"adamson": "adamson", "norman": "norman", "replogle":  "replogle_k562_essential", "combined": "combined", "telohaec": "telohaec", "adam_corrected": "adam_corrected", "adam_corrected_upr": "adam_corrected_upr"}
    if  "adam_corrected_upr" in string: #adam_corrected is a substring of adam_corrected_upr, deal with this case first 
        return "adam_corrected_upr"
    if "adam_corrected" in string:
        return "adam_corrected"
    for key_phrase in mapp:
        if key_phrase in string:
            return mapp[key_phrase]

def get_avg_baseline(mode):
    ##key: method, key: dataset, key: score metric, value: list of scores
    baseline_map = {"scGPT": {}, "gears": {}, "mean_control": {},
    "mean_perturbed": {},"smart_mean_control": {}, "smart_mean_perturbed": {}, 
    "mean_control+perturbed": {}, "smart_mean_control+perturbed":{}}
    for method in baseline_map: 
        if method == "scGPT":
            baseline_root = "save/default_config_baseline/"
            for directory in os.listdir(baseline_root):
                baseline_dir = os.path.join(baseline_root, directory)
                dataset = get_dataset(baseline_dir)
                if dataset not in baseline_map["scGPT"]: 
                    baseline_map["scGPT"][dataset] = {}
                if mode == 1:  
                    scGPT_baseline = pickle.load(open(os.path.join(baseline_dir, f"scGPT_pert_delta_results_{dataset}.pkl"), "rb"))
                else:
                    scGPT_baseline, _ = pickle.load(open(os.path.join(baseline_dir, f"scGPT_results_{dataset}.pkl"), "rb"))
                if len(baseline_map["scGPT"][dataset]) == 0: ##instantiate
                    for key in scGPT_baseline: 
                        baseline_map["scGPT"][dataset][key] = [scGPT_baseline[key]]
                else:
                    for key in scGPT_baseline: 
                        baseline_map["scGPT"][dataset][key].append(scGPT_baseline[key])
        if method == "gears":
            baseline_dir = "pickles/gears_results"
            for file in os.listdir(baseline_dir):
                dataset = get_dataset(file)
                if dataset not in baseline_map["gears"]:  ##instantiate
                    baseline_map["gears"][dataset] = {}
                if mode == 1 and "gears_pert_delta_results_" in file:
                    gears_baseline = pickle.load(open(os.path.join(baseline_dir, file), "rb"))
                elif mode == 2 and "gears_results_" in file:
                    gears_baseline, _ = pickle.load(open(os.path.join(baseline_dir, file), "rb"))
                else: 
                    continue
                if len(baseline_map["gears"][dataset]) == 0:
                    for key in gears_baseline: 
                        baseline_map["gears"][dataset][key] = [gears_baseline[key]]
                else:
                    for key in gears_baseline: 
                        baseline_map["gears"][dataset][key].append(gears_baseline[key])
        if "mean" in method:
            baseline_root = "save/eval_human_cp_foundation/"
            for directory in os.listdir(baseline_root):
                dataset = get_dataset(directory)
                for mean_baseline in ["smart", "baseline"]:
                    prefix = "smart_" if mean_baseline == "smart" else ""
                    for mean_type in ["perturbed", "control+perturbed", "control"]:
                        if mode == 1: 
                            mean_map = pickle.load(open(os.path.join(baseline_root, directory, f"{mean_baseline}_mean_{mean_type}_pert_delta_results_{dataset}.pkl"), "rb"))
                        else:
                            mean_map = pickle.load(open(os.path.join(baseline_root, directory, f"{mean_baseline}_mean_{mean_type}_results_{dataset}.pkl"), "rb"))
                        if dataset in baseline_map[f"{prefix}mean_{mean_type}"]:
                            assert(mean_map == baseline_map[f"{prefix}mean_{mean_type}"][dataset])
                        else:
                            baseline_map[f"{prefix}mean_{mean_type}"][dataset] = mean_map
    ##save unreduced
    pickle.dump(baseline_map, open(f"pickles/unreduced_baseline_default_mode={mode}.pkl", "wb"))
    ##reduce 
    for method in baseline_map:
        for dataset in baseline_map[method]:
            for metric in baseline_map[method][dataset]:
                baseline_map[method][dataset][metric] = np.mean(baseline_map[method][dataset][metric]), np.std(baseline_map[method][dataset][metric])
    pickle.dump(baseline_map, open(f"pickles/baseline_default_mode={mode}.pkl", "wb"))

def get_no_pretraining_results(mode, x_labels):
    """
    get scGPT no pretraining results avg and std
    """
    paths = [] 
    for root, dirs, files in os.walk("save/no_pretraining/"):
        for file in files:
            model_type = "scGPT"
            if mode == 1 and f"scGPT_pert_delta_results" in file:
                paths.append(os.path.join(root, file))
            if mode == 2 and f"scGPT_results" in file:
                paths.append(os.path.join(root, file))
    adam_corrected_multirun_results = get_path_results("save/no_pretraining/adam_corrected_run_1/scGPT_pert_delta_results_adam_corrected.pkl", paths, x_labels, mode)
    adam_corrected_upr_multirun_results = get_path_results("save/no_pretraining/adam_corrected_upr_run_1/scGPT_pert_delta_results_adam_corrected_upr.pkl", paths, x_labels, mode)
    adamson_multirun_results = get_path_results("save/no_pretraining/adamson_run_1/scGPT_pert_delta_results_adamson.pkl", paths, x_labels, mode)
    norman_multirun_results = get_path_results("save/no_pretraining/norman_run_1/scGPT_pert_delta_results_norman.pkl", paths, x_labels, mode)
    replogle_multirun_results = get_path_results("save/no_pretraining/replogle_k562_essential_run_1/scGPT_pert_delta_results_replogle_k562_essential.pkl", paths, x_labels, mode)
    return_map = {"adam_corrected": adam_corrected_multirun_results, "adam_corrected_upr": adam_corrected_upr_multirun_results, "adamson": adamson_multirun_results, "norman": norman_multirun_results, "replogle_k562_essential": replogle_multirun_results}
    return return_map

def get_model_title_from_path(string):
    if "no_pretraining" in string:
        return "scGPT (no-pretraining)"
    if "transformer_encoder_control" in string:
        return "scGPT (randomly initialized transformer encoder)"
    if "input_encoder_control" in string:
        return "scGPT (randomly initialized input encoder)"
    if "simple_affine" in string:
        if "simple_affine_large" not in string:
            return "Simple Affine (no transformer)"
        else:
            return "Simple Affine (replace transformer with MLP)"
    if "LoRa" in string:
        return "LoRa Fine-Tuned scGPT"
    return "Modified scGPT"

def get_model_type(string, formal=False):
    if "scGPT" in string or "scgpt" in string:
        return "scGPT" if formal else "scgpt"
    if "gears" in string or "GEARS" in string:
        return "GEARS" if formal else "gears"
    if "simple_affine" in string:
        return "Simple Affine" if formal else "simple_affine"
    if "linear_additive" in string:
        return "Linear Additive" if formal else "linear_additive"
    if "latent_additive" in string:
        return "Latent Additive" if formal else "latent_additive"
    if "decoder_only" in string:
        return "Decoder Only" if formal else "decoder_only"
    if "smart_mean" in string:
        return "CRISPR-informed Mean" if formal else "smart_mean"
    if "baseline_mean" in string:
        return "Training Mean" if formal else "baseline_mean"
    raise Exception(f"model for {string} not found")
    
def get_path_results(path, paths, x_labels, mode, return_unreduced=False):
    """
    For a given path to results file:
    will return the mean score (list - one entry for each of x_labels), std (list), and also original unreduced values (dict)
        if part of a mult-run: score will be the average across all runs
        if part of a singleton run: score will be the result of that one run, std will be 0 
    """
    ##get y_model and (if part of a multi-run experiment) y_model_std 
    if "_run_" in path: ##if this file path is part of a multi-set run, then find the other files and aggregate them into avg and std scores
        run_number = re.findall(r"run_[0-9]+", path)[0]
        stripped_path = path.replace(run_number, "")
        same_run_paths = [p for p in paths if "run_" in p and p.replace(re.findall(r"run_[0-9]+", p)[0], "") == stripped_path]
        if len(same_run_paths) != 10:
            print(f"WARNING: path: {path}, len(same_run_paths) = {len(same_run_paths)}")
        for index, srp in enumerate(same_run_paths):
            if mode == 1: 
                model_res = pickle.load(open(srp, "rb"))
            else:
                model_res, _ = pickle.load(open(srp, "rb"))
            if index == 0: ##instantiate for first one
                sr_results = {key: [value] for key, value in model_res.items()} ##turn single value into list
            else: 
                for key in model_res: 
                    sr_results[key].append(model_res[key])
        ##avg reduce sr_results
        reduced_sr_results = {key: (np.mean(sr_results[key]), np.std(sr_results[key])) for key in sr_results}
        y_model = [reduced_sr_results[key][0] for key in x_labels]
        y_model_std = [reduced_sr_results[key][1] for key in x_labels]
    else: ##singleton run               
        if mode == 1: 
            model_res = pickle.load(open(path, "rb"))
        else:
            model_res, _ = pickle.load(open(path, "rb"))
        y_model = [model_res[key] for key in x_labels]
        y_model_std = [0] * len(y_model)
    if return_unreduced:
        return y_model, y_model_std, sr_results
    else: 
        return y_model, y_model_std

def get_baseline_dataset_map(mode):
    """
    Return dictionary with key: dataset, key: model, value: results dictionary
    """
    baseline_map = pickle.load(open(f"pickles/baseline_default_mode={mode}.pkl", "rb"))
    dataset_map = {} ##key: dataset, key: model, value: results dictionary
    model_keys = baseline_map.keys()
    for dataset in datasets:
        dataset_map[dataset] = {key: baseline_map[key][dataset] for key in model_keys}
    return dataset_map 

def get_baseline_y_std_map(dataset_map, dataset, x_labels):
    """
    Return dictionary with key: model, value: tuple(scores list, std list) corresponding to x_labels
    for dataset
    """
    model_perf_map = dataset_map[dataset] # key: model, value: results dictionary
    ##key: model, value: tuple(scores list, std list) corresponding to x_labels
    baseline_y_std_map = {model: ([model_perf_map[model][x_label][0] for x_label in x_labels], [model_perf_map[model][x_label][1] for x_label in x_labels]) for model in model_perf_map} 
    return baseline_y_std_map

def plot_model_scores(mode):
    """
    Plots each save directory results, mode == 1 will plot the metrics from scGPT paper, mode == 2 will plot original GEARS metrics 
    """
    dataset_map = get_baseline_dataset_map(mode)
    ##find all paths to scGPT (or simple affine) result files within save, check if part of a multi-run 
    paths = [] 
    for root, dirs, files in os.walk("save/"):
        if "archive" in root:
            continue
        for file in files:
            if "baseline" in file or "smart_mean" in file or "result" not in file: ##skip baseline file or non-result files
                continue 
            model_type = get_model_type(os.path.join(root, file))
            if model_type not in ["scgpt", "simple_affine"]:
                continue
            if mode == 1 and f"pert_delta_results" in file:
                paths.append(os.path.join(root, file))
            if mode == 2 and f"_results" in file and "pert_delta" not in file:
                paths.append(os.path.join(root, file))
    print(paths)
    ##iterate over file paths and plot the results
    for path in paths: 
        ##get dataset that results are for 
        dataset = get_dataset(path)
        if dataset not in dataset_map.keys():
            continue
        model_type = get_model_title_from_path(path)
        if dataset == "combined": 
            continue
        ##get baseline data to plot
        x_labels = dataset_map[dataset]["gears"].keys()
        baseline_y_std_map = get_baseline_y_std_map(dataset_map, dataset, x_labels)
        ##get results for this path to plot
        y_model, y_model_std = get_path_results(path, paths, x_labels, mode)
        ##plot 
        fig, ax = plt.subplots()
        x = np.array(range(0, len(x_labels)))
        ax.set_xticks(x)
        if "default_config_baseline" in path: #plot the baselines including different types of mean
            anchor = 1.35
            y_dict = {"GEARS": (baseline_y_std_map["gears"][0], baseline_y_std_map["gears"][1], "#3B75AF"), 
                    "Mean": (baseline_y_std_map["mean_perturbed"][0], baseline_y_std_map["mean_perturbed"][1], "salmon"),
                    "CRISPR-informed Mean": (baseline_y_std_map["smart_mean_perturbed"][0], baseline_y_std_map["smart_mean_perturbed"][1] , "goldenrod"),
                    "scGPT Fully Fine-Tuned Baseline": (baseline_y_std_map["scGPT"][0], baseline_y_std_map["scGPT"][1], "#519E3E")}
        else:
            anchor = 1.37
            if "simple_affine" not in path: ##plot scGPT variant
                y_dict = {"GEARS": (baseline_y_std_map["gears"][0], baseline_y_std_map["gears"][1], "#3B75AF"), 
                        "Mean": (baseline_y_std_map["mean_perturbed"][0], baseline_y_std_map["mean_perturbed"][1], "salmon"),
                        "CRISPR-informed Mean": (baseline_y_std_map["smart_mean_perturbed"][0], baseline_y_std_map["smart_mean_perturbed"][1] , "goldenrod"),
                        "scGPT Fully Fine-Tuned Baseline": (baseline_y_std_map["scGPT"][0], baseline_y_std_map["scGPT"][1], "#519E3E"),
                        model_type: (y_model, y_model_std, "mediumpurple")}
            else: ##for simple affine, we want to compare with the results of no pre-training scGPT for a fair comparison 
                scGPT_no_pretraining_avg, scGPT_no_pretraining_std = get_no_pretraining_results(mode, x_labels)[dataset]
                y_dict = {"GEARS": (baseline_y_std_map["gears"][0], baseline_y_std_map["gears"][1], "#3B75AF"), 
                        "Mean": (baseline_y_std_map["mean_perturbed"][0], baseline_y_std_map["mean_perturbed"][1], "salmon"),
                        "CRISPR-informed Mean": (baseline_y_std_map["smart_mean_perturbed"][0], baseline_y_std_map["smart_mean_perturbed"][1] , "goldenrod"),
                        "scGPT (no-pretraining)": (scGPT_no_pretraining_avg, scGPT_no_pretraining_std, "mediumpurple"),
                        model_type: (y_model, y_model_std, "grey")}
        ##make the plot and annotate it
        width = 0.15
        for method in y_dict:
            ax.bar(x, y_dict[method][0], yerr=y_dict[method][1],  width=width, error_kw={"elinewidth":0.5, "capsize":0.5}, label=method, color=y_dict[method][2])
            for i,j in zip(x, y_dict[method][0]):
                ax.annotate(f"{round(j, 2)}", xy=(i - .06, j +.02),fontsize=5)
            x = x + width
        ax.hlines(y=0.0, xmin=0, xmax=len(x_labels), linestyles="dashed", color="black", linewidth=0.5)
        plt.title(f"Model Evaluations for {get_dataset_title(dataset)} Dataset")
        x_label_map = {"pearson": "Pearson", "pearson_de": "Pearson DE", "pearson_delta": "Pearson Delta", "pearson_de_delta": "Pearson DE Delta"}
        plt_x_labels = [x_label_map[x] for x in x_labels]
        ax.set_xticklabels(plt_x_labels, fontsize=8.5)
        plt.xticks(rotation=15)
        ax.set_ylabel("Pearson Score")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper right', prop={"size":7}, bbox_to_anchor=(1, anchor))
        plt.gcf().subplots_adjust(top=.76)
        if "_run_" in path: ##just save one plot for the multi-runs
            prefix = path[0: path.find("_run_")]
            plt.savefig(f"outputs/multirun_{prefix.replace('/', '_')}_mode={mode}.png", dpi=300)
        else:
            plt.savefig(f"outputs/{path.replace('/', '_')}_mode={mode}.png", dpi=300)

def plot_subset_model_scores(mode, include_simple_affine=False):
    """
    Plot bar graphs for just a subset of the results of interest, namely the different weight loading schemes
    """
    dataset_map = get_baseline_dataset_map(mode)
    if include_simple_affine:
        root_dirs = ["save/no_pretraining", "save/transformer_encoder_control", "save/input_encoder_control", "save/simple_affine"]
        permitted_models = ["scgpt", "simple_affine"]
    else:
        root_dirs = ["save/no_pretraining", "save/transformer_encoder_control", "save/input_encoder_control"]
        permitted_models = ["scgpt"]
    paths = []
    for root_dir in root_dirs: 
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if "results" not in file:
                    continue
                # model_type = get_model_type(os.path.join(root, file))
                model_type = get_model_type(file)
                if model_type not in permitted_models:
                    continue 
                if mode == 1 and "pert_delta_results" in file:
                    paths.append(os.path.join(root, file))
                if mode == 2 and "_results" in file and "pert_delta" not in file:
                    paths.append(os.path.join(root, file))
    unreduced_map = {perm: {dataset: "" for dataset in datasets} for perm in root_dirs} ##key: permutation, key: dataset, value: dictionary of metric key to list of scores, e.g. {save/no_pretraining: norman: {pearson: , pearson_de, ...}}
    
    perm_map = {perm: {dataset: "" for dataset in datasets} for perm in root_dirs} ##key: permutation as directory path, key: dataset, value: (y_model, y_std)
    baseline_map = {dataset: "" for dataset in datasets} #key dataset: value: baseline_y_std_map (key: model, value: tuple(avg scores list, std list) corresponding to x_labels)
    ##iterate over file paths and fill out perm_map
    for path in paths: 
        ##get dataset that results are for 
        dataset = get_dataset(path)
        if dataset not in dataset_map.keys():
            continue
        ##get baseline data to plot
        x_labels = list(dataset_map[dataset]["gears"].keys())
        baseline_y_std_map = get_baseline_y_std_map(dataset_map, dataset, x_labels)
        baseline_map[dataset] = baseline_y_std_map
        ##get results for this path to plot
        y_model, y_model_std, unreduced = get_path_results(path, paths, x_labels, mode, return_unreduced=True)
        for key in perm_map:
            if key in path: 
                assigned_key = key
                break
        
        if dataset in ["adam_corrected_upr", "norman", "replogle_k562_essential"]: ##only include the 3 core datasets in this calculation for statistical significance (for main results)
            if unreduced_map[assigned_key][dataset] == "": 
                unreduced_map[assigned_key][dataset] = unreduced

        if perm_map[assigned_key][dataset] != "":
            assert(perm_map[assigned_key][dataset] == (y_model, y_model_std)) ##this should be the same for paths of the same multi-run
        else:
            perm_map[assigned_key][dataset] = (y_model, y_model_std)
    ##statistical t-test
    get_p_val_comparisons(unreduced_map, x_labels)
    ##print model to model comparisons 
    # compare_models_across_datasets(perm_map, baseline_map, x_labels)
    ##plot 
    for dataset in baseline_map.keys():
        fig, ax = plt.subplots()
        x = np.array(range(0, len(x_labels)))
        ax.set_xticks(x)
        anchor = 1.37
        y_dict = {"GEARS": (baseline_map[dataset]["gears"][0], baseline_map[dataset]["gears"][1], "#3B75AF"), 
                "Mean": (baseline_map[dataset]["mean_perturbed"][0], baseline_map[dataset]["mean_perturbed"][1], "salmon"),
                "CRISPR-informed Mean": (baseline_map[dataset]["smart_mean_perturbed"][0], baseline_map[dataset]["smart_mean_perturbed"][1] , "goldenrod"),
                "scGPT Fully Fine-Tuned Baseline": (baseline_map[dataset]["scGPT"][0], baseline_map[dataset]["scGPT"][1], "#519E3E"),
                "scGPT (no pre-training)": (perm_map["save/no_pretraining"][dataset][0], perm_map["save/no_pretraining"][dataset][1], "mediumpurple"),
                "scGPT (randomly initialized input encoder) ": (perm_map["save/input_encoder_control"][dataset][0], perm_map["save/input_encoder_control"][dataset][1], "purple"),
                "scGPT (randomly initialized transformer encoder) ": (perm_map["save/transformer_encoder_control"][dataset][0], perm_map["save/transformer_encoder_control"][dataset][1], "darkviolet")
                }
        if include_simple_affine:
            y_dict.update({"Simple Affine": (perm_map["save/simple_affine"][dataset][0], perm_map["save/simple_affine"][dataset][1], "grey") })
        ##make the plot and annotate it
        width = 0.12
        for method in y_dict:
            ax.bar(x, y_dict[method][0], yerr=y_dict[method][1],  width=width, error_kw={"elinewidth":0.5, "capsize":0.5}, label=method, color=y_dict[method][2])
            for i,j in zip(x, y_dict[method][0]):
                ax.annotate(f"{round(j, 2)}", xy=(i - .06, j +.02),fontsize=4.5)
            x = x + width
        ax.hlines(y=0.0, xmin=0, xmax=len(x_labels), linestyles="dashed", color="black", linewidth=0.5)
        plt.title(f"Model Evaluations for {get_dataset_title(dataset)} Dataset", fontsize=10)
        x_label_map = {"pearson": "Pearson", "pearson_de": "Pearson DE", "pearson_delta": "Pearson Delta", "pearson_de_delta": "Pearson DE Delta"}
        plt_x_labels = [x_label_map[x] for x in x_labels]
        ax.set_xticklabels(plt_x_labels, fontsize=8.5)
        plt.xticks(rotation=15)
        ax.set_ylabel("Pearson Score")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper right', prop={"size":6}, bbox_to_anchor=(1, anchor))
        plt.gcf().subplots_adjust(top=.76)
        plt.savefig(f"outputs/aggregated_results_{dataset}_mode={mode}.png", dpi=300)

def get_p_val_comparisons(unreduced_map, x_labels):
    """
    Helper function for plot_subset_model_scores
    will print out pairwise p-values for t-test 
    """
    ##consolidate map
    consolidated = {key: {x_label: [] for x_label in x_labels} for key in unreduced_map} ##key: perm, value: {x_label: list of values}
    for perm in unreduced_map: 
        for dataset in unreduced_map[perm]:
            for key in unreduced_map[perm][dataset]:
                consolidated[perm][key] = consolidated[perm][key] + unreduced_map[perm][dataset][key]

    ##get baseline and add it to consolidated 
    unreduced_baseline = pickle.load(open("pickles/unreduced_baseline_default_mode=1.pkl", "rb"))
    for base_method in ["scGPT", "gears", "mean_perturbed", "smart_mean_perturbed"]:
        b_m = {x_label: [] for x_label in x_labels} ##key: x_label, value: list of scores across datasets
        ##consolidate across datasets for baseline, and organize to match consolidated
        for dataset in unreduced_baseline[base_method]:
            for x_label in unreduced_baseline[base_method][dataset]:
                if dataset in ["adam_corrected_upr", "norman", "replogle_k562_essential"]: ##only include the 3 core datasets in this calculation for statistical significance (for main results)
                    if base_method in ["scGPT", "gears"]:
                        b_m[x_label] = b_m[x_label] + unreduced_baseline[base_method][dataset][x_label]
                    else: ##mean baselines are differently formatted ({mean method: {dataset: {metric: score,... } }) because there is no stochasticity / idea of an independently trained model, 
                        b_m[x_label] = b_m[x_label] + [unreduced_baseline[base_method][dataset][x_label]] * 10 ##need to account for the 10 comparisons that were made (even if same result)
        ##add baseline results to consolidated
        consolidated[base_method] = b_m
    # print(consolidated)
    pairs = list(itertools.combinations(list(consolidated.keys()), 2))
    for p1, p2 in pairs:
        for x_label in x_labels: 
            t_statistic, p_value = scipy.stats.ttest_ind(consolidated[p1][x_label], consolidated[p2][x_label], alternative='two-sided')
            if p_value < 0.05:
                significance_char = "**"
            else:
                significance_char = ""
            print(f"{p1} | {p2}: {x_label}: sample_sizes: {len(consolidated[p1][x_label])}, {len(consolidated[p2][x_label])}, mean_difference: {abs(np.mean(consolidated[p1][x_label]) - np.mean(consolidated[p2][x_label]))}, p_val: {significance_char}{p_value}{significance_char}")
            # print(f"    {np.mean(consolidated[p1][x_label])} | {np.mean(consolidated[p2][x_label])}")
            # print(f"    {consolidated[p1][x_label]} | {consolidated[p2][x_label]}")

def plot_model_losses():
    paths = [] 
    for root, dirs, files in os.walk("save/"):
        if "archive" in root:
            continue
        for file in files:
            if "loss_map" in file:
                paths.append(os.path.join(root, file))
    for path in paths:
        print(path)
        mapp = pickle.load(open(path, "rb"))
        dataset = get_dataset(path)
        fig, ax = plt.subplots()
        if isinstance(mapp["train"], list): ##for backwards compatability before we started using other types of losses than mse 
            train_y = mapp['train']
            val_y = mapp['val']
        if isinstance(mapp["train"], dict):
            train_y = mapp["train"]["epoch_loss"]
            val_y = mapp["val"]["avg_loss"]
        x = np.array(range(0, len(train_y)))
        ax.plot(x, train_y, label="train loss")
        ax.plot(x, val_y, label="val loss")
        plt.title(f"Loss Curves: {dataset.capitalize()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper right', prop={"size":7}, bbox_to_anchor=(1, 1.32))
        plt.gcf().subplots_adjust(top=.76)
        plt.savefig(f"outputs/loss_curve_{path.replace('/', '_')}.png", dpi=300)
        
def plot_cell_and_pert_counts():
    datasets = ["adam_corrected_upr", "norman", "replogle_k562_essential"]
    splits = ["train", "val", "test"]
    metrics = ["data_size", "perturbations"]
    gene_set_dict = {data_name: set() for data_name in datasets} 
    perturbation_dict = {data_name: {split: set() for split in splits} for data_name in datasets}
    data_size_dict = {data_name: {split: -1 for split in splits} for data_name in datasets}
    for data_name in datasets:
        pert_data = get_pert_data(data_name)        
        gene_set_dict[data_name] = set(pert_data.adata.var["gene_name"].tolist())
        for load_type in splits:
            loader = pert_data.dataloader[f"{load_type}_loader"]
            data_size_dict[data_name][load_type] = len(loader.dataset)
            perturbations = set()
            for batch, batch_data in enumerate(loader):
                for i in range(0, len(batch_data)):
                    perturbations.add(batch_data.pert[i])
            perturbation_dict[data_name][load_type] = perturbations
    for data_name in datasets:
        assert(len(perturbation_dict[data_name]["train"].intersection(perturbation_dict[data_name]["val"])) == 0)
        assert(len(perturbation_dict[data_name]["train"].intersection(perturbation_dict[data_name]["test"])) == 0)
        assert(len(perturbation_dict[data_name]["val"].intersection(perturbation_dict[data_name]["test"])) == 0)
    for data_name in datasets:
        for split in splits:
            print(f"{data_name}: ctrl in {split}: ", "ctrl" in perturbation_dict[data_name][split])
    ##make perturbation count dict for plotting 
    perturbation_count_dict = {data_name: {split: len(set(perturbation_dict[data_name][split])) for split in splits} for data_name in datasets}
    ##make graphs for counts 
    for plot_type in ["Perturbation Counts", "Data Size"]:
        dictionary = perturbation_count_dict if plot_type == "Perturbation Counts" else data_size_dict
        fig, ax = plt.subplots()
        x_labels = datasets
        width = 0.30
        x = np.array(range(0, len(x_labels)))
        ax.set_xticks(x)
        color_map = {"train": "green", "val": "silver", "test":"cornflowerblue"}
        for load_type in splits:
            y = []
            for data_name in datasets:
                y.append(dictionary[data_name][load_type])
            ax.bar(x, y, width=width, label=load_type.replace("val", "validation"), color=color_map[load_type])
            x = x + width
        plt.title(f"{plot_type}")
        ax.set_xticklabels([get_dataset_title(x) for x in x_labels])
        ax.set_xlabel("Dataset")
        y_label = "Cells" if plot_type == "Data Size" else "Perturbations"
        ax.set_ylabel(y_label)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper right', prop={"size":7}, bbox_to_anchor=(1, 1.32))
        plt.gcf().subplots_adjust(top=.76)
        plt.savefig(f"outputs/count_{plot_type}.png", dpi=300)  
    ##make graph plotting overlap of genes between datasets
    fig, ax = plt.subplots()
    labels = venn.get_labels([gene_set_dict[data_name] for data_name in datasets], fill="number")
    if len(datasets) == 3:
        # v = venn3([gene_set_dict[data_name] for data_name in datasets], datasets)
        fig, ax = venn.venn3(labels, names=datasets)
    if len(datasets) == 4:
        fig, ax = venn.venn4(labels, names=datasets)
    plt.title("Gene Overlap Between Datasets")
    plt.savefig(f"outputs/venn_diagram_gene_overlap.png", dpi=300)
    ##overlap of perturbations between datasets
    ##get dataset to set of pertubations mapping
    dataset_to_pert_set = {data_name: perturbation_dict[data_name]["train"].union(perturbation_dict[data_name]["val"]).union(perturbation_dict[data_name]["test"]) for data_name in datasets}
    ##since syntax is slightly different between datasets, e.g. 'SRP68+ctrl' vs 'ctrl+SPR68', extract just the genes
    for data_name in datasets:
        new_entries = []
        s = dataset_to_pert_set[data_name]
        for string in s:
            gene_names = get_gene_names(string)
            if len(gene_names) == 1:
                new_entries.append(gene_names[0])
            else:
                gene_names = sorted(gene_names) ##sort so that string representation will always be the same regardless of string membership 
                new_entries.append("".join(gene_names))
        dataset_to_pert_set[data_name] = set(new_entries)
    for data_name in dataset_to_pert_set:
        print(data_name, list(dataset_to_pert_set[data_name])[0:10])
    fig, ax = plt.subplots()
    labels = venn.get_labels([dataset_to_pert_set[data_name] for data_name in datasets], fill="number")
    if len(datasets) == 3:
        fig, ax = venn.venn3(labels, names=datasets)
    if len(datasets) == 4:
        fig, ax = venn.venn4(labels, names=datasets)
    plt.title("Perturbation Overlap Between Datasets")
    plt.savefig(f"outputs/venn_diagram_pert_overlap.png", dpi=300)

def plot_gene_counts():
    gene_len_dict = {}
    for data_name in datasets:
        pert_data = get_pert_data(data_name)        
        gene_len_dict[data_name] = len(set(pert_data.adata.var["gene_name"].tolist()))
    fig, ax = plt.subplots()
    x_labels = datasets
    x = np.array(range(0, len(x_labels)))
    plt_x_labels = [get_dataset_title(x) for x in x_labels]
    ax.set_xticks(x)
    ax.set_xticklabels(plt_x_labels)
    ax.set_xlabel("Dataset")
    y = [gene_len_dict[x_label] for x_label in x_labels]
    ax.bar(x, y)
    for i,j in zip(x, y):
        ax.annotate(f"{j}", xy=(i - 0.11, j + 120),fontsize=10)
    plt.title("Number of Genes in Dataset")
    plt.ylim((0, max(y) + 500))
    plt.savefig("outputs/gene_sizes.png", dpi=300)

def print_cell_types():
    for data_name in datasets:
        pert_data = PertData("./data")
        pert_data.load(data_name=data_name)
        pert_data.prepare_split(split="simulation", seed=1)
        pert_data.get_dataloader(batch_size=64, test_batch_size=64)
        cell_types = set(pert_data.adata.obs["cell_type"])
        print(data_name, cell_types)
        print(data_name, pert_data.adata.obs.keys())

def print_expression_values():
    for data_name in datasets:
        pert_data = PertData("./data")
        pert_data.load(data_name=data_name)
        pert_data.prepare_split(split="simulation", seed=1)
        pert_data.get_dataloader(batch_size=64, test_batch_size=64)
        c_mean = np.mean(pert_data.adata.X.A, axis=0)
        mean = np.mean(pert_data.adata.X.A)
        std = np.std(pert_data.adata.X.A)
        mini = np.min(pert_data.adata.X.A)
        maxi =  np.max(pert_data.adata.X.A)
        print(data_name)
        print(pert_data.adata.X.A)
        print(f"column mean: {c_mean}")
        print(f"mean: {mean}, std: {std}, min: {mini}, max: {maxi}")

def get_weight_similarity():
    """
    Iterate over each dataset and get pretrained_dict_1 = model trained from scratch, pretrained_dict_2 = foundation weights 
    Will plot cosine similarity box plots for shared weights between the two state dicts
    """
    weight_dictionary = {data_name: torch.load(f"save/no_pretraining/{data_name}/best_model.pt")                         for data_name in datasets}
    weight_dictionary = {data_name: {k.replace("Wqkv.", "in_proj_"): v for k, v in weight_dictionary[data_name].items()} for data_name in datasets}
    weight_dictionary["foundation"] = torch.load( "models/scgpt-pretrained/scGPT_human/best_model.pt")
    weight_dictionary["foundation"] = {k.replace("Wqkv.", "in_proj_"): v for k, v in weight_dictionary["foundation"].items()}
    all_models = list(weight_dictionary.keys())
    combos = [(a, b) for idx, a in enumerate(all_models) for b in all_models[idx + 1:]]
    for model_1, model_2 in combos: 
        pretrained_dict_1 = weight_dictionary[model_1]
        pretrained_dict_2 = weight_dictionary[model_2]
        delta = {}
        for key in pretrained_dict_1:
            if key in pretrained_dict_2:
                random_init =  torch.empty(pretrained_dict_1[key].shape).data.uniform_(-0.1, 0.1).detach().cpu().numpy().flatten()
                random_sim_1 = 1 - scipy.spatial.distance.cosine(pretrained_dict_1[key].detach().cpu().numpy().flatten(), random_init)
                random_sim_2 = 1 - scipy.spatial.distance.cosine(pretrained_dict_2[key].detach().cpu().numpy().flatten(), random_init)
                print(random_sim_1, random_sim_2)
                # assert(random_sim_1 < 0.20) ##should in theory be very low 
                # assert(random_sim_2 < 0.20)
                avg_random_sim = (random_sim_1 + random_sim_2) / float(2.0)
                sim = 1 - scipy.spatial.distance.cosine(pretrained_dict_1[key].detach().cpu().numpy().flatten(), pretrained_dict_2[key].detach().cpu().numpy().flatten())
                delta[key] = (sim, avg_random_sim)
            else:
                print(f"key: {key} not matched in the two dictionaries")
        sorted_delta = sorted(delta.items(), key=lambda x: x[1][0])
        for tup in sorted_delta:
            print(tup)
        ##make box plots
        fig, ax = plt.subplots()
        all_points = [tup[1][0] for tup in sorted_delta]
        norm_points = [tup[1][0] for tup in sorted_delta if "norm" in tup[0]]
        not_norm_points = [tup[1][0] for tup in sorted_delta if "norm" not in tup[0]]
        plot_map = {f"all\n(n={len(all_points)})": all_points, f"norm\n(n={len(norm_points)})": norm_points, f"not norm\n(n={len(not_norm_points)})": not_norm_points}
        ax.boxplot(plot_map.values())
        ax.set_xticklabels(plot_map.keys(), fontsize=8)
        plt.title(f"Cosine Similarity Between {model_1.title()} and {model_2.title()}", fontsize=10)
        plt.xlabel("Weight Type", fontsize=8)
        plt.ylabel("Cosine Similarity", fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylim((-1.1, 1.1))
        plt.savefig(f"outputs/weight_sim_{model_2}_{model_1}.png", dpi=300)

def find_best_models(root_dirs, mode=1):
    """
    Returns a map from model to path with highest score
    Searched root_dirs
    will also print a mapping from directory to best model contained within that directory by dataset
    """
    ##find all paths to result files within save, check if part of a multi-run 
    model_types = ["scgpt", "simple_affine", "gears", "linear_additive", "latent_additive", "decoder_only"]
    directory_to_best = {root: {dataset: ("", 0) for dataset in datasets} for root in root_dirs}
    paths = [] 
    for root_dir in root_dirs:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if "result" not in file:
                    continue
                model_type = get_model_type(file)
                if model_type not in model_types:
                    continue
                if mode == 1 and "_pert_delta_results" in file:
                    paths.append(os.path.join(root, file))
                    continue
                if mode == 2 and "_results" in file and "pert_delta" not in file:
                    paths.append(os.path.join(root, file))

    ##iterate over file paths and find best model on test set 
    best_map = {model_type: {dataset: ("", 0) for dataset in datasets} for model_type in model_types}
    for path in paths: 
        ##get dataset that results are for 
        dataset = get_dataset(path)
        if dataset not in best_map[model_type]:
            continue
        model_type = get_model_type(path)
        if mode == 1: 
            model_res = pickle.load(open(path, "rb"))
            score = model_res["pearson_delta"] ##or can use pearson_de_delta
        else:
            model_res, _ = pickle.load(open(path, "rb"))
            score = model_res["pearson"]
        if score > best_map[model_type][dataset][1]:
            best_map[model_type][dataset] = (path, score)
        for root_dir in root_dirs:
            if root_dir in path and score > directory_to_best[root_dir][dataset][1]:
                directory_to_best[root_dir][dataset] = (path, score)
    print(best_map, "\n")
    for root_dir in root_dirs:
        for dataset in datasets:
            print(root_dir, dataset, directory_to_best[root_dir][dataset])
        print("\n")
    return best_map

def plot_wasserstein_pert_gene_comparison():
    """
    Compares two wasserstein distance distributions: 
    1: target gene T:              expression of target gene in cells perturbed by query  <--> expression of target gene in cells perturbed by something other than query 
    2: de genes != target gene:  expression of de genes != target gene for cells perturbed by query  <--> expression of de genes != target gene for cells perturbed by something other than query 
    """
    datasets = ["adam_corrected_upr", "norman", "replogle_k562_essential"]
    dataset_to_w = {data_name: () for data_name in datasets} ##key: dataset, value: (w1 mean, w1 std, w1 len, w2 mean, w2 std, w2 len)
    ##iterate over each dataset and compute wassersteins from test set 
    for data_name in datasets: 
        pert_data = get_pert_data(data_name)
        adata = pert_data.adata
        wasserstein_1_list, wasserstein_2_list = [], []
        for query in set(adata.obs["condition"]): 
            if query == "ctrl":
                continue
            gene2idx = pert_data.node_map ##key: normal gene name, value: index 
            cond2name = dict(adata.obs[["condition", "condition_name"]].values) ##key: condition, value: condition_name
            gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values)) ##key: ENSG gene name, value: normal gene name
            de_idx = [gene2idx[gene_raw2id[i]] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]] ##adata.uns["top_non_dropout_de_20"][cond2name[query]] is a list of ENSG genes
            all_genes = [gene_raw2id[i] for i in adata.var.index.values]
            ##cells perturbed by query
            perturbed_values = adata[adata.obs["condition"] == query].to_df().to_numpy()
            ##all perturbed cells not perturbed by query
            non_control = adata[adata.obs["condition"] != "ctrl"]
            other_perturbed_values = non_control[non_control.obs["condition"] != query].to_df().to_numpy()
            ##get the target gene from the query string
            perturbed_genes = extract_genes(query)
            ##compute W1
            for p_g in perturbed_genes:
                index_p_g = all_genes.index(p_g)
                p_g_perturbed = perturbed_values[:, index_p_g] ##expression of target gene in cells perturbed by query 
                p_g_other_perturbed = other_perturbed_values[:, index_p_g] ##expression of target gene in cells perturbed by something other than query 
                wasserstein_1_list.append(scipy.stats.wasserstein_distance(p_g_perturbed, p_g_other_perturbed))
            ##de genes that are not equal to any of the target genes
            de_genes = [gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]] ##or: de_genes = np.array(all_genes)[de_idx]
            de_other_genes = [gene for gene in de_genes if gene not in perturbed_genes]
            ##compute W2
            for de_o_g in de_other_genes: 
                index_de_o_g = all_genes.index(de_o_g)
                de_o_g_perturbed = perturbed_values[:, index_de_o_g] ##expression of de_o_g for cells perturbed by query 
                de_o_g_other_perturbed = other_perturbed_values[:, index_de_o_g] ##expression of de_o_g for cells perturbed by something other than query 
                wasserstein_2_list.append(scipy.stats.wasserstein_distance(de_o_g_perturbed, de_o_g_other_perturbed))
        dataset_to_w[data_name] = (wasserstein_1_list, wasserstein_2_list)
    ##plot using data from dataset_to_w
    fig, ax = plt.subplots()
    x_labels = [f"Target gene T", f"DE Genes ≠ T"]
    color_map = {"adam_corrected": "lightsteelblue", "adam_corrected_upr": "lightsteelblue", "adamson": "lightsteelblue", "norman": "tan", "replogle_k562_essential": "slategrey"}
    spacer = -0.4
    widths = 0.3
    for data_name in datasets: 
        ##make into boxplots
        y = [dataset_to_w[data_name][0], dataset_to_w[data_name][1]]
        ##significance test
        t_statistic, p_value = scipy.stats.ttest_ind(dataset_to_w[data_name][0], dataset_to_w[data_name][1], alternative='two-sided')
        print(f"Wasserstein statistical tests: {data_name}: t_statistic: {t_statistic}, p_value: {p_value}")
        bp_targets = ax.boxplot(y, positions=np.array(range(0, len(y)))*2.0 + spacer, sym='', widths=widths)
        spacer = spacer + 0.4
        set_box_color(bp_targets, color_map[data_name], plt)
        plt.plot([], c=color_map[data_name], label=get_dataset_title(data_name))
    ##set x tick labels
    ticks = x_labels
    plt.xticks(range(0, len(ticks) * 2, 2), ticks, rotation=0)
    plt.xlim(-2, len(ticks)*2)
    plt.rcParams.update({'mathtext.default': 'regular' })
    plt.title("Wasserstein Distance Between\n$Cells_{target=T}$ and All Other Perturbed $Cells_{target≠T}$")
    ax.set_ylabel("Wasserstein Distance")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"size":7}, bbox_to_anchor=(1, 1.35))
    plt.gcf().subplots_adjust(top=.76)
    plt.savefig(f"outputs/wasserstein_distance_comparison.png", dpi=300)

def plot_avg_pearson_to_avg_perturbed_state():
    """
    For each target T, compute the pearson between (1) average of cells with target = T and (2) average of perturbed cells with target != T
    Plot boxplots of distributions
    """
    datasets = ["adam_corrected_upr", "norman", "replogle_k562_essential"]
    dataset_map = {dataset: "" for dataset in datasets}
    for dataset in datasets: 

        pert_data = get_pert_data(dataset)
        # if data_name in ["adamson", "norman", "replogle_k562_essential"]: 
        #     pert_data = PertData("./data")
        #     pert_data.load(data_name=data_name)
        #     pert_data.prepare_split(split="simulation", seed=1)
        #     pert_data.get_dataloader(batch_size=64, test_batch_size=64)
        # if data_name == "adam_corrected":
        #     pert_data = get_adam_corrected_dataset(split="simulation", batch_size=64, test_batch_size=64, generate_new=False, just_upr=False)
        # if data_name == "adam_corrected_upr":
        #     pert_data = get_adam_corrected_dataset(split="simulation", batch_size=64, test_batch_size=64, generate_new=False, just_upr=True)
        # if data_name == "replogle_k562_gwps":
        #     pert_data = get_replogle_gwps_pert_data(split="simulation", batch_size=64, test_batch_size=64, generate_new=False)
        # if "replogle" in dataset:
        #     modify_pertdata_anndata(pert_data)
        
        adata = pert_data.adata 
        perturbed_adata = adata[adata.obs["condition"] != "ctrl"]
        perturbed_adata_values = perturbed_adata.to_df().values
        avg_perturbed_vector = adata[adata.obs["condition"] != "ctrl"].to_df().mean().values
        control_adata_values = adata[adata.obs["condition"] == "ctrl"].to_df().values
        avg_control_vector = adata[adata.obs["condition"] == "ctrl"].to_df().mean().values
        ##get perturbed_to_avg_perturbed, avg_perturbed will be computed for each target T by excluding T from the average 
        perturbed_to_avg_perturbed = []
        for condition in set(perturbed_adata.obs["condition"]):
            if condition == "ctrl":
                raise Exception("control wasn't excluded!")
            my_avg =  perturbed_adata[perturbed_adata.obs["condition"] == condition].to_df().mean().values
            other_avg = perturbed_adata[perturbed_adata.obs["condition"] != condition].to_df().mean().values
            perturbed_to_avg_perturbed.append(scipy.stats.pearsonr(my_avg, other_avg)[0])
        avg_perturbed_to_perturbed, std_perturbed_to_perturbed = np.mean(perturbed_to_avg_perturbed), np.std(perturbed_to_avg_perturbed)
        dataset_map[dataset] = perturbed_to_avg_perturbed
    fig, ax = plt.subplots()
    x_labels = list(dataset_map.keys())
    y = [dataset_map[x_label] for x_label in x_labels]
    ax.boxplot(y)
    plt.ylim((0.0, 1.05))
    plt.xticks(list(range(1, len(x_labels) + 1)), [get_dataset_title(x_label) for x_label in x_labels])
    plt.rcParams.update({'mathtext.default': 'regular' })
    plt.title('Distribution of Pearson Correlations Between\n$Cells_{target=T}$ and All Other Perturbed $Cells_{target≠T}$')
    ax.set_ylabel("Pearson Correlation")
    ##scale for consistency with other figures' scale
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    plt.gcf().subplots_adjust(top=.76)
    plt.savefig("outputs/pearson_perturbed_to_perturbed.png", dpi=300)

def find_best_run_number(string, model_type):
    if model_type == "scgpt":
        run_number = re.findall(r"run_[0-9]+", string)[0]
        return run_number.split("_")[1]
    if model_type == "gears":
        run_number = re.findall(r"[0-9]+.pkl", string)[0]
        return run_number.split(".")[0]

def plot_rank_scores(mode=1, include_perturbench=False):
    fig, ax = plt.subplots()
    root_dirs = [ "save/default_config_baseline/", "save/perturbench/", "pickles/gears_results/"]
    best_models = find_best_models(root_dirs, mode=mode)
    model_types = ["gears", "mean_perturbed", "smart_mean_perturbed", "scgpt", "linear_additive", "latent_additive", "decoder_only"]
    proper_x_labels = {"gears": "GEARS", "scgpt": "scGPT", "mean_perturbed": "Mean", "smart_mean_perturbed": "CRISPR-informed\nMean", "linear_additive": "Linear Additive", "latent_additive": "Latent Additive", "decoder_only": "Decoder Only"}
    width = 0.15
    color_map = {"adam_corrected": "lightsteelblue", "adam_corrected_upr": "lightsteelblue", "adamson": "lightsteelblue", "norman": "tan", "replogle_k562_essential": "slategrey"}
    spacer = -0.4
    widths = 0.3
    scgpt_vs_smart = []
    gears_vs_smart = []
    for dataset in ["adam_corrected_upr", "norman", "replogle_k562_essential"]:
        gears_best_run_file = best_models["gears"][dataset][0]
        gears_best_run_number = find_best_run_number(gears_best_run_file, "gears")
        ##for boxplot
        model_to_rank_map = {"gears": pickle.load(open(f"pickles/gears_results/gears_rank_metrics_{dataset}_{gears_best_run_number}.pkl", "rb")), 
                            "scgpt": pickle.load(open(f"pickles/rank_metrics_{dataset}_scGPT.pkl", "rb")), 
                            "mean_perturbed": pickle.load(open(f"pickles/rank_metrics_{dataset}_mean_perturbed.pkl", "rb")),
                            "smart_mean_perturbed": pickle.load(open(f"pickles/rank_metrics_{dataset}_smart_mean_perturbed.pkl", "rb"))
        }
        if include_perturbench: 
            p_models = ["linear_additive", "latent_additive", "decoder_only"]
            model_to_rank_map.update({p_model: pickle.load(open(best_models[p_model][dataset][0], "rb")) for p_model in p_models}) 

        ##compute avg rank and compare methods
        method_to_avg = {}
        for model_type in model_to_rank_map:
            avg_rank = np.mean(list(model_to_rank_map[model_type].values()))
            method_to_avg[model_type] = avg_rank
        print(f"{dataset} method to avg: {method_to_avg}")
        scgpt_vs_smart.append(method_to_avg["scgpt"] / method_to_avg["smart_mean_perturbed"])
        gears_vs_smart.append(method_to_avg["gears"] / method_to_avg["smart_mean_perturbed"])
        ##plot
        y = [list(model_to_rank_map[model_type].values()) for model_type in model_types]
        bp_dataset = ax.boxplot(y, positions=np.array(range(0, len(model_types)))*2.0 + spacer, sym='', widths=widths)
        spacer = spacer + 0.4
        set_box_color(bp_dataset, color_map[dataset], plt)
        plt.plot([], c=color_map[dataset], label=get_dataset_title(dataset))
    print(f"scgpt vs smart mean change fold avg: {scgpt_vs_smart} {np.mean(scgpt_vs_smart)}")
    print(f"gears vs smart mean change fold avg: {gears_vs_smart} {np.mean(gears_vs_smart)}")
    ##set x tick labels
    ticks = [proper_x_labels[x_label] for x_label in model_types]
    if include_perturbench: 
        plt.xticks(range(0, len(ticks) * 2, 2), ticks, rotation=20, fontsize=7.5)
    else:
        plt.xticks(range(0, len(ticks) * 2, 2), ticks, rotation=8, fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlim(-2, len(ticks)*2)
    plt.title(f"Rank Comparison Between Different Models")
    ax.set_ylabel("Rank")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"size":7}, bbox_to_anchor=(1, 1.30))
    plt.gcf().subplots_adjust(top=.76)
    plt.savefig(f"outputs/rank_metrics.png", dpi=300)
    
def plot_simple_affine_run_times():
    datasets = ["adam_corrected_upr", "norman", "replogle_k562_essential"]
    model_map = {dataset: {"Simple Affine": ["save/simple_affine/", []], "scGPT": ["save/default_config_baseline/", []]} for dataset in datasets}
    for dataset in model_map: 
        for model in model_map[dataset]:
            for root, dirs, files in os.walk(model_map[dataset][model][0]):
                for file in files:
                    if ".log" in file and dataset == get_dataset(root):
                        with open(os.path.join(root, file)) as file:
                            for line in file:
                                if "time: " in line:
                                    elapsed_time = float(re.findall(r"time:\s+[0-9]+.[0-9]+s", line)[0].split("time: ")[1].replace("s", "")) #/s+ captures any number of white spaces, when t < 10 seconds will have double white space
                                    model_map[dataset][model][1].append(elapsed_time)
    ##codense to avg and std 
    for dataset in model_map:
        for model in model_map[dataset]:
            model_map[dataset][model][1] = (np.mean(model_map[dataset][model][1]), np.std(model_map[dataset][model][1]))
    ##make bar graph comparisons, x-axis = model, y-axis = time, legend by dataset
    fig, ax = plt.subplots()
    width = 0.15
    x = np.array([1,2])
    ax.set_xticks(x)
    x_labels = ["Simple Affine", "scGPT"]
    ax.set_xticklabels(x_labels)
    color_map = {"adam_corrected": "lightsteelblue", "adam_corrected_upr": "lightsteelblue", "adamson": "lightsteelblue", "norman": "tan", "replogle_k562_essential": "slategrey"}
    for dataset in model_map:
        y = [model_map[dataset][x_label][1][0] for x_label in x_labels]
        yerr = [model_map[dataset][x_label][1][1] for x_label in x_labels]
        ax.bar(x, y, yerr=yerr, label=get_dataset_title(dataset), color=color_map[dataset], width=width, error_kw={"elinewidth":0.5, "capsize":0.5})
        x = x + width
    plt.title(f"Epoch Training Time Comparsion")
    ax.set_ylabel("Training Time (seconds)")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"size":7}, bbox_to_anchor=(1, 1.30))
    plt.gcf().subplots_adjust(top=.76)
    plt.savefig(f"outputs/training_time.png", dpi=300)
    
def compare_number_model_params():
    pert_data = PertData("./data")
    pert_data.load(data_name="adamson")
    var = get_variables(load_model="models/scgpt-pretrained/scGPT_human", config_path="config/default_config.json")
    model_file, vocab, n_genes, gene_ids, ntokens = get_model_setup(var, pert_data, logger=scg.logger)
    scGPT = TransformerGenerator(
            ntoken=ntokens,
            d_model=var["embsize"],
            nhead=var["nhead"],
            d_hid=var["d_hid"],
            nlayers=var["nlayers"],
            nlayers_cls=var["n_layers_cls"],
            n_cls=1,
            vocab=vocab,
            dropout=var["dropout"],
            pad_token=var["pad_token"],
            pad_value=var["pad_value"],
            pert_pad_id=var["pert_pad_id"],
            use_fast_transformer=var["use_fast_transformer"],
        )
    scgpt_params = sum(p.numel() for p in scGPT.parameters())
    print(f"Number of scGPT parameters: {scgpt_params}")
    from simple_affine import SimpleAffine 
    sa = SimpleAffine(
        ntoken=ntokens,
        d_model=var["embsize"],
        nlayers=var["nlayers"],
        nlayers_cls=var["n_layers_cls"],
        vocab=vocab,
        dropout=var["dropout"],
        pad_token=var["pad_token"],
        pert_pad_id=var["pert_pad_id"],
    )
    simple_affine_params = sum(p.numel() for p in sa.parameters())
    print(f"Number of Simple Affine parameters: {simple_affine_params}")
    fig, ax = plt.subplots()
    width = 0.15
    x = np.array([1,2])
    ax.set_xticks(x)
    x_labels = ["Simple Affine", "scGPT"]
    ax.set_xticklabels(x_labels)
    y = [simple_affine_params, scgpt_params]
    y = [float(y_ / 1000000.0) for y_ in y]
    ax.bar(x, y, width=width, color=["grey", "#519E3E"])
    for i,j in zip(x, y):
        ax.annotate(f"{j:.2E}", xy=(i - .06, j + 800000),fontsize=7)
    plt.title(f"Trainable Parameters Comparsion")
    ax.set_ylabel("Number of Parameters (in Millions)")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    plt.gcf().subplots_adjust(top=.76)
    plt.savefig(f"outputs/parameter_size.png", dpi=300)

def plot_perturbench_comparison(mode):
    dataset_map = get_baseline_dataset_map(mode)
    baseline_map = {dataset: "" for dataset in datasets} #key dataset: value: baseline_y_std_map (key: model, value: tuple(scores list, std list) corresponding to x_labels)
    perturbench_models = ["linear_additive", "latent_additive", "decoder_only"]
    perturbench_map = {perm: {dataset: "" for dataset in datasets} for perm in perturbench_models} ##key perturbench permutation as directory path, key: dataset, value: (y_model, y_std)
    paths = [] 
    for root, dirs, files in os.walk("save/perturbench/"):
        for file in files:
            if "latent_additive" not in file and "linear_additive" not in file and "decoder_only" not in file:
                continue
            for perturbench_model in perturbench_models:
                if perturbench_model in file:
                    model_type = perturbench_model
            if mode == 1 and f"{model_type}_pert_delta_results" in file:
                paths.append(os.path.join(root, file))
            if mode == 2 and f"{model_type}_results" in file:
                paths.append(os.path.join(root, file))
    for path in paths: 
        dataset = get_dataset(path)
        if dataset not in dataset_map.keys():
            continue
        x_labels = list(dataset_map[dataset]["gears"].keys())
        baseline_y_std_map = get_baseline_y_std_map(dataset_map, dataset, x_labels)
        baseline_map[dataset] = baseline_y_std_map
        y_model, y_model_std = get_path_results(path, paths, x_labels, mode)
        for key in perturbench_map:
            if key in path: 
                assigned_key = key
                break
        if  perturbench_map[assigned_key][dataset] != "":
            assert( perturbench_map[assigned_key][dataset] == (y_model, y_model_std)) ##this should be the same for paths of the same multi-run
        else:
            perturbench_map[assigned_key][dataset] =  (y_model, y_model_std)
    ##plot 
    for dataset in baseline_map.keys():
        x_labels = list(dataset_map[dataset]["gears"].keys())
        baseline_y_std_map = get_baseline_y_std_map(dataset_map, dataset, x_labels)
        baseline_map[dataset] = baseline_y_std_map
        fig, ax = plt.subplots()
        x = np.array(range(0, len(x_labels)))
        ax.set_xticks(x)
        anchor = 1.37
        y_dict = {"GEARS": (baseline_map[dataset]["gears"][0], baseline_map[dataset]["gears"][1], "#3B75AF"), 
                "Mean": (baseline_map[dataset]["mean_perturbed"][0], baseline_map[dataset]["mean_perturbed"][1], "salmon"),
                "CRISPR-informed Mean": (baseline_map[dataset]["smart_mean_perturbed"][0], baseline_map[dataset]["smart_mean_perturbed"][1] , "goldenrod"),
                "scGPT Fully Fine-Tuned Baseline": (baseline_map[dataset]["scGPT"][0], baseline_map[dataset]["scGPT"][1], "#519E3E"),
                "Linear Additive": (perturbench_map["linear_additive"][dataset][0], perturbench_map["linear_additive"][dataset][1], "dimgray"),
                "Latent Additive": (perturbench_map["latent_additive"][dataset][0], perturbench_map["latent_additive"][dataset][1], "darkgrey"),
                "Decoder Only": (perturbench_map["decoder_only"][dataset][0], perturbench_map["decoder_only"][dataset][1], "lightgrey")
                }
        ##make the plot and annotate it
        width = 0.12
        for method in y_dict:
            ax.bar(x, y_dict[method][0], yerr=y_dict[method][1],  width=width, error_kw={"elinewidth":0.5, "capsize":0.5}, label=method, color=y_dict[method][2])
            for i,j in zip(x, y_dict[method][0]):
                ax.annotate(f"{round(j, 2)}", xy=(i - .06, j +.02),fontsize=4.5)
            x = x + width
        ax.hlines(y=0.0, xmin=0, xmax=len(x_labels), linestyles="dashed", color="black", linewidth=0.5)
        plt.title(f"Model Evaluations for {get_dataset_title(dataset)} Dataset", fontsize=10)
        x_label_map = {"pearson": "Pearson", "pearson_de": "Pearson DE", "pearson_delta": "Pearson Delta", "pearson_de_delta": "Pearson DE Delta"}
        plt_x_labels = [x_label_map[x] for x in x_labels]
        ax.set_xticklabels(plt_x_labels, fontsize=8.5)
        plt.xticks(rotation=15)
        ax.set_ylabel("Pearson Score")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper right', prop={"size":6}, bbox_to_anchor=(1, anchor))
        plt.gcf().subplots_adjust(top=.76)
        plt.savefig(f"outputs/perturbench_comparison_{dataset}_mode={mode}.png", dpi=300)


def make_metric_violinplots(dir = '', output_prefix='violinplot', exclude_target_gene=False, exclude_combos=False):
    """
    function written by Abby for supplemental figure
    dir = relative path to dataset file results
    output_prefix = beginning of path to save the output plot
    exclude_target_gene = whether to exclude the gene that was perturbed and the condition where the gene was targeted
    exclude_combos = whether to exclude combination perturbations
    """
    for dataset in datasets: 
        print(f'exclude combos 0: {exclude_combos}')
        # Find all pickle files with the given prefix
        pickle_files = glob.glob(f"{dir}pert_data_structure*{dataset}_run*.pkl")
        # Initialize an empty DataFrame to store all data
        model_names = []
        all_data = pd.DataFrame()
        # Loop through each pickle file and concatenate the data
        all_gene_level_metrics_df = pd.DataFrame()
        all_perturbation_level_metrics_df = pd.DataFrame()
        for pickle_file in pickle_files:
            print(f"pickle_file: {pickle_file}")
            # get model name from filename
            print(f"dataset: {dataset}")
            dataset_first_appearance = dataset+'_s' # dataset name is followed by either _scGPT or _simple_affine
            dataset_second_appearance = dataset+'_r' # dataset name is followed by _run
            start = pickle_file.find(dataset_first_appearance) + len(dataset_first_appearance) - 1
            end = pickle_file.find(dataset_second_appearance) - 1
            model_name = pickle_file[start:end].replace('_load_save_','_')
            print(f"model_name: {model_name}")
            model_names.append(model_name)
            with open(pickle_file, 'rb') as file:
                data = pickle.load(file)
            data = find_combine_redundant_conditions(data)
            [model_preds_df, actual_data_df, control_means_df] = model_res_to_dfs(data)
            model_preds_df = clean_up_data(dataset, model_preds_df)
            actual_data_df = clean_up_data(dataset, actual_data_df)
            control_means_df = clean_up_data(dataset, control_means_df)
            current_model_gene_level_metrics = get_gene_level_metrics(model_preds_df, actual_data_df, gene_names=model_preds_df.index, exclude_target_gene=exclude_target_gene, exclude_combos=exclude_combos)
            current_model_gene_level_metrics.columns = [col + f'_{model_name}' for col in current_model_gene_level_metrics.columns]
            all_gene_level_metrics_df = pd.concat([all_gene_level_metrics_df, current_model_gene_level_metrics], axis=1)
            current_model_perturbation_level_metrics = get_perturbation_level_metrics(model_preds_df, actual_data_df, control_means_df, perturbation_labels = model_preds_df.columns, exclude_target_gene=exclude_target_gene, exclude_combos=exclude_combos)
            current_model_perturbation_level_metrics.columns = [col + f'_{model_name}' for col in current_model_perturbation_level_metrics.columns]
            all_perturbation_level_metrics_df = pd.concat([all_perturbation_level_metrics_df, current_model_perturbation_level_metrics], axis=1)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # gene-level violin plot
        gene_plot_data = all_gene_level_metrics_df[['Pearson_correlation_simple_affine_simple_affine','Pearson_correlation_simple_affine_simple_affine_with_pretraining', 'Pearson_correlation_scGPT_no_pretraining', 'Pearson_correlation_scGPT_default_config_baseline']]
        num_genes = gene_plot_data.shape[0]
        if num_genes < 500:
            sns.swarmplot(data=gene_plot_data, ax=ax[0])
        else:
            sns.violinplot(data=gene_plot_data, ax=ax[0])
        # fix x-axis labels
        new_labels = ['Simple Affine no pretraining', 'Simple Affine pretrained', 'scGPT no pretraining', 'scGPT pretrained']
        ax[0].set_xticklabels(new_labels)
        ax[0].set_title(f'{get_dataset_title(dataset)}: Gene-level Pearson Correlation')
        ax[0].set_xlabel('Model')
        ax[0].set_ylabel('Pearson Correlation (for Genes)')
        ax[0].set_ylim(-1.3, 1.3)
        # perturbation-level violin plot
        pert_plot_data = all_perturbation_level_metrics_df[['Pearson_delta_simple_affine_simple_affine','Pearson_delta_simple_affine_simple_affine_with_pretraining', 'Pearson_delta_scGPT_no_pretraining', 'Pearson_delta_scGPT_default_config_baseline']]
        num_perts = pert_plot_data.shape[0]
        if num_perts < 500:
            sns.swarmplot(data=pert_plot_data, ax=ax[1], size=3)
        else:
            sns.violinplot(data=pert_plot_data, ax=ax[1], size=3)
        # fix x-axis labels
        ax[1].set_xticklabels(new_labels)
        ax[1].set_title(f'{get_dataset_title(dataset)}: Perturbation-level Pearson Deltas')
        ax[1].set_xlabel('Model')
        ax[1].set_ylabel('Pearson Delta (for Perturbations)')
        ax[1].set_ylim(-1.3, 1.3)
        for axis in ax:
            for label in axis.get_xticklabels():
                label.set_rotation(45)
        plt.subplots_adjust(bottom=0.3)
        # Save or show the plot
        if output_prefix:
            plt.savefig(f'{output_prefix}genes_and_perturbation_deltas_{dataset}_et_{exclude_target_gene}_ec_{exclude_combos}.png', dpi=300)
        else:
            plt.show()

def find_combine_redundant_conditions(data):
    """
    function written by Abby to combine values for conditions like GENE+ctrl and ctrl+GENE (appears in Norman data, maybe other datasets with combination perturbations)
    """
    conditions = list(data.keys())
    for condition in conditions:
        # strip off 'ctrl' part of the condition to find the actual gene name
        gene = condition.replace('ctrl+','')
        gene = gene.replace('+ctrl','')
        if f'ctrl+{gene}' in condition:
            print(f'found ctrl+{gene}')
            corresponding_condition = f'{gene}+ctrl'
            if corresponding_condition in conditions:
                print(f'found corresponding condition {corresponding_condition}')
                data = combine_conditions(data, corresponding_condition, condition)
            print(data.keys())
    return data

def combine_conditions(data, condition1, condition2):
    """
    helper function written by Abby
    """
    if condition1 in data and condition2 in data:
        data[condition1]['all_truth'] = np.concatenate((data[condition1]['all_truth'], data[condition2]['all_truth']))
        data[condition1]['all_pred'] = np.concatenate((data[condition1]['all_pred'], data[condition2]['all_pred']))
        del data[condition2]
    return data

def model_res_to_dfs(pert_struct):
    """
    helper function written by Abby - puts model results from pickle file into a DataFrame
    """
    # Initialize dictionaries to collect mean predictions and actual values
    pred_dict = {}
    actual_dict = {}
    # Iterate over the dictionary to calculate the mean predictions for each condition
    for condition, values in pert_struct.items():
        # Ensure the structure is as expected
        if 'all_genes' in values and 'all_pred' in values:
            gene_names = values['all_genes']
            predictions = values['all_pred'].transpose()
            actual = values['all_truth'].transpose()
            # Ensure predictions is a 2D array-like structure
            if not isinstance(predictions, (list, np.ndarray)) or not all(isinstance(row, (list, np.ndarray)) for row in predictions):
                raise ValueError(f"'predictions' should be a 2D array-like structure in condition: {condition}")
            # Convert predictions to a DataFrame and calculate the mean across cells
            predictions_df = pd.DataFrame(predictions, index=gene_names)
            actual_df = pd.DataFrame(actual, index=gene_names)
            mean_predictions = predictions_df.mean(axis=1)
            mean_actual = actual_df.mean(axis=1)
            # Collect the mean predictions and actual values in the dictionaries
            pred_dict[condition] = mean_predictions
            actual_dict[condition] = mean_actual
        else:
            if condition != 'actual_mean_perturbed':
                print(f"Missing 'all_genes' or 'all_pred' in condition: {condition}")
        # also get control means to use for deltas:
        if 'all_ctrl_means' in values:
            control_means_df = pd.DataFrame(values['all_ctrl_means'].transpose(), index=gene_names)
        else:
            if condition !='actual_mean_perturbed':
                print(f"Missing 'all_ctrl_means' in condition: {condition}")
    # Create DataFrames from the collected mean predictions and actual values
    preds_df = pd.concat(pred_dict, axis=1)
    preds_df.columns = [x.replace('+ctrl','').replace('ctrl+','') for x in preds_df.columns]
    actual_df = pd.concat(actual_dict, axis=1)
    actual_df.columns = [x.replace('+ctrl','').replace('ctrl+','') for x in actual_df.columns]
    return [preds_df, actual_df, control_means_df]

def clean_up_data(dataset, dataframe):
    """
    helper function written by Abby
    """
    if dataset=='replogle_k562_essential':
        # There are two rows (two genes in original data) with gene symbol TBCE. Want to rename to TBCE-0 and TBCE-1
        # find all indices named 'TBCE'
        tbce_indices = dataframe.index[dataframe.index == 'TBCE']
        # Create a new index with the renamed entries
        new_index = []
        tbce_count = 0
        for idx in dataframe.index:
            if idx == 'TBCE':
                new_index.append(f'TBCE-{tbce_count}')
                tbce_count += 1
            else:
                new_index.append(idx)
        # Assign the new index to the DataFrame
        dataframe.index = new_index
    return dataframe

def get_gene_level_metrics(preds_df, actual_df, gene_names, exclude_target_gene=False, exclude_combos=False):
    """
    helper function written by Abby - calculate metrics (Pearson correlation, MSE, R2) for each gene across perturbations
    """
    # Initialize lists to store results
    correlations = []
    p_values = []
    RMSE = []
    MSE = []
    R2 = []
    print(f'exclude combos: {exclude_combos}')
    # transpose dataframes for gene-level metrics
    preds_df = preds_df.transpose()
    actual_df = actual_df.transpose()
    # Loop through each gene name
    for gene_name in gene_names:
        if exclude_target_gene:
            # Exclude rows with index that matches the gene name
            current_preds_df = preds_df[preds_df.index != gene_name]
            current_actual_df = actual_df[actual_df.index != gene_name]
        else:
            current_preds_df = preds_df
            current_actual_df = actual_df
        if exclude_combos:
            # Get columns that do not have '+' in the column name
            filtered_rows = [row for row in current_preds_df.index if '+' not in row]
            # Create a new DataFrame with the filtered columns
            current_preds_df = current_preds_df[~current_preds_df.index.str.contains('\+')]
            current_actual_df = current_actual_df[~current_actual_df.index.str.contains('\+')]
        # Extract the relevant data for the specified gene
        if gene_name not in current_preds_df.columns:
            raise ValueError(f"Column {gene_name} not found in current_preds_df.")
        if gene_name not in current_actual_df.columns:
            raise ValueError(f"Column {gene_name} not found in current_actual_df.")
        actual_data = current_actual_df[gene_name]
        pred_data = current_preds_df[gene_name]
        # Calculate the Pearson correlation coefficient
        correlation, p_value = scipy.stats.pearsonr(actual_data, pred_data)
        # calculate (R)MSE
        mse = sklm.mean_squared_error(actual_data, pred_data)
        # depends on version of scikit-learn whether you need root_mean_squared_error or squared=False
        # rmse = sklm.root_mean_squared_error(actual_data, pred_data)
        rmse = sklm.mean_squared_error(actual_data, pred_data, squared=False)
        # calculate r-squared
        r2 = sklm.r2_score(actual_data, pred_data)
        # Append results to lists
        correlations.append(correlation)
        p_values.append(p_value)
        RMSE.append(rmse)
        MSE.append(mse)
        R2.append(r2)
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Pearson_correlation': correlations,
        'p_value': p_values,
        'RMSE': RMSE,
        'MSE': MSE,
        'R2': R2
    }, index=gene_names)
    return results_df

def get_perturbation_level_metrics(preds_df, actual_df, control_means_df, perturbation_labels, exclude_target_gene=False, exclude_combos=False):
    """
    helper function written by Abby - calculate metrics (Pearson correlation, Pearson delta, RMSE, MSE, R2) for each perturbation across all genes; comparable to previous publications
    """
    # Initialize lists to store results
    correlations = []
    delta_correlations = []
    RMSE = []
    MSE = []
    R2 = []
    delta_R2 = []
    if exclude_combos:
        perturbation_labels = [p for p in perturbation_labels if '+' not in p]
    # Loop through each gene name
    for perturbation in perturbation_labels:
        if exclude_target_gene:
            # Exclude rows with index that matches the gene name
            # this would need to be modified to work properly for combinations of perturbations
            current_preds_df = preds_df[~preds_df.index.str.contains(perturbation)]
            current_actual_df = actual_df[~actual_df.index.str.contains(perturbation)]
            current_control_means_df = control_means_df[~control_means_df.index.str.contains(perturbation)]
        else:
            current_preds_df = preds_df
            current_actual_df = actual_df
            current_control_means_df = control_means_df
        # get delta (difference between actual-mean of control cells or predicted-mean of control cells)
        current_preds_delta_df = current_preds_df.subtract(current_control_means_df[0], axis=0)
        current_actual_delta_df = current_actual_df.subtract(current_control_means_df[0], axis=0)
        if perturbation not in current_preds_df.columns or perturbation not in current_actual_df.columns:
            raise ValueError(f"Columns {perturbation} not found in the DataFrame.")
        actual_data = current_actual_df[perturbation]
        actual_delta = current_actual_delta_df[perturbation]
        pred_data = current_preds_df[perturbation]
        pred_delta = current_preds_delta_df[perturbation]
        # Calculate the Pearson correlation coefficient
        correlation, p_value = scipy.stats.pearsonr(actual_data, pred_data)
        delta_correlation, delta_p_value = scipy.stats.pearsonr(actual_delta, pred_delta)
        # calculate (R)MSE
        # depends on version of scikit learn which of the two below lines you should use
        # rmse = sklm.root_mean_squared_error(actual_data, pred_data)
        rmse = sklm.mean_squared_error(actual_data, pred_data, squared=False)
        mse = sklm.mean_squared_error(actual_data, pred_data)
        # calculate r-squared
        r2 = sklm.r2_score(actual_data, pred_data)
        delta_r2 = sklm.r2_score(actual_delta, pred_delta)
        # Append results to lists
        correlations.append(correlation)
        delta_correlations.append(delta_correlation)
        RMSE.append(rmse)
        MSE.append(mse)
        R2.append(r2)
        delta_R2.append(delta_r2)
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Pearson_correlation': correlations,
        'Pearson_delta': delta_correlations,
        'RMSE': RMSE,
        'MSE': MSE,
        'R2': R2,
        'delta_R2': delta_R2
    }, index=perturbation_labels)
    return results_df

def plot_condition_specific_performance():
    """
    Two plots:
    1) scatterplot where each point is a target with x = mean model pearson delta, y = scGPT pearson delta

    2)
    Plot performance of different perturbation conditions
    plot will be specific dataset pearson delta / pearson de delta
    x-axis: conditions 
    y-axis: scores
    """
    if not os.path.isdir("outputs/breakdown/"):
        os.makedirs("outputs/breakdown/")
    ##load condition map 
    metrics = ["pearson_delta", "pearson_de_delta"]
    places = ["Top 20", "Bottom 20"]
    models = {"gears": "GEARS", "scGPT": "scGPT Fully Fine-Tuned Baseline", "smart_mean_perturbed": "CRISPR-informed Mean"}
    color_map = {"gears": "#3B75AF", "smart_mean_perturbed": "goldenrod", "scGPT": "#519E3E"}
    metric_label_map = {"pearson": "Pearson", "pearson_de": "Pearson DE", "pearson_delta": "Pearson Delta", "pearson_de_delta": "Pearson DE Delta"}
    model_map = {model: {dataset: "" for dataset in datasets} for model in models}
    root_dirs = ["save/default_config_baseline/", "pickles/gears_results/"]
    best_models = find_best_models(root_dirs, mode=1)
    for dataset in datasets: 
        for model in models: 
            if model == "gears":
                best_gears_model = best_models["gears"][dataset][0]
                model_run_id = find_best_run_number(best_gears_model, "gears")
                # model_run_id = re.findall(r"[0-9]+.pkl", best_gears_model)[0].split(".")[0]
                model_map[model][dataset] = pickle.load(open(f"pickles/gears_results/gears_condition_specific_results_{dataset}_{model_run_id}.pkl", "rb"))
            else: ##scGPT eval done with front-running already
                model_map[model][dataset] = pickle.load(open(f"save/test_condition_specific_performance/{model}_condition_specific_results_{dataset}.pkl", "rb"))

    ##scatterplot where each point is a target with x = mean model pearson delta, y = scGPT pearson delta
    for dataset in datasets:
        for metric in ["pearson_delta"]:
            for model in ["scGPT", "gears"]:
                fig, ax = plt.subplots()
                x1, y1 = [], [] ##when mean model is better
                x2, y2 = [], [] ##when deep model is better
                x3, y3 = [], [] ##when equal
                # targets = []
                for target in model_map["smart_mean_perturbed"][dataset]:
                    deep_score = model_map[model][dataset][target][metric]
                    mean_score = model_map["smart_mean_perturbed"][dataset][target][metric]
                    if mean_score > deep_score:
                        x1.append(deep_score)
                        y1.append(mean_score)
                    elif deep_score > mean_score:
                        x2.append(deep_score)
                        y2.append(mean_score)
                        # targets.append(target)
                    else:
                        x3.append(deep_score)
                        y3.append(mean_score)
                print(dataset, targets)
                ax.scatter(x1, y1, color=color_map["smart_mean_perturbed"])
                ax.scatter(x2, y2, color=color_map[model])
                ax.scatter(x3, y3, color="grey")
                ##label the targets where deep did better
                # print(targets)
                # for i, label in enumerate(targets):
                #     if label in ["RPS5+ctrl", "RPL35A+ctrl"]: ##manually move some of these because they overlap too much and can't see them
                #         plt.annotate(label, (x2[i], y2[i] + 0.03), fontsize=4)
                #     else:
                #         plt.annotate(label, (x2[i], y2[i]), fontsize=4)
                ax.set_xlabel(models[model])
                ax.set_ylabel("CRISPR-informed Mean")
                ax.plot([-.3, 1], [-.3, 1], '--', alpha=0.75, color="black", zorder=0) ##plot x = y line 
                plt.xlim((-.3, 1.03))
                plt.ylim((-.3, 1.03))
                plt.title(f"{models[model]} vs CRISPR-informed Mean{metric_label_map[metric]}\nby Target for {get_dataset_title(dataset)}")
                plt.savefig(f"outputs/breakdown/scatter_{dataset}_{model}_{metric}.png", dpi=300)

    ##bar graphs of top 20 and bottom 20
    ##select CRISPR mean's top 20 and bottom 20 conditions per metric per study and keep consistent for all models 
    dataset_to_conditions = {place: {metric: {dataset: [] for dataset in datasets} for metric in metrics} for place in places}
    for dataset in model_map["smart_mean_perturbed"]:
        for metric in metrics: 
            n = 20
            items = sorted(model_map["smart_mean_perturbed"][dataset].items(), key=lambda x:x[1][metric])
            top_n = items[-1 * n:]
            bottom_n = items[0:n]
            dataset_to_conditions["Top 20"][metric][dataset] = [x[0] for x in top_n]
            dataset_to_conditions["Bottom 20"][metric][dataset] = [x[0] for x in bottom_n]
    ##make plots
    width = 0.15
    for metric in metrics:
        for dataset in datasets:
            for place in places: 
                fig, ax = plt.subplots()
                # x_labels = dataset_to_conditions[dataset]
                x_labels = dataset_to_conditions[place][metric][dataset]
                plt_x_labels = list(x_labels)
                x = np.array(range(0, len(x_labels)))
                ax.set_xticks(x)
                mean_superior_indices = []
                ##indices of x_labels where mean is the best, append them with a ** in plot
                for i in range(0, len(x_labels)):
                    x_label = x_labels[i]
                    if model_map["smart_mean_perturbed"][dataset][x_label][metric] > model_map["scGPT"][dataset][x_label][metric] \
                        and model_map["smart_mean_perturbed"][dataset][x_label][metric] > model_map["gears"][dataset][x_label][metric]:
                        mean_superior_indices.append(i)
                for m_s_i in mean_superior_indices:
                    plt_x_labels[m_s_i] = "** " + x_labels[m_s_i]
                ax.set_xticklabels(plt_x_labels, fontsize=7)
                plt.yticks(fontsize=7)  
                plt.xticks(rotation=90)
                ax.set_ylabel(metric_label_map[metric], fontsize=7)
                min_y = 1.0
                for model in models: 
                    y = [model_map[model][dataset][x_label][metric] for x_label in x_labels]
                    ax.bar(x, y, width=width, label=models[model], color=color_map[model])
                    x = x + width
                    min_y = min(min_y, min(y))
                plt.ylim((min_y - 0.1, 1.01))
                if dataset == "adam_corrected_upr": ##this dataset just has 20 test set perturbations
                    plt.title(f"Performance of All Test Set Conditions for {get_dataset_title(dataset)}", fontsize=9)
                else:
                    plt.title(f"Performance of CRISPR-informed Mean's {place}\nPerturbation Conditions for {get_dataset_title(dataset)}", fontsize=9)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
                ax.legend(loc='upper right', prop={"size":7}, bbox_to_anchor=(1, 1.38))
                plt.gcf().subplots_adjust(top=.78, bottom=.25)
                plt.savefig(f"outputs/breakdown/{place}_{dataset}_{metric}.png", dpi=300,  bbox_inches='tight')

def get_genes_sorted_by_std(dataset):
    pert_data = get_pert_data(dataset)
    std_map = {}
    X = pert_data.adata.X.toarray()
    gene_list = pert_data.adata.var["gene_name"].tolist()
    for i in range(0, len(gene_list)):
        gene = gene_list[i]
        std_map[gene] = np.std(X[:,i])
    sorted_map = sorted(std_map.items(), key=lambda x: x[1])
    return sorted_map

def plot_gene_specific_performance():
    """
    plot pearson of top 20 performing genes for scGPT, fully fine-tuned and GEARS, and CRISPR-informed mean
    """
    if not os.path.isdir("outputs/gene_breakdown/"):
        os.makedirs("outputs/gene_breakdown/")
    models = {"gears": "GEARS", "scGPT": "scGPT Fully Fine-Tuned Baseline", "smart_mean_perturbed": "CRISPR-informed Mean"}
    color_map = {"gears": "#3B75AF", "smart_mean_perturbed": "goldenrod", "scGPT": "#519E3E"}
    metric_label_map = {"pearson": "Pearson", "pearson_de": "Pearson DE", "pearson_delta": "Pearson Delta", "pearson_de_delta": "Pearson DE Delta"}
    places = ["Top 20", "Bottom 20"]
    model_map = {model: {dataset: {place: [] for place in places} for dataset in datasets} for model in models}
    root_dirs = ["save/default_config_baseline/", "pickles/gears_results/"]
    best_models = find_best_models(root_dirs, mode=1)
    for dataset in datasets: 
        for model in models: 
            if model == "gears":
                best_gears_model = best_models["gears"][dataset][0]
                model_run_id = find_best_run_number(best_gears_model, "gears")
                results = pickle.load(open(f"pickles/gears_results/gears_gene_specific_results_{dataset}_{model_run_id}.pkl", "rb"))
            else: ##scGPT eval done with front-running already
                results = pickle.load(open(f"save/test_condition_specific_performance/{model}_gene_specific_results_{dataset}.pkl", "rb"))
            ##filter out NaN pearsons 
            results = {key: results[key] for key in results if not np.isnan(results[key])}
            ##filter results to top 20 genes and bottom 20 genes
            sorted_results = sorted(results.items(), key=lambda x: x[1])
            model_map[model][dataset]["Bottom 20"] = sorted_results[0:20]
            model_map[model][dataset]["Top 20"] = sorted_results[-20:]
    
    for dataset in datasets:
        scgpt_set = set([x[0] for x in model_map["scGPT"][dataset]["Top 20"]])
        mean_set = set([x[0] for x in model_map["smart_mean_perturbed"][dataset]["Top 20"]])
        gears_set = set([x[0] for x in model_map["gears"][dataset]["Top 20"]])
        print("overlap between top 20 scGPT and top 20 mean model: ")
        print("    ", dataset, len(scgpt_set.intersection(mean_set)))
        print("overlap between top 20 GEARS and top 20 mean model: ")
        print("    ", dataset, len(gears_set.intersection(mean_set)))
        # sorted_genes_by_std = [x[0] for x in get_genes_sorted_by_std(dataset)]
        # for gene in scgpt_set: 
        #     print(f" {sorted_genes_by_std.index(gene)} / {len(sorted_genes_by_std)}")
    
    width = 0.15
    for dataset in datasets:
        for model in models:
            for place in places:
                fig, ax = plt.subplots()
                x_labels = [item[0] for item in model_map[model][dataset][place]]
                x = np.array(range(0, len(x_labels)))
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels)
                plt.xticks(rotation=90)
                y =  [item[1] for item in model_map[model][dataset][place]]
                ax.bar(x, y, width=width, label=models[model], color=color_map[model])
                ax.set_ylabel("Pearson Score")
                plt.ylim((min(y) - 0.1, 1.0))
                suffix = "Highest" if place == "Top 20" else "Lowest"
                plt.title(f"{place} {suffix} Performing Genes\nfor {models[model]} on {get_dataset_title(dataset)}", fontsize=11)
                # plt.gcf().subplots_adjust(bottom=.25)
                plt.gcf().subplots_adjust(top=.60, bottom=.40)
                plt.savefig(f"outputs/gene_breakdown/{model}_{dataset}_{place}.png", dpi=300, bbox_inches='tight')

def plot_cross_validation():
    x_label_map = {"pearson": "Pearson", "pearson_de": "Pearson DE", "pearson_delta": "Pearson Delta", "pearson_de_delta": "Pearson DE Delta"}
    x_labels = list(x_label_map.keys())
    models = {"gears": "GEARS", "baseline_mean_perturbed": "Mean", "smart_mean_perturbed": "CRISPR-informed Mean", "scGPT": "scGPT Fully Fine-Tuned Baseline", "linear_additive": "Linear Additive", "latent_additive": "Latent Additive", "decoder_only": "Decoder Only"}
    color_map = {"gears": "#3B75AF", "baseline_mean_perturbed": "salmon", "smart_mean_perturbed": "goldenrod", "scGPT": "#519E3E", "linear_additive": "dimgray", "latent_additive": "darkgrey", "decoder_only": "lightgrey"}
    paths = [] 
    model_results = {model: {dataset: "" for dataset in datasets} for model in models}
    ##get scGPT, mean, CRISPR-informed mean, and perturbench model cross val results
    for root, dirs, files in os.walk("save/cross_val/"):
        for file in files:
            if "_pert_delta_results" in file:
                paths.append(os.path.join(root, file))
    for dataset in datasets:
        for model in ["scGPT", "baseline_mean_perturbed", "smart_mean_perturbed"]:
            y_model, y_model_std = get_path_results(f"save/cross_val/scgpt_{dataset}_run_1/{model}_pert_delta_results_{dataset}.pkl", paths, x_label_map.keys(), mode=1)
            model_results[model][dataset] = y_model, y_model_std
        for model in ["linear_additive", "latent_additive", "decoder_only"]:
            y_model, y_model_std = get_path_results(f"save/cross_val/perturbench_{dataset}_run_1/{model}_pert_delta_results_{dataset}.pkl", paths, x_label_map.keys(), mode=1)
            model_results[model][dataset] = y_model, y_model_std
    ##get gears mean cross val results
    for root, dirs, files in os.walk("pickles/gears_results_cv/"):
        for file in files:
            if "_pert_delta_results" in file:
                paths.append(os.path.join(root, file))
    gears_map = {dataset: {x_label: [] for x_label in x_label_map} for dataset in datasets}
    for dataset in datasets: 
        for run_number in [1,2,3,4]:
            gears_res = pickle.load(open(f"pickles/gears_results_cv/gears_pert_delta_results_{dataset}_cross_val_{run_number}.pkl", "rb"))
            print(dataset, gears_res)
            for x_label in x_label_map:
                gears_map[dataset][x_label].append(gears_res[x_label])
    gears_map = {dataset: {x_label: (np.mean(gears_map[dataset][x_label]), np.std(gears_map[dataset][x_label])) for x_label in x_labels} for dataset in gears_map} 
    gears_map = {dataset: ([gears_map[dataset][x_label][0] for x_label in x_labels], [gears_map[dataset][x_label][1] for x_label in x_labels]) for dataset in gears_map}
    model_results["gears"] = gears_map
    ##now make plots
    width = 0.12
    for dataset in datasets:
        fig, ax = plt.subplots()
        x = np.array(range(0, len(x_labels)))
        plt_x_labels = [x_label_map[x] for x in x_label_map]
        ax.set_xticks(x)
        ax.set_xticklabels(plt_x_labels, fontsize=8.5)
        plt.xticks(rotation=15)
        for model in model_results:
            y = model_results[model][dataset][0]
            y_std = model_results[model][dataset][1]
            ax.bar(x, y, yerr=y_std, width=width, label=models[model], color=color_map[model], error_kw={"elinewidth":0.5, "capsize":0.5})
            for i,j in zip(x, y):
                ax.annotate(f"{round(j, 2)}", xy=(i - .06, j +.02),fontsize=4.5)
            x = x + width 
        plt.title(f"Cross-Validation Results for {get_dataset_title(dataset)}", fontsize=10)
        ax.set_ylabel("Pearson Score")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper right', prop={"size":6}, bbox_to_anchor=(1, 1.37))
        plt.gcf().subplots_adjust(top=.76)
        plt.savefig(f"outputs/cross_val_{dataset}.png", dpi=300)
   


shutil.rmtree("outputs/")       
os.mkdir("outputs/")  

#short run-times
get_avg_baseline(mode=1)
plot_perturbench_comparison(mode=1)
plot_subset_model_scores(mode=1)
plot_rank_scores(mode=1, include_perturbench=True)
plot_model_scores(mode=1)
plot_simple_affine_run_times()
compare_number_model_params()
plot_cross_validation() 
plot_cell_and_pert_counts()
plot_gene_counts()
plot_condition_specific_performance()
plot_gene_specific_performance()
make_metric_violinplots(dir="/hpfs/projects/mlcs/mlhub/perturbseq/scGPT_baseline_results/", output_prefix="outputs/", exclude_target_gene=False, exclude_combos=True)
make_metric_violinplots(dir="/hpfs/projects/mlcs/mlhub/perturbseq/scGPT_baseline_results/", output_prefix="outputs/", exclude_target_gene=False, exclude_combos=False)

##longer run-time
plot_avg_pearson_to_avg_perturbed_state()
plot_wasserstein_pert_gene_comparison()

#not for manuscript figure generation
# plot_subset_model_scores(mode=1, include_simple_affine=True) ##for checking statistical significance, actual plot is too crowded
# plot_model_losses()
# root_dirs = ["save/no_pretraining/", "save/default_config_baseline/", "save/simple_affine/", "save/simple_affine_with_pretraining/", "save/perturbench/", "pickles/gears_results/"]
# find_best_models(mode=1)