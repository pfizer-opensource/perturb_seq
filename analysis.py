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

datasets = ["adamson", "norman", "replogle_k562_essential"]

def get_dataset_title(string):
    ""
    m = {"adamson": "Adamson", "norman": "Norman", "replogle_k562_essential": "Replogle K562 Essential", "replogle_k562_gwps": "Replogle K562 GWPS"}
    if string not in m: 
        raise Exception(f"{string} not present in title map")
    else:
        return m[string]

def get_dataset(string):
    mapp = {"adamson": "adamson", "norman": "norman", "replogle":  "replogle_k562_essential", "combined": "combined"}
    for key_phrase in mapp:
        if key_phrase in string:
            return mapp[key_phrase]

def get_avg_baseline(mode):
    ##key: method, key: dataset, key: score metric, value: list of scores
    baseline_map = {"scGPT": {}, "gears": {}, "mean_control": {},
    "mean_perturbed": {}, "mean_control+perturbed": {},
    "smart_mean_control": {}, "smart_mean_perturbed": {}, "smart_mean_control+perturbed":{}}
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
                    for mean_type in ["control+perturbed", "control", "perturbed"]:
                        if mode == 1: 
                            mean_map = pickle.load(open(os.path.join(baseline_root, directory, f"{mean_baseline}_mean_{mean_type}_pert_delta_results_{dataset}.pkl"), "rb"))
                        else:
                            mean_map = pickle.load(open(os.path.join(baseline_root, directory, f"{mean_baseline}_mean_{mean_type}_results_{dataset}.pkl"), "rb"))
                        if dataset in baseline_map[f"{prefix}mean_{mean_type}"]:
                            assert(mean_map == baseline_map[f"{prefix}mean_{mean_type}"][dataset])
                        else:
                            baseline_map[f"{prefix}mean_{mean_type}"][dataset] = mean_map
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
    adamson_multirun_results = get_path_results("save/no_pretraining/adamson_run_1/scGPT_pert_delta_results_adamson.pkl", paths, x_labels, mode)
    norman_multirun_results = get_path_results("save/no_pretraining/norman_run_1/scGPT_pert_delta_results_norman.pkl", paths, x_labels, mode)
    replogle_multirun_results = get_path_results("save/no_pretraining/replogle_k562_essential_run_1/scGPT_pert_delta_results_replogle_k562_essential.pkl", paths, x_labels, mode)
    return {"adamson": adamson_multirun_results, "norman": norman_multirun_results, "replogle_k562_essential": replogle_multirun_results}

def get_model_title_from_path(string):
    if "no_pretraining" in string:
        return "scGPT (no-pretraining)"
    if "transformer_encoder_control" in string:
        return "scGPT (randomly initialized transformer encoder)"
    if "input_encoder_control" in string:
        return "scGPT (randomly initialized input encoder)"
    if "simple_affine" in string:
        return "Simple Affine (no transformer)"
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
    
def get_path_results(path, paths, x_labels, mode):
    """
    For a given path to results file:
    will return the score and std as list, one entry for each of x_labels
        if part of a mult-run: score will be the average across all runs
        if part of a singleton run: score will be the result of that one run, std will be 0 
    """
    ##get y_model and (if part of a multi-run experiment) y_model_std 
    if "_run_" in path: ##if this file path is part of a multi-set run, then find the other files and aggregate them into avg and std scores
        prefix = path[0: path.find("_run_")]
        if "perturbench" in path:
            suffix = path[path.find("_run_") + 7: ]
            same_run_paths = [p for p in paths if prefix in p and suffix in p]
        else:
            same_run_paths = [p for p in paths if prefix in p]
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
        sr_results = {key: (np.mean(sr_results[key]), np.std(sr_results[key])) for key in sr_results}
        y_model = [sr_results[key][0] for key in x_labels]
        y_model_std = [sr_results[key][1] for key in x_labels]
    else: ##singleton run               
        if mode == 1: 
            model_res = pickle.load(open(path, "rb"))
        else:
            model_res, _ = pickle.load(open(path, "rb"))
        y_model = [model_res[key] for key in x_labels]
        y_model_std = [0] * len(y_model)
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
        for file in files:
            if "baseline" in file: ##skip baseline file
                continue 
            model_type = get_model_type(os.path.join(root, file))
            if model_type not in ["scgpt", "simple_affine"]:
                continue
            if mode == 1 and f"pert_delta_results" in file:
                paths.append(os.path.join(root, file))
            if mode == 2 and f"_results" in file and "pert_delta" not in file:
                paths.append(os.path.join(root, file))
    ##iterate over file paths and plot the results
    for path in paths: 
        ##get dataset that results are for 
        dataset = get_dataset(path)
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
    Plot bar graphs for just a subset of the results of interest
    """
    dataset_map = get_baseline_dataset_map(mode)
    paths = []
    if include_simple_affine:
        root_dirs = ["save/no_pretraining", "save/transformer_encoder_control", "save/input_encoder_control", "save/simple_affine"]
    else:
        root_dirs = ["save/no_pretraining", "save/transformer_encoder_control", "save/input_encoder_control"]
    for root_dir in root_dirs: 
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                model_type = get_model_type(os.path.join(root, file))
                if model_type not in ["scgpt", "simple_affine"]:
                    continue 
                if mode == 1 and "pert_delta_results" in file:
                    paths.append(os.path.join(root, file))
                if mode == 2 and "_results" in file and "pert_delta" not in file:
                    paths.append(os.path.join(root, file))
    scgpt_perm_map = {perm: {dataset: "" for dataset in datasets} for perm in root_dirs} ##key scgpt permutation as directory path, key: dataset, value: (y_model, y_std)
    baseline_map = {dataset: "" for dataset in datasets} #key dataset: value: baseline_y_std_map (key: model, value: tuple(scores list, std list) corresponding to x_labels)
    ##iterate over file paths and fill out scgpt_perm_map
    for path in paths: 
        ##get dataset that results are for 
        dataset = get_dataset(path)
        ##get baseline data to plot
        x_labels = list(dataset_map[dataset]["gears"].keys())
        baseline_y_std_map = get_baseline_y_std_map(dataset_map, dataset, x_labels)
        baseline_map[dataset] = baseline_y_std_map
        ##get results for this path to plot
        y_model, y_model_std = get_path_results(path, paths, x_labels, mode)
        for key in scgpt_perm_map:
            if key in path: 
                assigned_key = key
                break
        if scgpt_perm_map[assigned_key][dataset] != "":
            assert(scgpt_perm_map[assigned_key][dataset] == (y_model, y_model_std)) ##this should be the same for paths of the same multi-run
        else:
            scgpt_perm_map[assigned_key][dataset] = (y_model, y_model_std)
    ##print model to model comparisons 
    compare_models_across_datasets(scgpt_perm_map, baseline_map, x_labels)
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
                "scGPT (no pre-training)": (scgpt_perm_map["save/no_pretraining"][dataset][0], scgpt_perm_map["save/no_pretraining"][dataset][1], "mediumpurple"),
                "scGPT (randomly initialized input encoder) ": (scgpt_perm_map["save/input_encoder_control"][dataset][0], scgpt_perm_map["save/input_encoder_control"][dataset][1], "purple"),
                "scGPT (randomly initialized transformer encoder) ": (scgpt_perm_map["save/transformer_encoder_control"][dataset][0], scgpt_perm_map["save/transformer_encoder_control"][dataset][1], "darkviolet")
                }
        if include_simple_affine:
            y_dict.update({"Simple Affine": (scgpt_perm_map["save/simple_affine"][dataset][0], scgpt_perm_map["save/simple_affine"][dataset][1], "grey") })
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

def compare_models_across_datasets(scgpt_perm_map, baseline_map, x_labels):
    """
    Helper method for plot_subset_model_scores, will do pairwise model performance comparisons and print the pairwise differences of averages
    """
    model_pairs_map = {} ##key: (model1, model2) always sorted tuple, key: metric, value: list of (model1 avg - model2 avg)
    for dataset in baseline_map:
        model_pairs = itertools.combinations(set(list(baseline_map[dataset].keys()) + list(scgpt_perm_map.keys())), 2)
        model_pairs = [sorted(x) for x in model_pairs]
        for i in range(0, len(x_labels)):
            x_label = x_labels[i]
            for m1, m2 in model_pairs:
                if (m1, m2) not in model_pairs_map:
                    model_pairs_map[(m1, m2)] = {x_l: [] for x_l in x_labels}
                if m1 in baseline_map[dataset].keys():
                    score1 = baseline_map[dataset][m1][0][i]
                if m1 in scgpt_perm_map.keys():
                    score1 = scgpt_perm_map[m1][dataset][0][i]
                if m2 in baseline_map[dataset].keys():
                    score2 =  baseline_map[dataset][m2][0][i]
                if m2 in scgpt_perm_map.keys():
                    score2 = scgpt_perm_map[m2][dataset][0][i]
                # model_pairs_map[(m1, m2)][x_label].append(baseline_map[dataset][m1][0][i] - baseline_map[dataset][m2][0][i])
                model_pairs_map[(m1, m2)][x_label].append(score1 - score2)
    ##drop pearson and pearson de
    for key in model_pairs_map:
        model_pairs_map[key] = {key2: model_pairs_map[key][key2] for key2 in model_pairs_map[key].keys() if key2 not in ["pearson", "pearson_de"]}
    for key in model_pairs_map:
        print(key)
        print("   ", model_pairs_map[key])
    ##consolidate to mean 
    for m1,m2 in model_pairs_map: 
        for metric in model_pairs_map[(m1,m2)]:
            model_pairs_map[(m1,m2)][metric] = np.mean(model_pairs_map[(m1,m2)][metric])
    for key in model_pairs_map:
        print(key)
        print("   ", model_pairs_map[key])

def plot_model_losses():
    paths = [] 
    for root, dirs, files in os.walk("save/"):
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
        
def compare_shuffle_to_not(mode):
    """
    Plot deltas of scores when we don't do random shuffling of perturbation indices vs when we do
    """
    if mode == 1: 
        x_labels = ["pearson", "pearson_de", "pearson_delta", "pearson_de_delta"] 
    if mode == 2:
        x_labels = ["mse", "mse_de", "pearson", "pearson_de"]
    shuffle_dict = {boolean: {dataset: {x_label: [] for x_label in x_labels} for dataset in datasets} for boolean in [True, False]}
    combined_paths = []
    root_dirs = ["save/default_config_baseline", "save/shuffled_pert_token"]
    for root_dir in root_dirs: 
        for  directory in os.listdir(root_dir):
            combined_paths.append(os.path.join(root_dir, directory))
    for save_dir in combined_paths:    
        shuffle_boolean = True if "shuffled" in save_dir else False 
        ##get dataset that results are for 
        dataset = get_dataset(save_dir)
        if mode == 1:
            res_file = os.path.join(save_dir, f"scGPT_pert_delta_results_{dataset}.pkl")
        if mode == 2:
            res_file = os.path.join(save_dir, f"scGPT_results_{dataset}.pkl")
        if os.path.isfile(res_file): 
            if mode == 1:
                model_res = pickle.load(open(res_file, "rb"))
            if mode == 2:
                model_res, _ = pickle.load(open(res_file, "rb"))
            ##write to shuffle_dict
            for key in x_labels:
                shuffle_dict[shuffle_boolean][dataset][key].append(model_res[key])
    ##consolidate lists to summary mean and std statistics
    for boolean in shuffle_dict:
        for dataset in shuffle_dict[boolean]:
            for metric in shuffle_dict[boolean][dataset]:
                shuffle_dict[boolean][dataset][metric] = np.mean(shuffle_dict[boolean][dataset][metric]), np.std(shuffle_dict[boolean][dataset][metric])
    ##plot delta plots, expect negative delta for MSEs and positive deltas for Pearsons
    fig, ax = plt.subplots()
    x = np.array(range(0, len(x_labels)))
    ax.set_xticks(x)
    width = 0.30
    for dataset in datasets:
        y = [shuffle_dict[False][dataset][metric][0] - shuffle_dict[True][dataset][metric][0] for metric in x_labels] ##the delta
        ax.bar(x, y, width=width, label=f"{dataset.capitalize()} Dataset Delta")
        for i,j in zip(x, y):
            ax.annotate("{:.02}".format(j), xy=(i - .05, j),fontsize=5)
        x = x + width
    ax.hlines(y=0.0, xmin=0, xmax=len(x_labels), linestyles="dashed", color="black", linewidth=0.5)
    plt.title("Effect on Performance of Randomly Shuffling Perturbation Indices")
    ax.set_xticklabels(x_labels, fontsize=8)
    plt.xticks(rotation=15)
    ax.set_ylabel("Delta Scores (no shuffling - shuffling)")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"size":7}, bbox_to_anchor=(1, 1.32))
    plt.gcf().subplots_adjust(top=.76)
    plt.savefig(f"outputs/random_shuffle_control_{mode}.png", dpi=300)

def plot_cell_and_pert_counts():
    splits = ["train", "val", "test"]
    metrics = ["data_size", "perturbations"]
    gene_set_dict = {data_name: set() for data_name in datasets} 
    perturbation_dict = {data_name: {split: set() for split in splits} for data_name in datasets}
    data_size_dict = {data_name: {split: -1 for split in splits} for data_name in datasets}
    for data_name in datasets:
        if data_name in ["adamson", "norman", "replogle_k562_essential"]: 
            pert_data = PertData("./data")
            pert_data.load(data_name=data_name) ##seems to instantiate a lot of PertData attributes
            pert_data.prepare_split(split="simulation", seed=1)
            pert_data.get_dataloader(batch_size=64, test_batch_size=64)
        if data_name == "replogle_k562_gwps":
            pert_data = get_replogle_gwps_pert_data(split="simulation", batch_size=64, test_batch_size=64, generate_new=False)
        if "replogle" in data_name:
            modify_pertdata_dataloaders(pert_data, logger=None)
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

def test_x_y_dataloaders():
    """
    For each dataset dataloader, make sure each array in .x and .y are what we think they are:
    for control x_i = y_i, for perturbed check if array is present in adata.X.A and perturbation matches
    """
    for data_name in datasets:
        if data_name in ["adamson", "norman", "replogle_k562_essential"]: 
            pert_data = PertData("./data")
            pert_data.load(data_name=data_name) ##seems to instantiate a lot of PertData attributes
            pert_data.prepare_split(split="simulation", seed=1)
            pert_data.get_dataloader(batch_size=64, test_batch_size=64)
        if data_name == "replogle_k562_gwps":
            pert_data = get_replogle_gwps_pert_data(split="simulation", batch_size=64, test_batch_size=64, generate_new=False)
        if "replogle" in data_name:
            modify_pertdata_dataloaders(pert_data, logger=None)
        for batch, batch_data in enumerate(pert_data.dataloader["test_loader"]):
            batch_size = len(batch_data.y)
            x = batch_data.x  # (batch_size * n_genes, 2)  
            ori_gene_values = x[:, 0].view(batch_size, len(batch_data.y[0]))
            for i in range(0, len(batch_data)):
                if batch_data.pert[i] == "ctrl":
                    assert(torch.equal(ori_gene_values[i], batch_data.y[i])) ##for control, x_i == y_i
                    sc_expr = ori_gene_values[i].numpy()
                    found = False
                    for j, array in enumerate(pert_data.adata.X.A):
                        if np.array_equal(array, sc_expr):
                            assert(pert_data.adata.obs["condition"][j] == 'ctrl') ##make sure array is found and that corresponding condition in adata == ctrl 
                            found = True
                            break
                    if found == False:
                        raise Exception("not found")
                else:
                    sc_expr = batch_data.y[i].numpy()
                    found = False
                    for j, array in enumerate(pert_data.adata.X.A):
                        if np.array_equal(array, sc_expr):
                            assert(pert_data.adata.obs["condition"][j] != "ctrl")
                            assert(batch_data.pert[i] in pert_data.adata.obs["condition"][j]) ##make sure condition matches
                            found = True
                            break
                    if found == False:
                        raise Exception("not found")

def find_best_models(mode=1):
    ##find all paths to result files within save, check if part of a multi-run 
    model_types = ["scgpt", "simple_affine", "gears", "linear_additive", "latent_additive", "decoder_only"]
    paths = [] 
    root_dirs = ["save/default_config_baseline", "save/simple_affine", "pickles/gears_results/", "save/perturbench/"]
    for root_dir in root_dirs:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
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
        model_type = get_model_type(path)
        if mode == 1: 
            model_res = pickle.load(open(path, "rb"))
            score = model_res["pearson_de_delta"]
        else:
            model_res, _ = pickle.load(open(path, "rb"))
            score = model_res["pearson"]
        
        if score > best_map[model_type][dataset][1]:
            best_map[model_type][dataset] = (path, score)
    return best_map

def plot_wasserstein_pert_gene_comparison():
    """
    Compares two wasserstein distance distributions: 
    1: target gene T:              expression of target gene in cells perturbed by query  <--> expression of target gene in cells perturbed by something other than query 
    2: de genes != target gene:  expression of de genes != target gene for cells perturbed by query  <--> expression of de genes != target gene for cells perturbed by something other than query 
    """
    dataset_to_w = {data_name: () for data_name in datasets} ##key: dataset, value: (w1 mean, w1 std, w1 len, w2 mean, w2 std, w2 len)
    ##iterate over each dataset and compute wassersteins from test set 
    for data_name in datasets: 
        pert_data = PertData("./data")
        pert_data.load(data_name=data_name)
        pert_data.prepare_split(split="simulation", seed=1)
        pert_data.get_dataloader(batch_size=64, test_batch_size=64)
        if "replogle" in data_name:
            modify_pertdata_anndata(pert_data)
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
    color_map = {"adamson": "lightsteelblue", "norman": "tan", "replogle_k562_essential": "slategrey"}
    spacer = -0.4
    widths = 0.3
    for data_name in datasets: 
        ##make into boxplots
        y = [dataset_to_w[data_name][0], dataset_to_w[data_name][1]]
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
    dataset_map = {dataset: "" for dataset in datasets}
    for dataset in datasets: 
        pert_data = PertData("./data")
        pert_data.load(data_name=dataset)
        pert_data.prepare_split(split="simulation", seed=1)
        pert_data.get_dataloader(batch_size=64, test_batch_size=64)
        if "replogle" in dataset:
            modify_pertdata_anndata(pert_data)
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
    best_models = find_best_models(mode=mode)
    model_types = ["gears", "mean_perturbed", "smart_mean_perturbed", "scgpt", "linear_additive", "latent_additive", "decoder_only"]
    proper_x_labels = {"gears": "GEARS", "scgpt": "scGPT", "mean_perturbed": "Mean", "smart_mean_perturbed": "CRISPR-informed\nMean", "linear_additive": "Linear Additive", "latent_additive": "Latent Additive", "decoder_only": "Decoder Only"}
    width = 0.15
    color_map = {"adamson": "lightsteelblue", "norman": "tan", "replogle_k562_essential": "slategrey"}
    spacer = -0.4
    widths = 0.3
    scgpt_vs_smart = []
    gears_vs_smart = []
    for dataset in datasets:
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
    model_map = {dataset: {"Simple Affine": ["save/simple_affine/", []], "scGPT": ["save/default_config_baseline/", []]} for dataset in datasets}
    for dataset in model_map: 
        for model in model_map[dataset]:
            for root, dirs, files in os.walk(model_map[dataset][model][0]):
                for file in files:
                    if ".log" in file and dataset in root:
                        with open(os.path.join(root, file)) as file:
                            for line in file:
                                if "time: " in line:
                                    elapsed_time = float(re.findall(r"time: [0-9]+.[0-9]+s", line)[0].split("time: ")[1].replace("s", ""))
                                    model_map[dataset][model][1].append(elapsed_time)
    ##codense to avg and std 
    for dataset in model_map:
        for model in model_map[dataset]:
            model_map[dataset][model][1] = (np.mean(model_map[dataset][model][1]), np.std(model_map[dataset][model][1]))
    print(model_map)
    for dataset in model_map:
        print(f"{dataset}: {model_map[dataset]['scGPT'][1][0]  / model_map[dataset]['Simple Affine'][1][0]}")
    ##make bar graph comparisons, x-axis = model, y-axis = time, legend by dataset
    fig, ax = plt.subplots()
    width = 0.15
    x = np.array([1,2])
    ax.set_xticks(x)
    x_labels = ["Simple Affine", "scGPT"]
    ax.set_xticklabels(x_labels)
    color_map = {"adamson": "lightsteelblue", "norman": "tan", "replogle_k562_essential": "slategrey"}
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
    ax.bar(x, y, width=width, color=["grey", "#519E3E"])
    for i,j in zip(x, y):
        ax.annotate(f"{j:.2E}", xy=(i - .06, j + 800000),fontsize=7)
    plt.title(f"Trainable Parameters Comparsion")
    ax.set_ylabel("Number of Parameters")
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

shutil.rmtree("outputs/")       
os.mkdir("outputs/")  

get_avg_baseline(mode=1)
plot_model_scores(mode=1)
plot_subset_model_scores(mode=1)
plot_perturbench_comparison(mode=1)
plot_rank_scores(mode=1, include_perturbench=True)
plot_cell_and_pert_counts()
plot_simple_affine_run_times()
compare_number_model_params()
plot_avg_pearson_to_avg_perturbed_state()
plot_wasserstein_pert_gene_comparison()
find_best_models(mode=1)
plot_model_losses()
compare_shuffle_to_not(mode=1)
test_x_y_dataloaders()