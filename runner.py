"""
Main runner script for training, evaluating, and analyzing results
Much of the core scGPT code is based off the scGPT authors' tutorial: https://github.com/bowang-lab/scGPT/blob/7301b51a72f5db321fccebb51bc4dd1380d99023/tutorials/Tutorial_Perturbation.ipynb#L831
"""
from library import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test", help="train or test")
    parser.add_argument("--random_shuffle", type=bool_flag, default=False, help="True if want to randomly shuffle pert_flag, else False, used for control testing")
    parser.add_argument("--data_name", type=str, default="adamson", help="which dataset to use adamson, norman, replogle_k562_essential")
    parser.add_argument("--load_model", type=str, default="models/scgpt-pretrained/scGPT_human", help="which pretrained model to load")
    parser.add_argument("--filter_perturbations", type=bool_flag, default=False, help="True if want to remove perturbated cells that have a perturbation NOT part of the gene set, else False")
    parser.add_argument("--transformer_encoder_control", type=bool_flag, default=False, help="True if want to intentionally NOT load any pre-trained transformer encoder weights prior to training, else False, used for control testing")
    parser.add_argument("--attention_control", type=bool_flag, default=False, help="True if want to intentionally NOT load the pre-trained self attention weights prior to training, else False, used for control testing")
    parser.add_argument("--input_encoder_control", type=bool_flag, default=False, help="True if want to intentionally NOT load the pre-trained input encoding weights (gene encoder + expression encoder) prior to training, else False, used for control testing")
    parser.add_argument("--pretrain_control", type=bool_flag, default=False, help="True if want to intentionally NOT load any pre-trained weights prior to training, else False, used for control testing")
    parser.add_argument("--save_dir", type=str, default="default", help="set to 'default' if want the default save_dir, else set to specific path")
    parser.add_argument("--always_keep_pert_gene", type=bool_flag, default=False, help="True if we want to always inform the model of which gene was perturbed during training")
    parser.add_argument("--freeze_input_encoder", type=bool_flag, default=False, help="True if we want to freeze the input encoder weights during training - just the gene and expression encoder, leave the perturbation encoder unfrozen")
    parser.add_argument("--freeze_transformer_encoder", type=bool_flag, default=False, help="True if we want to freeze the transformer encoder weights during training")
    parser.add_argument("--use_lora", type=bool_flag, default=False, help="True if we want to use LoRa for finetuning")
    parser.add_argument("--lora_rank", type=int, default=8, help="if use_lora, specifies the inner dimension of the low-rank matrices to train")
    parser.add_argument("--config_path", type=str, default="config/default_config.json", help="path to JSON configuration file to use for setting up model")
    parser.add_argument("--model_type", type=str, default="scGPT", help="scGPT, simple_affine, mean_control, mean_perturbed, mean_control+perturbed, smart_mean_control, smart_mean_perturbed, smart_mean_control+perturbed")
    parser.add_argument("--validation_selection", type=str, default="pearson", help="how to select the best model during training, if 'pearson' will be by pearson correlation between predicted and actual expression over validation set, if 'loss' will be by minimal loss")
    parser.add_argument("--loss_type", type=str, default="mse", help="mse, mse+triplet, mse+pearson")
    parser.add_argument("--fixed_seed", type=bool_flag, default=True, help="True if we want to use a constant fixed seed")
    parser.add_argument("--optimal_pairings", type=str, default="None", help=" if we want to rearrange the training data loader to be control/perturbed matched instead of the default fixed random pairing, then set as either 'distance' or 'optimal_transport' else set as 'None'")
    parser.add_argument("--optimal_distance_metric", type=str, default="None", help="if using optimal_pairings, will determine what metric to use for distance: 'sqeuclidean', 'cosine', 'correlation'")
    parser.add_argument("--optimal_use_all_destination", type=bool, default=True, help="if using optimal_pairings, True will make sure all destination points are part of a pairing in the training / val set")
    parser.add_argument("--optimal_space", type=str, default="raw", help="if using optimal_pairings, space to perform distance measures, 'raw', 'pca', or 'umap' space")

    opt = parser.parse_args()
    check_args(opt)
    matplotlib.rcParams["savefig.transparent"] = False
    if opt.fixed_seed: 
        set_seed(42)
    else:
        set_seed(int(time.perf_counter())) ##introduce weight initialization variability across runs

    ##var values will depend on load_model and mode 
    var = get_variables(load_model=opt.load_model, config_path=opt.config_path)

    ##set up save dir and logger 
    if opt.save_dir == "default":
        save_dir = Path(f"./save/random_shuffle={opt.random_shuffle}/dev_perturb_{opt.data_name}-{time.strftime('%b%d-%H-%M')}/")
    else:
        save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    # log command line arguments 
    logger.info(f"{opt}")
    # log running date and current git commit
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"saving to {save_dir}")
    logger.info(f"var: {var}")

    ##setup PertData object
    if opt.data_name == "replogle_k562_gwps":
        pert_data = get_replogle_gwps_pert_data(split=var["split"], batch_size=var["batch_size"], test_batch_size=var["eval_batch_size"], generate_new=False)
    if opt.data_name in ["adamson", "norman", "replogle_k562_essential"]: 
        pert_data = PertData("./data")
        pert_data.load(data_name=opt.data_name) ##seems to instantiate a lot of PertData attributes
        pert_data.prepare_split(split=var["split"], seed=1)
        pert_data.get_dataloader(batch_size=var["batch_size"], test_batch_size=var["eval_batch_size"])
    if opt.filter_perturbations:
        logger.info("WARNING: filtering dataloaders! but keeping pert_data.adata the same")
        modify_pertdata_dataloaders(pert_data, logger)

    if opt.optimal_pairings != 'None':
        load_types = ["val", "train"]
        print(f"reconfiguring {load_types} loader(s) with optimal pairings")
        make_optimal_loaders(pert_data, optimal_pairings=opt.optimal_pairings, load_types=load_types, metric=opt.optimal_distance_metric, plot=False, use_all_destination=opt.optimal_use_all_destination, optimal_space=opt.optimal_space) ##just change the train/val loader for sake of fair evaluation 
    check_pert_split(opt.data_name, pert_data)

    logger.info(f"adata.obs: {pert_data.adata.obs}")
    logger.info(f"|conditions|: {len(set(pert_data.adata.obs['condition']))}")
    gene_name_list = pert_data.adata.var["gene_name"].tolist()
    gene_idx_map = {gene: gene_name_list.index(gene) for gene in gene_name_list} ##add dictionary from gene_name to index for passing to train to fix bug left by scGPT authors to get perturbation index (they don't account for latest version of GEARS loaders)
    logger.info(f"|gene_name_list|: {len(set(gene_name_list))}")

    model_file, vocab, n_genes, gene_ids, ntokens = get_model_setup(var, pert_data, logger)

    ##baselines from perturBench 
    if opt.mode in ["benchmark"]:
        encoder_input_dim = len(pert_data.adata.var["gene_name"].tolist()) ##want all the genes for these baseline models
        perturbench_models = {"linear_additive": LinearAdditive(encoder_input_dim, pert_data, var), "latent_additive": LatentAdditive(encoder_input_dim, pert_data, var),  "decoder_only": DecoderOnly(encoder_input_dim, pert_data, var)}
        logger.info(f"perturbench models: ")
        for perturbench_model in perturbench_models:
            p_model = perturbench_models[perturbench_model]
            p_model.to(var["device"])
            p_model = p_model.train_model(pert_data.dataloader["train_loader"], pert_data.dataloader["val_loader"], var, gene_ids)
            p_res = eval_perturb(pert_data.dataloader["test_loader"], p_model, gene_ids=gene_ids, gene_idx_map={}, var=var, loss_type=opt.loss_type) ##keep on cpu, no need to shuttle to gpu  
            p_metrics, p_metrics_pert  = compute_metrics(p_res)
            logger.info(f"test metrics: {perturbench_model}")
            pickle.dump((p_metrics, p_metrics_pert), open(save_dir / f"{perturbench_model}_results_{opt.data_name}.pkl", "wb"))
            p_metrics = compute_perturbation_metrics(p_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"])
            logger.info(f"{opt.data_name} {perturbench_model} delta test metrics: {p_metrics}")
            pickle.dump(p_metrics, open(save_dir / f"{perturbench_model}_pert_delta_results_{opt.data_name}.pkl", "wb"))
            test_perts = pickle.load(open(f"pickles/{opt.data_name}_perturbation_splits.pkl", "rb"))["test"]       
            ranks = get_rank(p_model, test_perts, pert_data=pert_data, var=var, gene_ids=gene_ids, gene_idx_map=gene_idx_map)
            pickle.dump(ranks, open(save_dir / f"rank_metrics_{opt.data_name}_{perturbench_model}.pkl", "wb"))
    
    ##mean predictor models - compute after data loaders are set (after a possible application of filter_perturbations)
    if opt.mode in ["train", "test"]:
        for baseline in ["baseline", "smart"]:
            for mean_type in ["control+perturbed", "control", "perturbed"]:
                ##baseline mean 
                if baseline == "baseline":
                    mean_pred_model = MeanPredictor(pert_data, opt.data_name, mean_type=mean_type)
                if baseline == "smart":
                    mean_pred_model = SmartMeanPredictor(pert_data, opt.data_name, mean_type=mean_type, crispr_type="crispra" if opt.data_name in ["norman"] else "crispri")
                # mean_pred_model.test_ordering(pert_data.dataloader["test_loader"])
                mean_res = eval_perturb(pert_data.dataloader["test_loader"], mean_pred_model, gene_ids=[], gene_idx_map={}, var={"device":"cpu"}, loss_type=opt.loss_type) ##keep on cpu, no need to shuttle to gpu for mean pred model 
                ##GEARS-type metrics
                mean_metrics, mean_metrics_pert  = compute_metrics(mean_res) ##from GEARS library
                logger.info(f"test metrics: ")
                pickle.dump((mean_metrics, mean_metrics_pert), open(save_dir / f"{baseline}_mean_{mean_type}_results_{opt.data_name}.pkl", "wb"))
                ##scGPT-type metrics
                mean_metrics = compute_perturbation_metrics(mean_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]) ##from scGPT library
                logger.info(f"{opt.data_name} {baseline} mean {mean_type} delta test metrics: {mean_metrics}")
                pickle.dump(mean_metrics, open(save_dir / f"{baseline}_mean_{mean_type}_pert_delta_results_{opt.data_name}.pkl", "wb"))
    
    if opt.model_type == "scGPT": 
        model = TransformerGenerator(
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
    
    elif opt.model_type == "simple_affine":
        from simple_affine import SimpleAffine 
        model = SimpleAffine(
            ntoken=ntokens,
            d_model=var["embsize"],
            nlayers=var["nlayers"],
            nlayers_cls=var["n_layers_cls"],
            vocab=vocab,
            dropout=var["dropout"],
            pad_token=var["pad_token"],
            pert_pad_id=var["pert_pad_id"],
        )
    elif "mean" in opt.model_type:
        if "smart" in opt.model_type:
            mean_type = opt.model_type.split("smart_mean_")[1]
            model = SmartMeanPredictor(pert_data, opt.data_name, mean_type=mean_type, crispr_type="crispra" if opt.data_name in ["norman"] else "crispri")
        else:
            mean_type = opt.model_type.split("_")[1]
            model = MeanPredictor(pert_data, opt.data_name, mean_type=mean_type)
    else:
        raise Exception("model_type must be one of scGPT, simple_affine, mean_control, mean_perturbed, mean_control+perturbed, smart_mean_control, smart_mean_perturbed, smart_mean_control+perturbed")

    if opt.model_type in ["scGPT", "simple_affine"]:
        model = load_model(var, model, model_file, logger, attention_control=opt.attention_control, freeze_input_encoder=opt.freeze_input_encoder, freeze_transformer_encoder=opt.freeze_transformer_encoder, mode=opt.mode, use_lora=opt.use_lora, lora_rank=opt.lora_rank, pretrain_control=opt.pretrain_control, transformer_encoder_control=opt.transformer_encoder_control, input_encoder_control=opt.input_encoder_control)
        model.to(var["device"])

    if opt.mode == "train":
        loss_map = {"train": {}, "val": {}}
        criterion = masked_mse_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=var["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, var["schedule_interval"], gamma=0.9)
        scaler = torch.cuda.amp.GradScaler(enabled=var["amp"])
        best_val_loss = float("inf")
        best_model = None 
        best_val_score = -100000000000 
        patience = 0
        for epoch in range(1, var["epochs"] + 1):
            print(f'epoch: {epoch} RAM used: {psutil.virtual_memory()[2]}%, {psutil.virtual_memory()[3]/1000000000}')
            epoch_start_time = time.time()
            train_loader = pert_data.dataloader["train_loader"]
            valid_loader = pert_data.dataloader["val_loader"]
            train_loss = train(
                model=model,
                train_loader=train_loader,
                n_genes=n_genes,
                gene_ids=gene_ids,
                criterion=criterion,
                scaler=scaler,
                optimizer=optimizer, 
                scheduler=scheduler,
                logger=logger,
                epoch=epoch,
                gene_idx_map=gene_idx_map,
                random_shuffle=opt.random_shuffle,
                always_keep_pert_gene=opt.always_keep_pert_gene, 
                loss_type=opt.loss_type,
                var=var
            )
            val_res = eval_perturb(valid_loader, model, gene_ids, gene_idx_map, var, loss_type=opt.loss_type)
            loss_map = update_loss_map(loss_map, train_loss, val_res)
            val_metrics = compute_perturbation_metrics(val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"])
            logger.info(f"val_metrics at epoch {epoch}: ")
            logger.info(val_metrics)
            logger.info(f"    avg val loss: {val_res['avg_loss']}")

            elapsed = time.time() - epoch_start_time
            logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

            if opt.validation_selection == "pearson": ##default: select model based on highest pearson score over all genes
                val_score = val_metrics["pearson"]
            if opt.validation_selection == "loss": ##select model based on what gives lowest loss in the validation set
                val_score = 1.0 / val_res["avg_loss"] ##invert: best model will have inverse furthest to the right on the (+) number line  

            if val_score > best_val_score:
                best_val_score = val_score
                best_model = copy.deepcopy(model)
                logger.info(f"Best model with score {val_score:5.4f}")
                patience = 0
            else:
                patience += 1
                if patience >= var["early_stop"]:
                    logger.info(f"Early stop at epoch {epoch}")
                    break
            scheduler.step()
        torch.save(best_model.state_dict(), save_dir / "best_model.pt")
        pickle.dump(loss_map, open(save_dir / f"loss_map_{opt.data_name}.pkl", "wb"))
        logger.info(f"loss_map: {loss_map}")
    
    if opt.mode in ["test", "analysis"]:
        best_model = model

    if opt.mode in ["train", "test"]: ##test model always for both mode == train or test
        test_res = eval_perturb(pert_data.dataloader["test_loader"], best_model, gene_ids, gene_idx_map, var, loss_type=opt.loss_type)
        ##GEARS-type metrics
        metrics, metrics_pert  = compute_metrics(test_res) ##from GEARS library
        logger.info(f"test metrics: ")
        pickle.dump((metrics, metrics_pert), open(save_dir / f"{opt.model_type}_results_{opt.data_name}.pkl", "wb"))
        ##scGPT-type metrics
        test_metrics = compute_perturbation_metrics(test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]) ##from scGPT utils library
        logger.info(f"{opt.data_name} delta test metrics: {test_metrics}")
        pickle.dump(test_metrics, open(save_dir / f"{opt.model_type}_pert_delta_results_{opt.data_name}.pkl", "wb"))

    if opt.mode == "analysis":    
        for plot_type in ["boxplots", "scatterplots"]:
            if not os.path.isdir(f"figures/{plot_type}/{opt.data_name}/{opt.model_type}"):
                os.makedirs(f"figures/{plot_type}/{opt.data_name}/{opt.model_type}")
        test_perts = pickle.load(open(f"pickles/{opt.data_name}_perturbation_splits.pkl", "rb"))["test"]       
        
        ##get rank metrics 
        ranks = get_rank(best_model, test_perts, pert_data=pert_data, var=var, gene_ids=gene_ids, gene_idx_map=gene_idx_map)
        pickle.dump(ranks, open(f"pickles/rank_metrics_{opt.data_name}_{opt.model_type}.pkl", "wb"))
       
        ##make boxplots
        pert_to_gene_iqr = plot_perturbation_boxplots(best_model, test_perts, save_directory=f"figures/boxplots/{opt.data_name}/{opt.model_type}/", pert_data=pert_data, var=var, gene_ids=gene_ids, gene_idx_map=gene_idx_map, data_name=opt.data_name, model_type=opt.model_type)
        for plot_differential in [True, False]:
            pearson_string = "pearson de" if plot_differential == False else "pearson de delta"
            print(f"{opt.data_name}/{opt.model_type} {pearson_string} mean/std: ", np.mean(list([x[0] for x in list(pert_to_gene_iqr[f"control_differential={plot_differential}"].values())])), np.std(list(x[0] for x in list(pert_to_gene_iqr[f"control_differential={plot_differential}"].values()))))
            print(f"{opt.data_name}/{opt.model_type} Wasserstein distance mean/std: ", np.mean(list([x[1] for x in list(pert_to_gene_iqr[f"control_differential={plot_differential}"].values())])), np.std(list(x[1] for x in list(pert_to_gene_iqr[f"control_differential={plot_differential}"].values()))))
        if not os.path.isdir("pickles/pert_to_gene_iqr/"):
            os.makedirs("pickles/pert_to_gene_iqr/")
        pickle.dump(pert_to_gene_iqr, open(f"pickles/pert_to_gene_iqr/pert_to_gene_iqr_{opt.data_name}_{opt.model_type}.pkl", "wb"))

        # ##make scatterplots
        # for p in test_perts:
        #     plot_perturbation_scatterplots(best_model, p, save_directory=f"figures/scatterplots/{opt.data_name}/{opt.model_type}/", pert_data=pert_data, var=var, gene_ids=gene_ids, gene_idx_map=gene_idx_map, model_type=opt.model_type)
        #convenience data structure for Rob and Abby 
        # perturbation_structure = get_complete_data_structure_for_perturbation(best_model, test_perts, pert_data=pert_data, var=var, gene_ids=gene_ids, gene_idx_map=gene_idx_map, data_name=opt.data_name, model_type=opt.model_type)
        # pickle.dump(perturbation_structure, open(f"pickles/pert_data_structure_{opt.data_name}_{opt.model_type}_load_{opt.load_model.replace('/', '_')}.pkl", "wb"))

if __name__ == "__main__":
    main()
