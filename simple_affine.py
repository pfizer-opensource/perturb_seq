from scgpt.model.generation_model import *

class SimpleAffine(nn.Module):
    """
    Equivalent model architecture to scGPT minus the transformer blocks
    """
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nlayers: int,
        nlayers_cls: int,
        vocab: Any,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pert_pad_id: int = 2,
        decoder_activation: Optional[str] = None,
        decoder_adaptive_bias: bool = False,
        explicit_zero_prob: bool = False,
    ):
        super().__init__()
        self.model_type = "SimpleAffine"
        self.d_model = d_model
        self.pad_token_id = vocab[pad_token]
        self.pert_pad_id = pert_pad_id
        self.explicit_zero_prob = explicit_zero_prob
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        self.pert_encoder = nn.Embedding(3, d_model, padding_idx=pert_pad_id)
        self.decoder = AffineExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
            activation=decoder_activation,
            adaptive_bias=decoder_adaptive_bias,
        )

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags,
        src_key_padding_mask: Tensor,
    ) -> Tensor:
        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src
        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        perts = self.pert_encoder(input_pert_flags)  # (batch, seq_len, embsize)
        batch = 0
        index = (input_pert_flags[batch] == 2).nonzero(as_tuple=True)[0]
        total_embs = src + values + perts
        return total_embs  # (batch, seq_len, embsize)

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags: Tensor,
        src_key_padding_mask: Tensor,
        do_sample: bool = False,
        CLS: bool = False, ##CLS, CCE, MVC, and ECS are all here for the sake of consistency with the transformer forward definition, but won't do anything with these args
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
        Returns:
            dict of output Tensors.
        """
        processed_values = values
        encoder_output = self._encode(
            src, processed_values, input_pert_flags, src_key_padding_mask
        )
        output = {}
        mlm_output = self.decoder(encoder_output, values)
        output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        return output

    def pred_perturb(
        self,
        batch_data,
        gene_ids=None,
        gene_idx_map=None, 
        var=None
    ) -> Tensor:
        """
        Args:
            batch_data: a dictionary of input data with keys.

        Returns:
            output Tensor of shape [N, seq_len]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_data.to(device)
        batch_size = len(batch_data.pert)
        x: torch.Tensor = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, -1)  # (batch_size, n_genes)
        pert_flags = get_pert_flags(batch_data, device, batch_size, len(gene_ids), gene_idx_map, random_shuffle=False, pert_pad_id=var["pert_pad_id"], not_perturbed_id=var["not_perturbed_id"], is_perturbed_id=var["is_perturbed_id"]) ##do not shuffle on validation ever - if testing random control condition, the model should be trained on shuffled flags already, no need to shuffle eval flags
        assert gene_ids is not None
        if var["include_zero_gene"] == "all":
            input_gene_ids = torch.arange(ori_gene_values.size(1), device=device) ##range(0, # of genes)
        else:  # batch-wise
            input_gene_ids = (ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0])
        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]
        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
        src_key_padding_mask = torch.zeros_like(
            input_values, dtype=torch.bool, device=device
        )
        with torch.cuda.amp.autocast(enabled=var["amp"]):
            output_dict = self(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                do_sample=True,
            )
        output_values = output_dict["mlm_output"].float() ##of shape (batch size, # of genes)
        pred_gene_values = torch.zeros_like(ori_gene_values)
        pred_gene_values[:, input_gene_ids] = output_values
        return pred_gene_values