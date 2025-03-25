from transformers import BertPreTrainedModel, BertModel
from torch import nn
import torch
from torchcrf import CRF
from mega_pytorch import MegaLayer


class HREBCRF(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # Dynamically learn r_lstm
        self.r_lstm = nn.Parameter(torch.tensor(0.5))
        # Dynamically learn r_mega
        self.r_mega = nn.Parameter(torch.tensor(0.5))
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.5)
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=2,  # 设置为2层
            batch_first=True,
            bidirectional=True,
            dropout=0.5  # 层间dropout
        )
        self.mega = MegaLayer(
            dim=config.hidden_size,
            ema_heads=32,
            attn_dim_qk=128,
            attn_dim_value=256,
            laplacian_attn_fn=False,
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bilstm(sequence_output)

        r_lstm = self.r_lstm
        r_mega = self.r_mega

        lstm_output = r_lstm * lstm_output + (1 - r_lstm) * sequence_output
        lstm_output = self.layer_norm(lstm_output)

        mega_output = self.mega(lstm_output)
        mega_output = r_mega * mega_output + (1 - r_mega) * lstm_output

        logits = self.classifier(mega_output)

        loss = None
        if labels is not None:
            labels = torch.where(
                labels == -100, torch.tensor(0, dtype=labels.dtype), labels)
            log_likelihood, tags = self.crf(
                logits, labels), self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
        tags = torch.Tensor(tags)

        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, tags

    def init_weights(self):
        for name, param in self.bilstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

        nn.init.uniform_(self.crf.transitions, -0.1, 0.1)
        nn.init.uniform_(self.crf.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.crf.end_transitions, -0.1, 0.1)
