from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import TrainerCallback
import torch


def tokenize(batch, tokenizer):
    result = {
        'label_ids': [],
        'input_ids': [],
        'token_type_ids': [],
    }
    max_length = 32

    for tokens, label in zip(batch['tokens'], batch['label_ids']):
        tokenids = tokenizer(tokens, add_special_tokens=False)

        token_ids = []
        label_ids = []
        for ids, lab in zip(tokenids['input_ids'], label):
            if len(ids) > 1 and lab % 2 == 1:
                token_ids.extend(ids)
                chunk = [lab + 1] * len(ids)
                chunk[0] = lab
                label_ids.extend(chunk)
            else:
                token_ids.extend(ids)
                chunk = [lab] * len(ids)
                label_ids.extend(chunk)

        token_type_ids = tokenizer.create_token_type_ids_from_sequences(
            token_ids)
        token_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
        label_ids.insert(0, 0)
        label_ids.append(0)
        result['input_ids'].append(token_ids)
        result['label_ids'].append(label_ids)
        result['token_type_ids'].append(token_type_ids)

    result = tokenizer.pad(result, padding='longest',
                           max_length=max_length, return_attention_mask=True)
    for i in range(len(result['input_ids'])):
        diff = len(result['input_ids'][i]) - len(result['label_ids'][i])
        result['label_ids'][i] += [0] * diff
    return result


def compute_metrics(pred, dataset):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.flatten().astype(int)
    _label = dataset["train"].features["ner_tags"].feature.names
    true_predictions = [
        _label[pred] for pred, label in zip(preds, labels) if label != -100 and label != 0
    ]

    true_labels = [
        _label[label] for pred, label in zip(preds, labels) if label != -100 and label != 0
    ]

    preds = true_predictions
    labels = true_labels

    f1 = f1_score(labels, preds, average='weighted')
    p = precision_score(labels, preds, average='weighted')
    r = recall_score(labels, preds, average='weighted')
    print(classification_report(labels, preds))
    return {
        'precision': p,
        'recall': r,
        'f1': f1,
    }


class WeightLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            model = kwargs.get('model')
            if model is not None:
                r_lstm = torch.sigmoid(model.r_lstm).item()
                r_mega = torch.sigmoid(model.r_mega).item()
                print(f"Current weight of:")
                print(f"r_lstm: {r_lstm:.4f}")
                print(f"r_mega: {r_mega:.4f}\n")
