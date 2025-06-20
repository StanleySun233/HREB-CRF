from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import TrainerCallback
import torch


def tokenize(batch, tokenizer, max_length=64):
    tokenized_inputs = tokenizer(
        batch["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
    )

    all_labels = []
    for i, labels_for_one_example in enumerate(batch["label_ids"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels_for_one_example[word_idx])
            else:
                original_label = labels_for_one_example[word_idx]
                if original_label % 2 == 1:
                    label_ids.append(original_label + 1)
                else:
                    label_ids.append(original_label)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["label_ids"] = all_labels
    return tokenized_inputs


def compute_metrics(pred, dataset):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.flatten().astype(int)
    _label = dataset["train"].features["ner_tags"].feature.names
    true_predictions = [
        _label[pred] for pred, label in zip(preds, labels) if label != -100 and not (label == 0 and pred == 0)
    ]

    true_labels = [
        _label[label] for pred, label in zip(preds, labels) if label != -100 and not (label == 0 and pred == 0)
    ]

    preds = true_predictions
    labels = true_labels
    filtered_labels = [l for l in _label if l != 'O']
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    p = precision_score(labels, preds, average='weighted', zero_division=0)
    r = recall_score(labels, preds, average='weighted', zero_division=0)
    print(classification_report(labels, preds, labels=filtered_labels, zero_division=0))
    
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
