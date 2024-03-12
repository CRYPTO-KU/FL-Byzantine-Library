from torch.utils.data import Dataset
import torch

def get_glue_dataset(args):
    from transformers import pipeline
    from datasets import load_dataset

    train_data = load_dataset("glue", "cola", split="train")
    test_data = load_dataset("glue", "cola", split="validation")

    pipe = pipeline("feature-extraction", model="bert-base-uncased", device=args.gpu_id, return_tensors="pt")

    def process_data(data):
        class CustomDataset(Dataset):

            def __init__(self, features, labels, max_length=None):
                self.targets = torch.LongTensor(labels)
                self.features = features

                for i in range(len(self.features)):
                    self.features[i] = torch.stack(self.features[i][0])

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, idx):
                return self.features[idx], self.targets[idx]

        print("Extracting Features...")
        data = data.map(lambda x: {"features": pipe(x["sentence"]), "label": x["label"]})

        data.set_format("pytorch")

        res = CustomDataset(data["features"], data["label"])

        return res

    train_data = process_data(train_data)
    test_data = process_data(test_data)

    return train_data, test_data