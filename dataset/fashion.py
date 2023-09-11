from dataset import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


def get_fashion_dataset():
    # load dataset from the hub
    dataset = load_dataset("fashion_mnist")
    # image_size = 28
    # channels = 1
    batch_size = 128

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    def fashion_transforms(examples):
        examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]

        return examples

    transformed_dataset = dataset.with_transform(fashion_transforms).remove_columns("label")

    # create dataloader
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

    return dataloader
