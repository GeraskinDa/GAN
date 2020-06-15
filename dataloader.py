from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os


class ImageDataloader(Dataset):
    """Dataset class of images for dataloader
    """
    def __init__(self, path_X, path_Y, transform_X=None, transform_Y=None):
        """Initializes ImageDataloader object

        Parameters
        ----------
        path_X: str
            Path to the X images
        path_Y: str
        transform_X: torchvision.transforms object
            Transforms for X images
        transform_Y: torchvision.transforms object
            Transforms for Y images
        """
        self.path_X = path_X
        self.path_Y = path_Y
        self.transform_X = transform_X
        self.transform_Y = transform_Y
        self.img_names_X = os.listdir(path_X)
        self.img_names_Y = os.listdir(path_Y)
        self.X_num = len(self.img_names_X)
        self.Y_num = len(self.img_names_Y)

    def __len__(self):
        return max(self.X_num, self.Y_num)

    def __getitem__(self, index):

        X_sample = Image.open(os.path.join(self.path_X, self.img_names_X[index % self.X_num])).convert('RGB')
        Y_sample = Image.open(os.path.join(self.path_Y, self.img_names_Y[index % self.Y_num])).convert('RGB')

        if self.transform_X is not None:
            X_sample = self.transform_X(X_sample)

        if self.transform_Y is not None:
            Y_sample = self.transform_Y(Y_sample)

        return {'X': X_sample, 'Y': Y_sample}


def get_dataloader(path_X, path_Y, transform_X=None, transform_Y=None, batch_size=1, shuffle=True):
    """Creates dataloader for images

    Parameters
    ----------
    path_X: str
        Path to the X images
    path_Y: str
        Path to the Y images
    transform_X: torchvision.transforms object
        Transforms for X images
    transform_Y: torchvision.transforms object
        Transforms for Y images
    batch_size: int
        Size of batch
    shuffle: bool

    Returns
    -------
    torch.utils.data.DataLoader class object
    """
    dataset = ImageDataloader(path_X, path_Y, transform_X, transform_Y)
    return DataLoader(dataset, batch_size, shuffle)