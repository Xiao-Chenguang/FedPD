import torch
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset
from model import convnet, mlp, resnet
import ray
import copy


DATASETS = {
    "cifar": datasets.CIFAR10,
    "mnist": datasets.MNIST,
    "emnist": datasets.EMNIST
}


def get_flat_grad_from(grad):
    flat_grad = torch.cat([torch.flatten(p) for p in grad])
    return flat_grad


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model:
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def average_grad(grads):
    # flatten the grads to tensors
    flat_grads = []
    for grad in grads:
        flat_grads.append(get_flat_grad_from(grad))

    average_flat_grad = torch.mean(torch.stack(flat_grads), dim=0)
    grad_0 = grads[0]
    average_grad_a = []
    for p in grad_0:
        average_grad_a.append(torch.zeros_like(p))

    set_flat_params_to(average_grad_a, average_flat_grad)
    return average_grad_a


def weighted_sum_functions(models, weights):
    # ensure "weights" has unit sum
    # sum_weights = sum(weights)
    # weights = [weight/sum_weights for weight in weights]

    if weights is None:
        weights = [1./len(models)]*len(models)

    average_model = copy.deepcopy(models[0])
    sds = [model.state_dict() for model in models]
    average_sd = sds[0]
    for key in sds[0]:
        # print(key, )
        if sds[0][key].dtype is torch.int64:
            average_sd[key] = torch.sum(torch.stack([sd[key].float() * weight for sd, weight in zip(sds, weights)]),
                                        dim=0).to(torch.int64)
        else:
            average_sd[key] = torch.sum(torch.stack([sd[key] * weight for sd, weight in zip(sds, weights)]), dim=0)
    average_model.load_state_dict(average_sd)
    return average_model


def compute_model_delta(model_1, model_2):
    sd1 = model_1.state_dict()
    sd2 = model_2.state_dict()

    for key in sd1:
        sd1[key] = sd1[key] - sd2[key]
    model_1.load_state_dict(sd1)
    return model_1

def split_dataset(n_workers, homo_ratio, dataset: VisionDataset, transform=None):
    data = dataset.data
    label = dataset.targets

    # centralized case, no need to split
    if n_workers == 1:
        return [make_dataset(data, label, dataset.train, transform)]

    homo_ratio = homo_ratio
    n_workers = n_workers

    n_data = data.shape[0]

    n_homo_data = int(n_data * homo_ratio)

    n_homo_data = n_homo_data - n_homo_data % n_workers
    n_data = n_data - n_data % n_workers

    if n_homo_data > 0:
        data_homo, label_homo = data[0:n_homo_data], label[0:n_homo_data]
        data_homo_list, label_homo_list = np.split(data_homo, n_workers), label_homo.chunk(n_workers)

    if n_homo_data < n_data:
        data_hetero, label_hetero = data[n_homo_data:n_data], label[n_homo_data:n_data]
        label_hetero_sorted, index = torch.sort(label_hetero)
        data_hetero_sorted = data_hetero[index]

        data_hetero_list, label_hetero_list = np.split(data_hetero_sorted, n_workers), label_hetero_sorted.chunk(
            n_workers)

    if 0 < n_homo_data < n_data:
        data_list = [np.concatenate([data_homo, data_hetero], axis=0) for data_homo, data_hetero in
                     zip(data_homo_list, data_hetero_list)]
        label_list = [torch.cat([label_homo, label_hetero], dim=0) for label_homo, label_hetero in
                      zip(label_homo_list, label_hetero_list)]
    elif n_homo_data < n_data:
        data_list = data_hetero_list
        label_list = label_hetero_list
    else:
        data_list = data_homo_list
        label_list = label_homo_list

    return [make_dataset(_data, _label, dataset.train, transform) for _data, _label in zip(data_list, label_list)]


class LocalDataset(VisionDataset):
    def __init__(self, data, label, train, transform=None, root: str = ""):
        super().__init__(root, transform)
        self.data = data
        self.label = label
        self.transform = transform
        self.train = train
        assert data.shape[0] == label.shape[0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        sample = self.data[item]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.label[item]


def make_dataset(data, label, train, transform):
    return LocalDataset(data, label, train, transform)


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

def make_transforms(args, train=True):
    if args.dataset == "cifar10":
        if train:
            if not args.no_data_augmentation:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    else:
        raise NotImplementedError

    return transform


def make_dataloader(args, dataset: LocalDataset):
    if dataset.train is True:
        batch_size = dataset.data.shape[0] // args.client_step_per_epoch
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    else:
        dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    return dataloader

def make_monitor_fn():
    def evaluate_fn(model):
        param_norm = torch.norm(torch.stack([torch.norm(param) for param in model.parameters()]))
        return [param_norm]
    return evaluate_fn


def make_evaluate_fn(dataloader, device, eval_type="accuracy", n_classes=0, loss_fn=None):
    if eval_type == "accuracy":
        def evaluate_fn(model):
            with torch.autograd.no_grad():
                n_data = 0
                n_correct = 0
                loss = 0
                for data, label in dataloader:
                    data = data.to(device)
                    label = label.to(device)
                    f_data = model(data)
                    loss += loss_fn(f_data, label).item() * data.shape[0]
                    pred = f_data.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    n_correct += pred.eq(label.view_as(pred)).sum().item()
                    n_data += data.shape[0]
            return [np.true_divide(n_correct, n_data), np.true_divide(loss, n_data)]
    elif eval_type == "class_wise_accuracy":
        def evaluate_fn(model):
            correct_hist = torch.zeros(n_classes).to(device)
            label_hist = torch.zeros(n_classes).to(device)
            for data, label in dataloader:
                data = data.to(device)
                label = label.to(device)
                label_hist += torch.histc(label, n_classes, max=n_classes)
                f_data = model(data)
                pred = f_data.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_index = pred.eq(label.view_as(pred)).squeeze()
                label_correct = label[correct_index]
                correct_hist += torch.histc(label_correct, n_classes, max=n_classes)

            correct_rate_hist = correct_hist / label_hist
            return [correct_rate_hist.cpu().numpy()]
    else:
        raise NotImplementedError

    return evaluate_fn


def save_model(args, fed_learner):
    return


def load_dataset(args):
    if args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(root='datasets/' + args.dataset, download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets))
        dataset_test = datasets.CIFAR10(root='datasets/' + args.dataset, train=False, download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
        n_classes = 10
        n_channels = 3
    else:
        raise NotImplementedError

    return dataset_train, dataset_test, n_classes, n_channels


def make_model(args, n_classes, n_channels, device):
    dense_hidden_size = tuple([int(a) for a in args.dense_hid_dims.split("-")])
    conv_hidden_size = tuple([int(a) for a in args.conv_hid_dims.split("-")])

    if args.model == "convnet":
        model = convnet.LeNet5(n_classes, n_channels, conv_hidden_size, dense_hidden_size, device)
    elif args.model == "mlp":
        model = mlp.MLP(n_classes, dense_hidden_size, device)
    elif args.model == "resnet":
        model = resnet.resnet20().to(device)
    else:
        raise NotImplementedError

    return model


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:min(len(lst), i + n)]


class Logger:
    def __init__(self, writer, test_fn, test_metric='accuracy'):
        self.writer = writer
        self.test_metric = test_metric
        self.test_fn = test_fn

    def log(self, step, model):
        metric = self.test_fn(model)
        if len(metric) == 1:
            t_accuracy = metric[0]
        elif len(metric) == 2:
            t_accuracy, t_loss = metric
        elif len(metric) == 4:
            t_accuracy, t_loss, tr_accuracy, tr_loss = metric
        else:
            raise NotImplementedError

        if self.test_metric == 'accuracy':
            self.writer.add_scalar("correct rate vs round/test", t_accuracy, step)
            if 't_loss' in locals(): self.writer.add_scalar("loss vs round/test", t_loss, step)
            if 'tr_accuracy' in locals(): self.writer.add_scalar("correct rate vs round/train", tr_accuracy, step)
            if 'tr_loss' in locals(): self.writer.add_scalar("loss vs round/train", tr_loss, step)

        elif self.test_metric == 'class_wise_accuracy':
            n_classes = len(t_accuracy)
            for i in range(n_classes):
                # the i th element is the accuracy for the test data with label i
                self.writer.add_scalar(f"class-wise correct rate vs round/test/class_{i}", t_accuracy[i], step)
                if 't_loss' in locals(): self.writer.add_scalar(f"class-wise loss vs round/test/class_{i}", t_loss[i], step)
                if 'tr_accuracy' in locals(): self.writer.add_scalar(f"class-wise correct rate vs round/test/class_{i}", tr_accuracy[i], step)
                if 'tr_loss' in locals(): self.writer.add_scalar(f"class-wise loss vs round/test/class_{i}", tr_loss[i], step)
        elif self.test_metric == 'model_monitor':
            self.writer.add_scalar("model param norm vs round", metric[0], step)
        else:
            raise NotImplementedError



def create_imbalance(dataset, reduce_classes=(0,), reduce_to_ratio=.2):
    data = dataset.data
    label = dataset.targets

    reduce_mask = torch.zeros(data.shape[0], dtype=torch.bool)
    for reduce_class in reduce_classes:
        reduce_mask = torch.logical_or(reduce_mask, label == reduce_class)
    preserve_mask = torch.logical_not(reduce_mask)

    label_reduce = label[reduce_mask]
    len_reduce = label_reduce.shape[0]
    label_reduce = label_reduce[:max(1, int(len_reduce * reduce_to_ratio))]
    label_preserve = label[preserve_mask]

    label = torch.cat([label_reduce, label_preserve], dim=0)

    preserve_mask_np = preserve_mask.numpy()
    reduce_mask_np = reduce_mask.numpy()

    data_reduce = data[reduce_mask_np]
    data_reduce = data_reduce[:max(1, int(len_reduce * reduce_to_ratio))]
    data_preserve = data[preserve_mask_np]

    data = np.concatenate([data_reduce, data_preserve], axis=0)

    remain_len = label.shape[0]

    rand_index = torch.randperm(remain_len)
    rand_index_np = rand_index.numpy()

    dataset.data = data[rand_index_np]
    dataset.targets = label[rand_index]

    return dataset


def _evaluate(loss_fn, device, model, dataloader):
    loss = torch.zeros(1).to(device)
    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device)
        loss += loss_fn(model(data), label)

    return loss.item()


@ray.remote(num_gpus=.3, num_cpus=4)
def _evaluate_ray(loss_fn, device, model, dataloader):
    loss = torch.zeros(1).to(device)
    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device)
        loss += loss_fn(model(data), label)

    return loss.item()