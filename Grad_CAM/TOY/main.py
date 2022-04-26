import argparse
from train import trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR', choices=['STL', 'CIFAR', 'OWN'])
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--model_name', type=str, default='model.pth')

    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('-s', '--save_model_in_epoch', action='store_true')
    config = parser.parse_args()
    print(config)

    trainer(config)
