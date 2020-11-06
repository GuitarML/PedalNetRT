import pytorch_lightning as pl
import argparse

from model import PedalNet


def main(args):
    model = PedalNet(args)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, row_log_interval=100
        # The following line is for use with the Colab notebook when training on TPUs.
        # Comment out the above line and uncomment the below line to use.
        
        # max_epochs=args.max_epochs, tpu_cores=args.tpu_cores, gpus=args.gpus, row_log_interval=100
    )
    trainer.fit(model)
    trainer.save_checkpoint('models/' + args.model + '.ckpt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_channels", type=int, default=16)
    parser.add_argument("--dilation_depth", type=int, default=10)
    parser.add_argument("--num_repeat", type=int, default=1)
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-3)

    parser.add_argument("--max_epochs", type=int, default=1_500)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tpu_cores", default="8")

    parser.add_argument("--data", default="data.pickle")
    parser.add_argument("--model", default="pedalnet")

    args = parser.parse_args()
    main(args)
