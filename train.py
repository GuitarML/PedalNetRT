import pytorch_lightning as pl
import argparse
import sys

from model import PedalNet


def main(args):
    """
    This trains the PedalNet model to match the output data from the input data.

    When you resume training from an existing model, you can override hparams such as
        max_epochs, batch_size, or learning_rate. Note that changing num_channels,
        dilation_depth, num_repeat, or kernel_size will change the shape of the WaveNet
        model and is not advised.

    """
    if args.resume_training != "":
        model = PedalNet.load_from_checkpoint(args.resume_training)
        # Check for any hparams overridden by user and update
        for arg in sys.argv[1:]:
            arg2 = arg.split("=")[0].split("--")[1]
            if arg2 != "resume_training" and arg2 != "cpu" and arg2 != "tpu_cores":
                arg3 = arg.split("=")[1]
                if arg2 in model.hparams:
                    if arg2 == "learning_rate":
                        model.hparams[arg2] = float(arg3)
                    else:
                        model.hparams[arg2] = int(arg3)
                    print("Hparam overridden by user: ", arg2, "=", arg3, "\n")
        if args.cpu == 0:
            trainer = pl.Trainer(
                resume_from_checkpoint=args.resume_training,
                gpus=args.gpus,
                row_log_interval=100,
                max_epochs=args.max_epochs,
            )
        else:
            trainer = pl.Trainer(
                resume_from_checkpoint=args.resume_training, row_log_interval=100, max_epochs=args.max_epochs
            )
        print("\nHparams for continued model training:\n")
        print(model.hparams, "\n")
    else:
        model = PedalNet(args)
        if args.cpu == 0:
            trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=args.gpus, row_log_interval=100)
            # The following line is for use with the Colab notebook when training on TPUs.
            # Comment out the above line and uncomment the below line to use.

            # max_epochs=args.max_epochs, tpu_cores=args.tpu_cores, gpus=args.gpus, row_log_interval=100
        else:
            trainer = pl.Trainer(max_epochs=args.max_epochs, row_log_interval=100)
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_channels", type=int, default=12)
    parser.add_argument("--dilation_depth", type=int, default=10)
    parser.add_argument("--num_repeat", type=int, default=1)
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-3)

    parser.add_argument("--max_epochs", type=int, default=1_500)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tpu_cores", default="8")
    parser.add_argument("--cpu", type=int, default=0)

    parser.add_argument("--data", default="data.pickle")

    parser.add_argument("--resume_training", default="")
    args = parser.parse_args()
    main(args)
