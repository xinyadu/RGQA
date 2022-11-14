import argparse 
import logging 
import os 
import random 
import timeit 
from datetime import datetime 

import torch 
import wandb 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything



from src.genie.data_module import RAMSDataModule
from src.genie.ACE_data_module import ACEDataModule
from src.genie.ACEWIKI_data_module import ACEWIKIDataModule
from src.genie.WIKI_data_module import WIKIDataModule
from src.genie.ACE_data_module_fewshot import ACEDataModuleFewshot
from src.genie.KAIROS_data_module import KAIROSDataModule 
from src.genie.model import GenIEModel 


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=['gen','constrained-gen']
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['RAMS', 'ACE', 'KAIROS', 'ACEWIKI', 'WIKI']
    )
    parser.add_argument('--tmp_dir', type=str)
    parser.add_argument(
        "--ckpt_name",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--load_ckpt",
        default=None,
        type=str, 
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--val_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default=None,
    )
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--coref_dir', type=str, default='data/kairos/coref_outputs')
    parser.add_argument('--use_info', action='store_true', default=False, help='use informative mentions instead of the nearest mention.')
    parser.add_argument('--mark_trigger', action='store_true')
    parser.add_argument('--sample-gen', action='store_true', help='Do sampling when generation.')
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--eval_only", action="store_true",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_clip_val", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    
    parser.add_argument("--gpus", default=None, help='-1 means train on all gpus')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    parser.add_argument("--ir", action='store_true', default=False, help='if augment ir result')
    parser.add_argument("--add_yn", action='store_true', default=False, help='whether to add Y and N in ans')
    # fewshot args
    parser.add_argument("--fewshot", action='store_true', default=False, help='if do fewshot train/test')
    parser.add_argument("--sampling", type=str, required=False, choices=['random', 'ppl', 'c_context', 'c_context_trg'])
    parser.add_argument("--shot", type=int, required=False, choices=[10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000])
    args = parser.parse_args()

    # # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    # Set seed
    seed_everything(args.seed)

    logger.info("Training/evaluation parameters %s", args)

    
    if not args.ckpt_name:
        d = datetime.now() 
        time_str = d.strftime('%m-%dT%H%M')
        args.ckpt_name = '{}_{}lr{}_{}'.format(args.model,  args.train_batch_size * args.accumulate_grad_batches, 
                args.learning_rate, time_str)


    args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}')
    
    os.makedirs(args.ckpt_dir)

    top_k_value = 12
    if args.fewshot:
        top_k_value = 1

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        save_last=True,
        save_top_k=top_k_value,
        monitor='val/loss',
        mode='min',
        save_weights_only=True,
        filename='{epoch}', # this cannot contain slashes 

    )

   


    lr_logger = LearningRateMonitor() 
    tb_logger = TensorBoardLogger('logs/')

    model = GenIEModel(args)
    if args.dataset == 'RAMS':
        dm = RAMSDataModule(args)
    elif args.dataset == 'ACE':
        if args.fewshot:
            dm = ACEDataModuleFewshot(args)
        else:
            dm = ACEDataModule(args)
    elif args.dataset == 'ACEWIKI':
        dm = ACEWIKIDataModule(args)
    elif args.dataset == 'WIKI':
        dm = WIKIDataModule(args)
    elif args.dataset == 'KAIROS':
        dm = KAIROSDataModule(args)



    if args.max_steps < 0 :
        args.max_epochs = args.min_epochs = args.num_train_epochs 
    
    

    trainer = Trainer(
        logger=tb_logger,
        min_epochs=args.num_train_epochs,
        max_epochs=args.num_train_epochs, 
        gpus=args.gpus, 
        checkpoint_callback=checkpoint_callback, 
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val, 
        num_sanity_val_steps=0, 
        val_check_interval=0.5, # use float to check every n epochs 
        precision=16 if args.fp16 else 32,
        callbacks = [lr_logger, ],

    ) 

    if args.load_ckpt:
        model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict']) 

    if args.eval_only: 
        dm.setup('test')
        trainer.test(model, datamodule=dm) #also loads training dataloader 
    else:
        dm.setup('fit')
        trainer.fit(model, dm) 
    

    

if __name__ == "__main__":
    main()