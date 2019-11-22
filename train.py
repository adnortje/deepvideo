"""
Python Training Script for Video Autoencoders

"""

# imports
import os
import time
import torch
import torch.nn as nn
import argparse as arg
import torch.optim as optimizer
import video_networks as vid_net
import image_networks as img_net
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from process_data import VideoDataLoaders
from modules import VAELoss, VAEIntpLoss, LiteFlowNetLoss, RateLoss

# ----------------------------------------------------------------------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------------------------------------------------------------------

# create Argument Parser
parser = arg.ArgumentParser(
    prog="Train: Deep Video Compression System: ",
    description="Python script used to train a deep video compression system"
)

parser.add_argument(
    "--sys",
    "-s",
    metavar="SYSTEM",
    type=str,
    required=True,
    choices=[
        "VideoAuto",
        "PFrameVideoAuto",
        "BFrameVideoAuto",
        "ImageVAE",
        "VideoVAE"
    ],
    help="Deep video compression system"
)

parser.add_argument(
    "--epochs",
    "-e",
    metavar="EPOCHS",
    type=int,
    default=150,
    help="Number of epochs"
)

parser.add_argument(
    "--learn_rate",
    "-lr",
    metavar="LEARN_RATE",
    type=float,
    default=0.0001,
    help="Learning rate"
)

parser.add_argument(
    "--gamma",
    "-g",
    metavar="GAMMA",
    type=float,
    default=0.1,
    help="Learning rate decay rate"
)

parser.add_argument(
    "--log",
    "-l",
    metavar="LOG_DIR",
    type=str,
    default="./",
    help="Log directory"
)

parser.add_argument(
    "--train",
    "-td",
    metavar="TRAIN_DIR",
    type=str,
    default="./",
    help="Training data directory"
)

parser.add_argument(
    "--save",
    "-sv",
    metavar="SAVE_LOC",
    type=str,
    default="./",
    help="Model save location"
)

parser.add_argument(
    "--vid_ext",
    "-ve",
    metavar="VID_EXT",
    type=str,
    default=".mp4",
    help="Video extension"
)

parser.add_argument(
    "--frame_size",
    "-f_s",
    metavar="FRAME_SIZE",
    type=int,
    default=32,
    help="Video frame size"
)

parser.add_argument(
    "--batch_size",
    "-bs",
    metavar="BATCH_SIZE",
    type=int,
    default=3,
    help="Batch size"
)

parser.add_argument(
    "--bottleneck_depth",
    "-bnd",
    metavar="BND",
    type=int,
    required=True,
    help="Bottleneck depth"
)

parser.add_argument(
    "--vae_bottleneck_depth",
    "-vae_bnd",
    metavar="VAE_BND",
    type=int,
    default=128,
    help="VAE Bottleneck depth"
)

parser.add_argument(
    "--n_gop",
    "-gop",
    metavar="GOP",
    type=int,
    required=True,
    help="No. frames in GOP"
)

parser.add_argument(
    "--hierarchical",
    action="store_true"
)

parser.add_argument(
    "--fine_tune_bitrate",
    action="store_true"
)

parser.add_argument(
    "--multiscale",
    action="store_true"
)

parser.add_argument(
    "--epe_flow_loss",
    action="store_true"
)

parser.add_argument(
    "--cos_flow_loss",
    action="store_true"
)

parser.add_argument(
    "--rate_loss_beta",
    "-rl_b",
    metavar="RATE_LOSS_BETA",
    type=float,
    default=1.0,
    help="Rate loss multiplier"
)

parser.add_argument(
    "--rate_loss_threshold",
    "-rl_t",
    metavar="RATE_LOSS_THRESHOLD",
    type=float,
    default=0.0,
    help="Rate loss threshold (bpp)"
)

parser.add_argument(
    "--rate_loss_L",
    "-rl_lf",
    metavar="RATE_LOSS_LEVELS",
    type=int,
    default=4,
    help="Rate loss L (bpp)"
)

parser.add_argument(
    "--pre_trained_weights",
    "-pt_w",
    metavar="PRE-TRAINED_WEIGHTS",
    type=str,
    default=None,
    help="Pre-trained Model Weights"
)

parser.add_argument(
    "--vae_loss_beta",
    "-vae_b",
    metavar="VAE_LOSS_BETA",
    type=float,
    default=1.0,
    help="Beta-VAE for disentanglement"
)

parser.add_argument(
    "--verbose",
    "-v",
    action="store_true"
)

parser.add_argument(
    "--nvvl",
    action="store_true"
)

parser.add_argument(
    "--checkpoint",
    "-chkp",
    action="store_true"
)

args = parser.parse_args()

# ----------------------------------------------------------------------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------------------------------------------------------------------

# def system
sys = None
criterion = None
bit_imp_map = None
flow_criterion = None
rate_criterion = None

# SYSTEM DEFINITION

if args.sys == "VideoAuto":
    # Video Autoencoder
    sys = vid_net.VideoAuto(
        bnd=args.bottleneck_depth,
        stateful=False
    )

elif args.sys == "PFrameVideoAuto":
    # B-Frame Video Autoencoder
    sys = vid_net.PFrameVideoAuto(
        bnd=args.bottleneck_depth,
        multiscale=args.multiscale
    )

elif args.sys == "BFrameVideoAuto":
    # B-Frame Video Autoencoder
    sys = vid_net.BFrameVideoAuto(
        bnd=args.bottleneck_depth,
        multiscale=args.multiscale
    )

elif args.sys == "ImageVAE":
    # Image VAE
    sys = img_net.ImageVAE(
        bnd=args.vae_bottleneck_depth
    )

elif args.sys == "VideoVAE":
    # Video VAE
    sys = vid_net.VideoVAE(
        bnd=args.bottleneck_depth,
        vae_bnd=args.vae_bottleneck_depth
    )

if args.pre_trained_weights is not None:
    # Load Pre-trained weights
    w_f = os.path.expanduser(args.pre_trained_weights)

    if not os.path.isfile(w_f):
        raise FileNotFoundError("Specified pre-trained weights file d.n.e!")
    else:
        sys.load_model(args.pre_trained_weights)


# LOSS DEFINITION

# OPTICAL FLOW LOSS
if args.epe_flow_loss:
    # EPE Flow Loss - penalizes magnitude and direction of vectors
    flow_criterion = LiteFlowNetLoss(
        flow_loss="EPE"
    )

if args.cos_flow_loss:
    # Cosine Distance Flow Loss - penalizes vector direction
    flow_criterion = LiteFlowNetLoss(
        flow_loss="COSINE"
    )

# BITRATE LOSS
if args.fine_tune_bitrate:
    # Note: only used for fine tuning bitrate

    print("Fine Tune Bitrate: {}".format(args.fine_tune_bitrate))

    sys.fine_tune_bitrate(L=args.rate_loss_L)

    rate_criterion = RateLoss(
        beta=args.rate_loss_beta,
        r0=args.rate_loss_threshold,
        f_s=args.frame_size,
        n_gop=args.n_gop,
        bnd=args.bottleneck_depth
    )

# RECONSTRUCTION LOSS
if args.sys in ["ImageVAE"]:
    # VAE Loss function
    criterion = VAELoss(
        r_loss="MSE",
        beta=args.vae_loss_beta
    )

elif args.sys in ["VideoVAE"]:
    # VAE Loss + Interpolation Loss
    criterion = VAEIntpLoss(
        r_loss="MSE",
        beta=args.vae_loss_beta
    )

else:
    # MSE Loss
    criterion = nn.MSELoss()

# use GPU if available
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

# place model on device
sys.to(device)

# Adam optimizer
opt = optimizer.Adam(
    sys.parameters(),
    args.learn_rate
)

# MultiStep scheduler
scheduler = lr_scheduler.MultiStepLR(
    optimizer=opt,
    milestones=[30, 80, 140],
    gamma=args.gamma
)

# check train, log and save locations
log_loc = os.path.expanduser(args.log)

if not os.path.isdir(log_loc):
    raise NotADirectoryError("Log directory d.n.e")

save_loc = os.path.expanduser(args.save)

if not os.path.isdir(save_loc):
    raise NotADirectoryError("Save directory d.n.e")

train_dir = os.path.expanduser(args.train)

if not os.path.isdir(train_dir):
    raise NotADirectoryError("Train directory d.n.e")


# Video DataLoader
dataLoaders = VideoDataLoaders(
    nvvl=args.nvvl,
    n_gop=args.n_gop,
    root_dir=train_dir,
    b_s=args.batch_size,
    f_s=args.frame_size,
    vid_ext=args.vid_ext,
    color_space="RGB"
).get_data_loaders()

# start epoch, time previously elapsed, best MSE
t_prev = 0.0
best_loss = 10e8
current_epoch = 1

# def state file
state_file = "".join([save_loc, "/", sys.name, "_chkp.pt"])

if os.path.isfile(state_file):

    print("Continue Training from m.r.c : ")

    # load checkpoint
    chkp = torch.load(state_file)

    # load previous train time
    t_prev = chkp['time']

    # load previous epoch
    current_epoch = chkp['epoch']

    # load previous loss
    best_loss = chkp['best_loss']

    # load model weights
    sys.load_state_dict(chkp['sys'])

    # load optimizer states
    opt.load_state_dict(chkp['optimizer'])

    # load scheduler states
    scheduler.load_state_dict(chkp['scheduler'])

    # delete checkpoint
    del chkp

else:
    print("Training New System : ")

# writer for loss logging
writer = None

if args.verbose:
    writer = SummaryWriter(log_loc)

# start timing
train_start = time.time()


for epoch in range(current_epoch, args.epochs + 1, 1):

    # start epoch
    if args.verbose:
        print("Epoch {}/{}".format(epoch, args.epochs))
        print("--------------------------------------")

    epoch_start = time.time()

    for phase in ['train', 'valid']:

        # running loss
        run_loss = 0.0

        if phase is 'train':
            sys.train(True)
            # step scheduler
            scheduler.step()

        elif phase is 'valid':
            sys.train(False)

        i = 0
        for i, data in enumerate(dataLoaders[phase], 0):
            
            # place data on GPU
            if args.nvvl:
                inpt = data['input'].permute(0, 2, 1, 3, 4)
            else:
                inpt = data.permute(0, 2, 1, 3, 4).to(device)

            # [0, 1] -> [-1, 1]
            inpt = (inpt - 0.5) / 0.5

            # zero model gradients
            opt.zero_grad()

            if args.sys in ["ImageVAE"]:
                # Image VAE
                inpt = inpt[:, :, 0]
                output, mu, logvar = sys(inpt)
                # VAE loss
                loss = criterion(output, target=inpt, mu=mu, logvar=logvar)

            elif args.sys in ["VideoVAE"]:
                # Video VAE
                output, mu, logvar = sys(inpt)
                intp = sys.interpolate(inpt)
                loss = criterion(
                    output, target=inpt[:, :, -1], mu=mu, logvar=logvar, intp=intp, target_intp=inpt[:, :, 1:-1]
                )

            elif args.sys in ["PFrameVideoAuto"]:
                # P-Frame Video Autoencoder
                output = sys(inpt)

                if args.fine_tune_bitrate:
                    output, bit_imp_map = output

                inpt = inpt[:, :, 1:]
                loss = criterion(output, target=inpt)

            elif args.sys in ["BFrameVideoAuto"]:
                # B-Frame Video Autoencoder
                output = sys(inpt)

                if args.fine_tune_bitrate:
                    output, bit_imp_map = output

                inpt = inpt[:, :, 1:-1]
                loss = criterion(output, target=inpt)

            else:
                # forward
                output = sys(inpt)
                loss = criterion(output, target=inpt)

            if args.epe_flow_loss or args.cos_flow_loss:
                # add Flow Loss
                flow_loss = flow_criterion(output, target=inpt)
                loss = loss + flow_loss

            if args.fine_tune_bitrate:
                # add Bitrate Loss
                loss = loss + rate_criterion(bit_imp_map)

            # running loss
            run_loss += loss.item()

            if phase == 'train':
                # backward & optimise
                loss.backward()
                opt.step()

            # del grad trees to save memory
            del loss, inpt, output

        # epoch loss averaged over batch number
        epoch_loss = run_loss / (i+1)

        if args.verbose:
            print("Phase: {} Loss : {}".format(phase, epoch_loss))
            writer.add_scalar('{}/loss'.format(phase), epoch_loss, epoch)

        # save best system
        if phase is 'valid' and epoch_loss < best_loss:
            best_loss = epoch_loss
            fn = ''.join([save_loc, '/', sys.name, '.pt'])
            torch.save(sys.state_dict(), fn)

    # end of epoch
    epoch_time = (time.time() - epoch_start) / 60

    if args.verbose:
        print("Epoch time: {} min".format(epoch_time))
        print("-------------------------------------")

    if args.checkpoint:
        # save checkpoint
        chkp = {
            'epoch': epoch+1,
            'sys': sys.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'time': time.time() - train_start + t_prev
        }
        torch.save(chkp, state_file)

# end of Training
train_time = (time.time() - train_start + t_prev) / 60
if args.verbose:
    print('Total Training Time {} min'.format(train_time))
    print('Best Loss : {}'.format(best_loss))
    print('FIN TRAINING')

# close writer
writer.close()