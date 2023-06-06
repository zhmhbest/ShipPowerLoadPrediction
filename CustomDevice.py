import torch


if 0 == torch.cuda.device_count():
    print("None cuda device.")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda", torch.cuda.current_device())
    print(f"Using {torch.cuda.get_device_name(torch.cuda.current_device())}.")
else:
    DEVICE = torch.device("cpu")
    print(f"Using cpu.")
