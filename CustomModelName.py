import os
import re


if not os.path.exists("./checkpoint"):
    os.mkdir("./checkpoint")


def get_model_name(name: str, index: int, others: str = "") -> str:
    expect_name = f"{name}-{index}"
    for file_name in os.listdir("./checkpoint"):
        if file_name.startswith(expect_name):
            return f"./checkpoint/{file_name}"
    return f"./checkpoint/{name}-{index}-{others}.pt"
