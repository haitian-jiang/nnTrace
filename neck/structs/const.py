DTYPE = {"fp32", "fp16", "bf16"}
MCH_EPS = {"bf16": 2**-7, "fp16": 2**-10, "fp32": 2**-23}


__all__ = [
    "DTYPE",
    "MCH_EPS",
]
