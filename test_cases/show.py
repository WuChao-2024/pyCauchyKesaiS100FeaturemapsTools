import os
import numpy as np
from libpyCauchyKesaiS100FeaturemapsTools import CauchyKesai


p_ = "hbm_models"

for name in os.listdir(p_):
    p = os.path.join(p_, name)
    print("\n\n\n")
    print("=+ " * 20)
    print(p)
    try:
        m = CauchyKesai(p)
        m.summary()
    except:
        print("error")