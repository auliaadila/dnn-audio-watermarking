# create_taps.py  (run once)
from utils import bwh
import numpy as np, pickle
taps = np.array(bwh(n=16, fc=4_000, fs=16_000, length=25))
pickle.dump(taps, open("model_b/lpf_taps.pkl", "wb"))
print(f"Created taps with {len(taps)} coefficients and saved to model_b/lpf_taps.pkl")