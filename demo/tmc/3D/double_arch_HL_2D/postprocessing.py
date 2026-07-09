import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# load loading history from file
loading_history_file = BASE_DIR / "loading_history.npz"
data = np.load(loading_history_file)

loading_history = data['loading_history']
m_node_displ = data['m_node_displ']

# load theoretical curve from data (.csv file)
theoretical_curve_file = BASE_DIR / "paper_data.csv"
theoretical_data = np.loadtxt(theoretical_curve_file, delimiter=';')
t_load_step = theoretical_data[:, 0] * 140 # scale to match original values 
t_displacement = theoretical_data[:, 1] * 40 # scale to match original values
    
plt.figure(figsize=(8, 6))
plt.plot(loading_history, m_node_displ, linestyle='--', color='b', label='TMC')
plt.plot(t_load_step, t_displacement, linestyle='-', color='r', label='Reference')
plt.xlabel('Load step', fontsize=14)
plt.ylabel('Vertical Displacement of the ring\'s mid-point (mm)', fontsize=14)
# plt.title('Loading History vs Displacement at Point', fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig(f'{BASE_DIR}/loading_history_plot.png', dpi=300)
plt.close()
