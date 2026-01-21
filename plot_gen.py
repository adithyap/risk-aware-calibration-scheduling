import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.stats as stats
from pathlib import Path

# Set scientific style for plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 2,
    'figure.autolayout': True
})

def generate_figure_a():
    """Generates Figure A: The Data Adaptation Pipeline"""
    print("Generating Figure A (Data Adaptation)...")
    
    # 1. Synthetic Data Generation
    t = np.linspace(0, 100, 500)
    
    # Simulate monotonic drift (exponential-ish) + noise
    baseline = 10
    drift_rate = 0.5
    noise = np.random.normal(0, 1.5, len(t))
    
    # First cycle (before splice)
    sensor_cycle1 = baseline + (drift_rate * t[:350]) + (0.005 * t[:350]**2) + noise[:350]
    
    # Virtual Threshold
    threshold_val = 60
    
    # Find crossing point (approximate for visual)
    splice_idx = 350
    splice_time = t[splice_idx]
    
    # Second cycle (reset/splice)
    # Reset to baseline + noise, then drift again
    sensor_cycle2 = baseline + (drift_rate * (t[350:] - t[350])) + noise[350:]
    
    # Combine
    sensor_data = np.concatenate([sensor_cycle1, sensor_cycle2])
    
    # Generate TTD (Sawtooth)
    # TTD = Time until crossing (at splice_time)
    ttd_cycle1 = splice_time - t[:350]
    # For cycle 2, let's assume the next crossing is far off
    ttd_cycle2 = (splice_time + 70) - t[350:] 
    ttd_data = np.concatenate([ttd_cycle1, ttd_cycle2])

    # 2. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- Panel 1: Raw Sensor ---
    ax1.plot(t, sensor_data, color='#1f77b4', label='Sensor Value')
    ax1.axhline(threshold_val, color='r', linestyle='--', alpha=0.8, linewidth=2, label='Virtual Threshold')
    
    # Draw vertical drop line for splice
    ax1.plot([t[splice_idx], t[splice_idx]], 
             [sensor_cycle1[-1], sensor_cycle2[0]], 
             color='#1f77b4', linestyle='-', linewidth=2)

    # Annotations
    ax1.annotate(
        'Threshold Crossing',
        xy=(t[splice_idx - 20], threshold_val),
        xytext=(t[splice_idx - 130], threshold_val + 28),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=10,
    )
                 
    ax1.annotate(
        'Splice Event\n(Reset to Baseline)',
        xy=(t[splice_idx], (sensor_cycle1[-1] + sensor_cycle2[0]) / 2),
        xytext=(t[splice_idx] + 15, 40),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=10,
    )

    ax1.set_ylabel('Sensor Signal')
    ax1.set_title('Panel 1: Virtual Thresholds & Synthetic Splice')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Label Generation ---
    ax2.plot(t, ttd_data, color='green', linewidth=2, label='Time-to-Drift (TTD)')
    ax2.axvline(t[splice_idx], color='gray', linestyle=':', alpha=0.7)
    
    # Fill under curve
    ax2.fill_between(t, 0, ttd_data, color='green', alpha=0.1)

    # Annotations
    ax2.annotate('TTD counts down to 0', xy=(t[100], ttd_data[100]), 
                 xytext=(t[50], ttd_data[100]+20),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
                 
    ax2.text(t[splice_idx] + 2, 5, 'Label Reset', fontsize=10, color='green', fontweight='bold')

    ax2.set_ylabel('Target Label (TTD)')
    ax2.set_xlabel('Time (Cycles)')
    ax2.set_title('Panel 2: Sawtooth Label Generation')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "data_adaptation_flow.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path.name} to {out_path.parent}")
    plt.close(fig)

def generate_figure_b():
    """Generates Figure B: The Risk-Aware Scheduler Flowchart"""
    print("Generating Figure B (Scheduler Flow)...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off') # Hide axes
    
    # -- Styles --
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
    # Matplotlib doesn't support a 'diamond' boxstyle; use 'round' with a highlight color instead.
    decision_props = dict(boxstyle='round,pad=0.3', facecolor='#fffbe6', edgecolor='orange', linewidth=1.5)
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # -- 1. Input Box --
    ax.text(1, 3, "Input\nWindowed Sensor\nData (40 cycles)", ha='center', va='center', bbox=box_props, fontsize=12)
    
    ax.annotate('', xy=(2.5, 3), xytext=(1.8, 3), arrowprops=arrow_props)
    
    # -- 2. Model Box --
    ax.text(3.8, 3, "Model\nTransformer with\nQuantile Head", ha='center', va='center', bbox=box_props, fontsize=12)
    
    ax.annotate('', xy=(5.5, 3), xytext=(4.8, 3), arrowprops=arrow_props)
    
    # -- 3. Output Visualization (Bell Curve) --
    # Draw a box frame
    rect = patches.Rectangle((5.5, 1.5), 3, 3, linewidth=1, edgecolor='gray', facecolor='white', linestyle='--')
    ax.add_patch(rect)
    
    # Draw Bell Curve manually inside the box
    x_dist = np.linspace(5.7, 8.3, 100)
    y_dist = 2 + 1.5 * np.exp(-(x_dist - 7)**2 / 0.2)
    ax.plot(x_dist, y_dist, color='#1f77b4')
    
    # Quantile Lines
    q1_x = 6.5
    q5_x = 7.0
    q9_x = 7.5
    
    # q0.1
    ax.plot([q1_x, q1_x], [1.8, 3.2], color='red', linestyle='-', linewidth=2)
    ax.text(q1_x, 1.6, r'$q_{0.1}$ (Risk)', ha='center', fontsize=11, color='red')
    
    # q0.5
    ax.plot([q5_x, q5_x], [1.8, 3.5], color='black', linestyle=':', linewidth=1)
    ax.text(q5_x, 3.6, r'$q_{0.5}$ (Median)', ha='center', fontsize=11)
    
    # q0.9
    ax.plot([q9_x, q9_x], [1.8, 3.2], color='gray', linestyle='--', linewidth=1)
    ax.text(q9_x, 1.6, r'$q_{0.9}$', ha='center', fontsize=11, color='gray')
    
    ax.text(7, 4.2, "Output Distribution", ha='center', fontweight='bold', fontsize=12)
    
    ax.annotate('', xy=(9.5, 3), xytext=(8.5, 3), arrowprops=arrow_props)
    
    # -- 4. Decision Gate --
    ax.text(10.5, 3, "Is $q_{0.1} <$\nSafety Margin?", ha='center', va='center', bbox=decision_props, fontsize=12)
    
    # -- 5. Actions --
    
    # YES path (Up)
    ax.annotate('Yes', xy=(10.5, 4.2), xytext=(10.5, 3.6), arrowprops=arrow_props, ha='center', fontsize=12, fontweight='bold')
    ax.text(10.5, 4.8, "ACTION:\nSchedule High Priority\n(Fail-Safe Trigger)", ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffdddd', edgecolor='red'), fontsize=12)
            
    # NO path (Down)
    ax.annotate('No', xy=(10.5, 1.8), xytext=(10.5, 2.4), arrowprops=arrow_props, ha='center', fontsize=12, fontweight='bold')
    ax.text(10.5, 1.2, "ACTION:\nMonitor / No Op", ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ddffdd', edgecolor='green'), fontsize=12)

    ax.set_title("Figure B: Risk-Aware Scheduling Logic", fontsize=16, pad=20)
    
    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "scheduler_flow.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path.name} to {out_path.parent}")
    plt.close(fig)

if __name__ == "__main__":
    generate_figure_a()
    generate_figure_b()
