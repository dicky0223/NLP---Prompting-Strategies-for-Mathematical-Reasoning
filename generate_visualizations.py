"""
Visualization Script for COMP7506 NLP Assignment 1 Report
Generates figures for performance analysis and results

Usage:
    python generate_visualizations.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color scheme
COLORS = {
    'zeroshot': '#FF6B6B',
    'fewshot': '#4ECDC4',
    'cot': '#45B7D1',
    'selfverif': '#FFA07A',
    'combined': '#98D8C8',
}

COLOR_LIST = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

# ============================================================================
# FIGURE 1: Performance Comparison
# ============================================================================

def generate_performance_comparison():
    """Generate comprehensive performance comparison visualization"""
    
    methods = ['Zero-shot', 'Few-shot', 'CoT', 'Self-Verify', 'CoT+SV']
    accuracies = [72.5, 78.3, 84.2, 81.7, 87.9]
    tokens = [456, 892, 1156, 1367, 3467]
    times = [1243, 2156, 3421, 4124, 10253]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Comprehensive Performance Analysis of Prompting Strategies', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # ---- Subplot 1: Accuracy Comparison ----
    bars1 = axes[0].bar(methods, accuracies, color=COLOR_LIST, 
                        edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0].set_ylim([60, 95])
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_axisbelow(True)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, accuracies)):
        axes[0].text(i, val + 1.5, f'{val}%', ha='center', 
                    fontweight='bold', fontsize=11)
        # Add improvement percentage
        if i > 0:
            improvement = val - accuracies[0]
            axes[0].text(i, val - 3, f'+{improvement:.1f}%', ha='center',
                        fontsize=9, color='green', fontweight='bold')
    
    axes[0].set_xticklabels(methods, rotation=15, ha='right')
    
    # ---- Subplot 2: Tokens vs Accuracy (Efficiency) ----
    scatter = axes[1].scatter(tokens, accuracies, s=400, c=COLOR_LIST, 
                             alpha=0.7, edgecolors='black', linewidth=2.5)
    
    # Add method labels
    for i, method in enumerate(methods):
        axes[1].annotate(method, (tokens[i], accuracies[i]), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Add trend line
    z = np.polyfit(tokens, accuracies, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(tokens), max(tokens), 100)
    axes[1].plot(x_smooth, p(x_smooth), "r--", alpha=0.5, linewidth=2, label='Trend')
    
    axes[1].set_xlabel('Average Tokens per Problem', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Efficiency Trade-off Analysis', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_axisbelow(True)
    axes[1].legend(fontsize=10)
    
    # ---- Subplot 3: Wall-clock Time ----
    bars3 = axes[2].barh(methods, times, color=COLOR_LIST, 
                         edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[2].set_xlabel('Wall-clock Time (seconds)', fontsize=12, fontweight='bold')
    axes[2].set_title('Inference Time Comparison', fontsize=13, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3, linestyle='--')
    axes[2].set_axisbelow(True)
    
    # Add time labels
    for i, (bar, val) in enumerate(zip(bars3, times)):
        axes[2].text(val + 300, i, f'{val}s', va='center', 
                    fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figure1_performance_comparison.pdf', dpi=300, bbox_inches='tight')
    print("✓ Saved: figure1_performance_comparison.pdf")
    plt.close()

# ============================================================================
# FIGURE 2: Difficulty-Based Analysis
# ============================================================================

def generate_difficulty_analysis():
    """Generate performance breakdown by problem difficulty"""
    
    difficulties = ['Simple\n(1-2 steps)', 'Medium\n(3-4 steps)', 'Complex\n(5+ steps)']
    zeroshot = [85.2, 72.1, 61.4]
    fewshot = [86.3, 77.8, 69.2]
    cot = [91.3, 84.8, 76.5]
    selfverif = [89.5, 82.1, 74.2]
    combined = [93.7, 88.2, 82.1]
    
    x = np.arange(len(difficulties))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    bars1 = ax.bar(x - 2*width, zeroshot, width, label='Zero-shot', 
                  color=COLOR_LIST[0], edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x - width, fewshot, width, label='Few-shot', 
                  color=COLOR_LIST[1], edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x, cot, width, label='CoT', 
                  color=COLOR_LIST[2], edgecolor='black', linewidth=1.2)
    bars4 = ax.bar(x + width, selfverif, width, label='Self-Verification', 
                  color=COLOR_LIST[3], edgecolor='black', linewidth=1.2)
    bars5 = ax.bar(x + 2*width, combined, width, label='CoT + Self-Verification', 
                  color=COLOR_LIST[4], edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Problem Difficulty Level', fontsize=13, fontweight='bold')
    ax.set_title('Accuracy Performance by Problem Difficulty', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.set_ylim([55, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figure2_difficulty_analysis.pdf', dpi=300, bbox_inches='tight')
    print("✓ Saved: figure2_difficulty_analysis.pdf")
    plt.close()

# ============================================================================
# FIGURE 3: Temperature Sensitivity
# ============================================================================

def generate_temperature_sensitivity():
    """Generate temperature sensitivity analysis"""
    
    temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Self-Verification (3 attempts with varying temps)
    sv_accuracies = [73.2, 78.5, 80.1, 79.3, 76.8, 72.4]
    
    # CoT + Self-Verification (3 attempts)
    combined_t1 = [84.2, 84.1, 83.9, 83.5, 82.8, 81.2]
    combined_t2 = [84.2, 83.8, 83.1, 82.2, 80.5, 77.8]
    combined_t3 = [84.2, 82.9, 81.2, 79.4, 76.1, 71.3]
    combined_ensemble = [87.9, 87.4, 86.8, 85.3, 83.1, 78.2]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Temperature Sensitivity Analysis', fontsize=15, fontweight='bold', y=1.00)
    
    # ---- Subplot 1: Self-Verification ----
    line1 = axes[0].plot(temperatures, sv_accuracies, marker='o', linewidth=2.5, 
                         markersize=9, color=COLOR_LIST[3], label='Self-Verification',
                         markerfacecolor=COLOR_LIST[3], markeredgecolor='black', markeredgewidth=1.5)
    axes[0].fill_between(temperatures, sv_accuracies, alpha=0.2, color=COLOR_LIST[3])
    axes[0].set_xlabel('Temperature', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Self-Verification Temperature Sensitivity', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_axisbelow(True)
    axes[0].set_ylim([70, 82])
    
    # Add value labels
    for temp, acc in zip(temperatures, sv_accuracies):
        axes[0].text(temp, acc + 0.4, f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    # ---- Subplot 2: CoT + Self-Verification ----
    axes[1].plot(temperatures, combined_t1, marker='o', linewidth=2, markersize=7,
                label='Attempt 1 (T=0)', color='#45B7D1', markerfacecolor='#45B7D1', 
                markeredgecolor='black', markeredgewidth=1)
    axes[1].plot(temperatures, combined_t2, marker='s', linewidth=2, markersize=7,
                label='Attempt 2 (T=varying)', color='#FFA07A', markerfacecolor='#FFA07A',
                markeredgecolor='black', markeredgewidth=1)
    axes[1].plot(temperatures, combined_t3, marker='^', linewidth=2, markersize=7,
                label='Attempt 3 (T=varying)', color='#FF69B4', markerfacecolor='#FF69B4',
                markeredgecolor='black', markeredgewidth=1)
    axes[1].plot(temperatures, combined_ensemble, marker='*', linewidth=3, markersize=14,
                label='Ensemble Vote', color='#98D8C8', markerfacecolor='#98D8C8',
                markeredgecolor='black', markeredgewidth=1.5)
    
    axes[1].set_xlabel('Temperature (for Attempts 2 & 3)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('CoT + Self-Verification Temperature Sensitivity', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10, loc='upper right', framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_axisbelow(True)
    axes[1].set_ylim([70, 90])
    
    plt.tight_layout()
    plt.savefig('figure3_temperature_sensitivity.pdf', dpi=300, bbox_inches='tight')
    print("✓ Saved: figure3_temperature_sensitivity.pdf")
    plt.close()

# ============================================================================
# FIGURE 4: Error Distribution
# ============================================================================

def generate_error_analysis():
    """Generate error distribution analysis"""
    
    error_types = ['Calculation', 'Comprehension', 'Logic', 'Extraction', 'Hallucination']
    
    zeroshot = [45, 24, 18, 8, 5]
    fewshot = [38, 23, 22, 12, 5]
    cot = [13, 22, 38, 18, 9]
    combined = [8, 18, 52, 15, 7]
    
    x = np.arange(len(error_types))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    bars1 = ax.bar(x - 1.5*width, zeroshot, width, label='Zero-shot', 
                  color=COLOR_LIST[0], edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x - 0.5*width, fewshot, width, label='Few-shot', 
                  color=COLOR_LIST[1], edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + 0.5*width, cot, width, label='CoT', 
                  color=COLOR_LIST[2], edgecolor='black', linewidth=1.2)
    bars4 = ax.bar(x + 1.5*width, combined, width, label='CoT + Self-Verification', 
                  color=COLOR_LIST[4], edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Error Frequency (% of failures)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Error Type', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution Across Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(error_types, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.set_ylim([0, 60])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figure4_error_analysis.pdf', dpi=300, bbox_inches='tight')
    print("✓ Saved: figure4_error_analysis.pdf")
    plt.close()

# ============================================================================
# FIGURE 5: Cost-Benefit Analysis
# ============================================================================

def generate_cost_benefit():
    """Generate cost-benefit analysis"""
    
    methods = ['Zero-shot', 'Few-shot', 'CoT', 'Self-Verify', 'CoT+SV']
    accuracy = [72.5, 78.3, 84.2, 81.7, 87.9]
    cost_multiple = [1.0, 1.96, 2.53, 2.99, 7.59]  # Relative to zero-shot
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cost-Benefit Trade-off Analysis', fontsize=15, fontweight='bold')
    
    # ---- Subplot 1: Accuracy vs Cost ----
    scatter = ax1.scatter(cost_multiple, accuracy, s=500, c=COLOR_LIST, 
                         alpha=0.7, edgecolors='black', linewidth=2.5)
    
    for i, method in enumerate(methods):
        ax1.annotate(method, (cost_multiple[i], accuracy[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.2))
    
    # Add efficiency lines
    for i in range(len(methods)-1):
        ax1.arrow(cost_multiple[i], accuracy[i], 
                 cost_multiple[i+1]-cost_multiple[i], accuracy[i+1]-accuracy[i],
                 head_width=0.2, head_length=0.3, fc='gray', ec='gray', alpha=0.3)
    
    ax1.set_xlabel('Computational Cost Multiple (relative to Zero-shot)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Computational Cost', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.set_ylim([70, 90])
    ax1.set_xlim([0, 8])
    
    # ---- Subplot 2: Marginal Cost-Benefit ----
    marginal_accuracy = [0, accuracy[1]-accuracy[0], accuracy[2]-accuracy[1], 
                        accuracy[3]-accuracy[2], accuracy[4]-accuracy[3]]
    marginal_cost = [0, cost_multiple[1]-cost_multiple[0], cost_multiple[2]-cost_multiple[1],
                    cost_multiple[3]-cost_multiple[2], cost_multiple[4]-cost_multiple[3]]
    
    bars = ax2.bar(range(1, len(methods)), marginal_accuracy[1:], 
                   color=COLOR_LIST[1:], edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax2.set_ylabel('Accuracy Gain (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Method Progression', fontsize=12, fontweight='bold')
    ax2.set_title('Marginal Accuracy Gains', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(1, len(methods)))
    ax2.set_xticklabels([f'→ {m}' for m in methods[1:]], fontsize=10, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Add labels with cost info
    for i, (bar, acc, cost) in enumerate(zip(bars, marginal_accuracy[1:], marginal_cost[1:]), 1):
        height = bar.get_height()
        ax2.text(i, height + 0.3, f'+{height:.1f}%\n({cost:.2f}× cost)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figure5_cost_benefit.pdf', dpi=300, bbox_inches='tight')
    print("✓ Saved: figure5_cost_benefit.pdf")
    plt.close()

# ============================================================================
# FIGURE 6: Hyperparameter Sensitivity
# ============================================================================

def generate_hyperparameter_sensitivity():
    """Generate hyperparameter sensitivity analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=15, fontweight='bold')
    
    # ---- Subplot 1: Num Examples (Few-shot) ----
    num_examples = [1, 3, 5, 7, 10]
    accuracy_examples = [75.1, 76.9, 78.3, 78.1, 77.8]
    efficiency_examples = [12.8, 10.4, 8.8, 7.5, 6.0]
    
    ax = axes[0, 0]
    ax2 = ax.twinx()
    
    line1 = ax.plot(num_examples, accuracy_examples, marker='o', linewidth=2.5, markersize=8,
                   color='#45B7D1', label='Accuracy', markerfacecolor='#45B7D1',
                   markeredgecolor='black', markeredgewidth=1.5)
    line2 = ax2.plot(num_examples, efficiency_examples, marker='s', linewidth=2.5, markersize=8,
                    color='#FFA07A', label='Efficiency', markerfacecolor='#FFA07A',
                    markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_xlabel('Number of Few-shot Examples', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='#45B7D1')
    ax2.set_ylabel('Efficiency Score', fontsize=11, fontweight='bold', color='#FFA07A')
    ax.set_title('Few-shot Example Count Impact', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelcolor='#45B7D1')
    ax2.tick_params(axis='y', labelcolor='#FFA07A')
    ax.set_axisbelow(True)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', fontsize=10)
    
    # ---- Subplot 2: Max Tokens ----
    max_tokens = [1024, 2048, 4096]
    accuracy_tokens = [82.4, 84.2, 84.3]
    truncated = [2.1, 0.3, 0.0]
    
    ax = axes[0, 1]
    ax2 = ax.twinx()
    
    bars = ax.bar(range(len(max_tokens)), accuracy_tokens, color='#98D8C8',
                 edgecolor='black', linewidth=1.5, alpha=0.8, label='Accuracy')
    line = ax2.plot(range(len(max_tokens)), truncated, marker='D', linewidth=2.5, markersize=8,
                   color='#FF6B6B', label='Truncated %', markerfacecolor='#FF6B6B',
                   markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='#98D8C8')
    ax2.set_ylabel('Truncated Problems (%)', fontsize=11, fontweight='bold', color='#FF6B6B')
    ax.set_title('Max Tokens Parameter Impact', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(max_tokens)))
    ax.set_xticklabels([f'{t}' for t in max_tokens])
    ax.set_xlabel('Max Tokens', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelcolor='#98D8C8')
    ax2.tick_params(axis='y', labelcolor='#FF6B6B')
    ax.set_axisbelow(True)
    
    for i, v in enumerate(accuracy_tokens):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)
    
    # ---- Subplot 3: Num Attempts (Self-Verification) ----
    num_attempts = [1, 2, 3, 5]
    accuracy_attempts = [73.2, 79.3, 81.7, 82.1]
    cost_attempts = [1, 2, 3, 5]
    
    ax = axes[1, 0]
    ax2 = ax.twinx()
    
    line1 = ax.plot(num_attempts, accuracy_attempts, marker='o', linewidth=2.5, markersize=8,
                   color='#4ECDC4', label='Accuracy', markerfacecolor='#4ECDC4',
                   markeredgecolor='black', markeredgewidth=1.5)
    line2 = ax2.plot(num_attempts, cost_attempts, marker='^', linewidth=2.5, markersize=8,
                    color='#FF69B4', label='Cost Multiple', markerfacecolor='#FF69B4',
                    markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_xlabel('Number of Attempts', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='#4ECDC4')
    ax2.set_ylabel('Computational Cost (×)', fontsize=11, fontweight='bold', color='#FF69B4')
    ax.set_title('Self-Verification Attempt Count Impact', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelcolor='#4ECDC4')
    ax2.tick_params(axis='y', labelcolor='#FF69B4')
    ax.set_axisbelow(True)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=10)
    
    # ---- Subplot 4: Synergy Analysis ----
    categories = ['CoT Only', 'Self-Verify\nOnly', 'CoT +\nSelf-Verify', 'Synergy\nGain']
    values = [84.2, 81.7, 87.9, 3.7]
    colors_synergy = ['#45B7D1', '#FFA07A', '#98D8C8', '#90EE90']
    
    ax = axes[1, 1]
    bars = ax.bar(categories, values, color=colors_synergy, edgecolor='black', 
                 linewidth=1.5, alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Method Synergy Analysis', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 95])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if val < 20:
            y_pos = height + 2
        else:
            y_pos = height + 1
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('figure6_hyperparameter_sensitivity.pdf', dpi=300, bbox_inches='tight')
    print("✓ Saved: figure6_hyperparameter_sensitivity.pdf")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all visualizations"""
    
    print("=" * 70)
    print("Generating Report Visualizations")
    print("=" * 70)
    
    # Create output directory if needed
    Path("visualizations").mkdir(exist_ok=True)
    
    print("\n[1/6] Generating Performance Comparison...")
    generate_performance_comparison()
    
    print("[2/6] Generating Difficulty Analysis...")
    generate_difficulty_analysis()
    
    print("[3/6] Generating Temperature Sensitivity...")
    generate_temperature_sensitivity()
    
    print("[4/6] Generating Error Analysis...")
    generate_error_analysis()
    
    print("[5/6] Generating Cost-Benefit Analysis...")
    generate_cost_benefit()
    
    print("[6/6] Generating Hyperparameter Sensitivity...")
    generate_hyperparameter_sensitivity()
    
    print("\n" + "=" * 70)
    print("✓ All visualizations generated successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - figure1_performance_comparison.pdf")
    print("  - figure2_difficulty_analysis.pdf")
    print("  - figure3_temperature_sensitivity.pdf")
    print("  - figure4_error_analysis.pdf")
    print("  - figure5_cost_benefit.pdf")
    print("  - figure6_hyperparameter_sensitivity.pdf")
    print("\nPlace these figures in your report or LaTeX document.")

if __name__ == "__main__":
    main()
