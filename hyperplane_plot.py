import matplotlib.pyplot as plt
import numpy as np

# Define the hyperplane: 2 - 23*x1 - x2 = 0
# Rearranged: x2 = 2 - 23*x1

# Create x1 values
x1 = np.linspace(-0.5, 0.3, 100)
x2 = 2 - 23 * x1

# Create the plot
plt.figure(figsize=(10, 8))
plt.plot(x1, x2, 'b-', linewidth=2, label='$y_1(x_1, x_2) = 0$')

# Mark the intercepts
x1_intercept = 2/23
x2_intercept = 2
plt.plot(x1_intercept, 0, 'ro', markersize=8, label=f'x₁-intercept: ({x1_intercept:.3f}, 0)')
plt.plot(0, x2_intercept, 'go', markersize=8, label=f'x₂-intercept: (0, {x2_intercept})')

# Add grid and labels
plt.grid(True, alpha=0.3)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.title('Hyperplane: $y_1(x_1, x_2) = 2 - 23x_1 - x_2 = 0$', fontsize=14)
plt.legend(fontsize=11)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('/home/beltranaceves/software/aau_machine_learning/hyperplane_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'hyperplane_plot.png'")
plt.show()
