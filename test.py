import matplotlib.pyplot as plt
import numpy as np

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 4))

# Define the vertices of the triangle
A = np.array([0, 0])
B = np.array([1, 0])
C = np.array([2, 0])
D = np.array([1, np.sqrt(3)])  # Height of the equilateral triangle
E = np.array([1, np.sqrt(3)/2])  # Midpoint of the height

# Draw the triangle
triangle = plt.Polygon([A, B, C, D], closed=True, fill=None, edgecolor='black', linewidth=1.5)
ax.add_patch(triangle)

# Draw the vertical line from A to E
plt.plot([A[0], E[0]], [A[1], E[1]], color='black', linewidth=1.5)

# Draw the line from B to E
plt.plot([B[0], E[0]], [B[1], E[1]], color='black', linewidth=1.5)

# Draw the line from C to E
plt.plot([C[0], E[0]], [C[1], E[1]], color='black', linewidth=1.5)

# Draw the line from A to D
plt.plot([A[0], D[0]], [A[1], D[1]], color='black', linewidth=1.5)

# Draw the line from B to D
plt.plot([B[0], D[0]], [B[1], D[1]], color='black', linewidth=1.5)

# Draw the line from C to D
plt.plot([C[0], D[0]], [C[1], D[1]], color='black', linewidth=1.5)

# Add labels with LaTeX formatting
ax.text(A[0], A[1] - 0.1, r'$\mathbf{A}$', fontsize=12, ha='center')
ax.text(B[0], B[1] - 0.1, r'$\mathbf{B}$', fontsize=12, ha='center')
ax.text(C[0], C[1] - 0.1, r'$\mathbf{C}$', fontsize=12, ha='center')
ax.text(D[0], D[1] + 0.1, r'$\mathbf{D}$', fontsize=12, ha='center')
ax.text(E[0], E[1] + 0.1, r'$\mathbf{E}$', fontsize=12, ha='center')

# Set limits and aspect
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2)
ax.set_aspect('equal')
ax.axis('off')  # Turn off the axis

# Show the plot
plt.tight_layout()
plt.show()