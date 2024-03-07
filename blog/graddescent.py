# Adjusting the plot to ensure the Z axis (Error) is clearly visible

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

# Plotting the function without grid lines
ax.plot_surface(x_vals, y_vals, z_vals, cmap='viridis', edgecolor='none', alpha=0.5)
ax.grid(False)  # Removes the grid lines

# Making the box background black
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')

# Plotting the path with markers
ax.plot(xs, ys, zs, color='r', marker='o')

# Annotating start and finish points
ax.text(start_x, start_y, start_z, "Start", color='white', fontsize=10)
ax.text(finish_x, finish_y, finish_z, "Finish", color='white', fontsize=10)

# Setting label colors and titles with smaller font size
ax.set_xlabel('Parameter 1', color='white', fontsize=10)
ax.set_ylabel('Parameter 2', color='white', fontsize=10)
ax.set_zlabel('Error', color='white', fontsize=10)
ax.set_title('Gradient Descent: Parameters vs. Error', color='white', fontsize=12)

# Adjusting tick color and removing grid lines
ax.tick_params(axis='x', colors='white', labelsize=8)
ax.tick_params(axis='y', colors='white', labelsize=8)
ax.tick_params(axis='z', colors='white', labelsize=8)

# Adjusting the view angle to keep the 3D perspective while ensuring the Z axis label is visible
ax.view_init(elev=30, azim=-45)

plt.show()

