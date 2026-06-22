import matplotlib.pyplot as plt
k_grids = [3,5,7,9]
cohesive_energies = [-7.21392,-7.21395,-7.21394,-7.21390]
plt.plot(k_grids, cohesive_energies, marker='o')
plt.xlabel('K-grid Density')
plt.ylabel('Cohesive Energy (eV)')
plt.title('Convergence of Carbon Cohesive Energy with K-grid Density')
plt.grid()
plt.ylim(-7.215, -7.213)
plt.savefig('carbon_convergence.png')