import numpy as np
import sys

# Get the number of data points from the command line arguments
size = int(sys.argv[1])

# Generate random data
data = np.random.rand(size, 2) * 100  

# Save to CSV
np.savetxt('dataset.csv', data, delimiter=',')
