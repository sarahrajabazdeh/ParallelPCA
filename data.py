import numpy as np
import sys

size = int(sys.argv[1])
data = np.random.rand(size, 2) * 100  # Generate random data

# Save to CSV
np.savetxt('dataset.csv', data, delimiter=',')
