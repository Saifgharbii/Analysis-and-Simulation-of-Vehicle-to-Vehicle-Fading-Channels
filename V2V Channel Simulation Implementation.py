import numpy as np  # Import NumPy for numerical operations
from scipy import special  # Import SciPy's special functions (e.g., Bessel functions)
from scipy import stats  # Import statistical functions from SciPy
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

def generate_gaussian_process(t, N, sigma, fmax):
    """
    Generate a Gaussian process using the sum of sinusoids method.
    
    Parameters:
    t (array): Time vector.
    N (int): Number of sinusoids.
    sigma (float): Standard deviation of the process.
    fmax (float): Maximum Doppler frequency.
    
    Returns:
    array: Gaussian process.
    """
    x = np.zeros_like(t)  # Initialize the Gaussian process with zeros
    c = np.sqrt(sigma**2 / N) * np.ones(N)  # Coefficients for each sinusoid
    f = fmax * np.sin(np.pi / (2 * N) * (np.arange(1, N + 1) - 0.5))  # Frequency distribution
    theta = 2 * np.pi * np.random.rand(N)  # Random phases

    for n in range(N):
        # Add sinusoidal components with random phase and frequency
        x += c[n] * np.cos(2 * np.pi * f[n] * t + theta[n])
    return x

def theoretical_envelope_pdf(z, sigma1, sigma2):
    """
    Calculate the theoretical PDF of the envelope.
    
    Parameters:
    z (array): Envelope values.
    sigma1 (float): Standard deviation of the first process.
    sigma2 (float): Standard deviation of the second process.
    
    Returns:
    array: Theoretical PDF of the envelope.
    """
    return (4 * z) / (sigma1**2 * sigma2**2) * special.k0(2 * z / (sigma1 * sigma2))

def theoretical_power_pdf(w, sigma1, sigma2):
    """
    Calculate the theoretical PDF of the power gain.
    
    Parameters:
    w (array): Power gain values.
    sigma1 (float): Standard deviation of the first process.
    sigma2 (float): Standard deviation of the second process.
    
    Returns:
    array: Theoretical PDF of the power gain.
    """
    return 2 / (sigma1**2 * sigma2**2) * special.k0(2 * np.sqrt(w) / (sigma1 * sigma2))

# Simulation parameters
N1 = N2 = N3 = N4 = 8  # Number of sinusoids for each process
fmax_Tx = fmax_Rx = 80  # Maximum Doppler frequencies for transmitter and receiver (Hz)
sigma1 = sigma2 = 1.0  # Standard deviations of the processes
Ts = 0.0001  # Sampling time interval
t = np.arange(0, 1, Ts)  # Time vector
Ns = len(t)  # Total number of samples

# Generate Gaussian processes for transmitter and receiver
x1 = generate_gaussian_process(t, N1, sigma1, fmax_Tx)  # Real part of Tx
y1 = generate_gaussian_process(t, N2, sigma1, fmax_Tx)  # Imaginary part of Tx
x2 = generate_gaussian_process(t, N3, sigma2, fmax_Rx)  # Real part of Rx
y2 = generate_gaussian_process(t, N4, sigma2, fmax_Rx)  # Imaginary part of Rx

# Calculate complex channel gains for Tx and Rx
h1 = x1 + 1j * y1  # Complex channel gain for Tx
h2 = x2 + 1j * y2  # Complex channel gain for Rx
h = h1 * h2  # Overall complex channel gain (product of Tx and Rx)

# Extract channel gain components
x = np.real(h)  # Real part of the overall channel gain
y = np.imag(h)  # Imaginary part of the overall channel gain
R = np.abs(h)  # Envelope of the channel gain
Omega = R**2  # Power gain of the channel
phase = np.angle(h)  # Phase of the channel gain

# Plot the Gaussian process distribution
plt.figure(figsize=(10, 6))
plt.hist(x1, bins=50, density=True, alpha=0.7, label='Simulated')  # Histogram of simulated data
x_range = np.linspace(min(x1), max(x1), 100)  # Range for theoretical PDF
plt.plot(x_range, stats.norm.pdf(x_range, 0, sigma1 / np.sqrt(2)), 
         'r-', label='Theoretical')  # Plot theoretical Gaussian PDF
plt.title('Gaussian Process Distribution (x1)')
plt.xlabel('Amplitude')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)
plt.show()

# Plot the envelope distribution
plt.figure(figsize=(10, 6))
plt.hist(R, bins=50, density=True, alpha=0.7, label='Simulated')  # Histogram of simulated envelope
r_range = np.linspace(0, max(R), 100)  # Range for theoretical PDF
plt.plot(r_range, theoretical_envelope_pdf(r_range, sigma1, sigma2), 
         'r-', label='Theoretical')  # Plot theoretical envelope PDF
plt.title('Envelope Distribution')
plt.xlabel('Envelope (R)')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)
plt.show()

# Plot the power gain distribution
plt.figure(figsize=(10, 6))
plt.hist(Omega, bins=50, density=True, alpha=0.7, label='Simulated')  # Histogram of power gain
w_range = np.linspace(0, max(Omega), 100)  # Range for theoretical PDF
plt.plot(w_range, theoretical_power_pdf(w_range, sigma1, sigma2), 
         'r-', label='Theoretical')  # Plot theoretical power gain PDF
plt.title('Power Gain Distribution')
plt.xlabel('Power Gain (Ω)')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)
plt.show()

# Define theoretical ACF
def theoretical_acf(tau, fmax):
    """
    Calculate the theoretical autocorrelation function.
    
    Parameters:
    tau (array): Time lags.
    fmax (float): Maximum Doppler frequency.
    
    Returns:
    array: Theoretical ACF.
    """
    return special.j0(2 * np.pi * fmax * tau)  # Zeroth-order Bessel function

# Calculate and normalize ACF for x1
max_lag = 1000  # Maximum lag for ACF calculation
lags = np.arange(-max_lag, max_lag + 1)  # Lag range
tau = lags * Ts  # Convert lags to time lags
acf_x1 = np.correlate(x1 - np.mean(x1), x1 - np.mean(x1), mode='full')  # Calculate ACF
acf_x1 = acf_x1[len(acf_x1) // 2 - max_lag:len(acf_x1) // 2 + max_lag + 1]  # Extract desired range
acf_x1 = acf_x1 / acf_x1[max_lag]  # Normalize ACF

# Plot ACF comparison
plt.figure(figsize=(10, 6))
plt.plot(tau, acf_x1, label='Simulated')  # Plot simulated ACF
plt.plot(tau, theoretical_acf(tau, fmax_Tx), 'r--', label='Theoretical')  # Plot theoretical ACF
plt.title('Autocorrelation Function Comparison')
plt.xlabel('Time lag (τ)')
plt.ylabel('ACF')
plt.legend()
plt.grid(True)
plt.show()

# Plot the phase distribution of the V2V channel
plt.figure(figsize=(10, 6))
plt.hist(phase, bins=50, density=True, alpha=0.7)  # Histogram of phase
plt.title('V2V Channel Phase Distribution')
plt.xlabel('Phase (radians)')
plt.ylabel('PDF')
plt.grid(True)
plt.show()
