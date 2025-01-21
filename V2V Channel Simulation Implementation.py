import numpy as np
from scipy import special
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import laplace 

def generate_gaussian_process(t, N, sigma, fmax):
    """Generate Gaussian process using sum of sinusoids method"""
    x = np.zeros_like(t)
    c = np.sqrt(sigma**2/N) * np.ones(N)
    f = fmax * np.sin(np.pi/(2*N) * (np.arange(1, N+1) - 0.5))
    theta = 2 * np.pi * np.random.rand(N)
    
    for n in range(N):
        x += c[n] * np.cos(2*np.pi*f[n]*t + theta[n])
    return x

def theoretical_envelope_pdf(z, sigma1, sigma2):
    """Calculate theoretical envelope PDF"""
    return (4*z)/(sigma1**2 * sigma2**2) * special.k0(2*z/(sigma1*sigma2))

def theoretical_power_pdf(w, sigma1, sigma2):
    """Calculate theoretical power gain PDF"""
    return 2/(sigma1**2 * sigma2**2) * special.k0(2*np.sqrt(w)/(sigma1*sigma2))

# Simulation parameters
N1 = N2 = N3 = N4 = 10^6  # Number of sinusoids
fmax_Tx = fmax_Rx = 80  # Maximum Doppler frequencies (Hz)
sigma1 = sigma2 = 1.0   # Standard deviations
Ts = 0.0001            # Sampling time
t = np.arange(0, 1, Ts)  # Time vector
Ns = len(t)            # Number of samples

# Generate Gaussian processes
x1 = generate_gaussian_process(t, N1, sigma1, fmax_Tx)
y1 = generate_gaussian_process(t, N2, sigma1, fmax_Tx)
x2 = generate_gaussian_process(t, N3, sigma2, fmax_Rx)
y2 = generate_gaussian_process(t, N4, sigma2, fmax_Rx)

# Calculate complex channel gains
h1 = x1 + 1j*y1
h2 = x2 + 1j*y2
h = h1 * h2

# Extract components
x = np.real(h)
y = np.imag(h)
R = np.abs(h)
Omega = R**2
phase = np.angle(h)


# Question 3
# Plot Gaussian Process Distribution
plt.figure(figsize=(10, 6))
plt.hist(x1, bins=50, density=True, alpha=0.7, label='Simulated')
x_range = np.linspace(min(x1), max(x1), 100)
plt.plot(x_range, stats.norm.pdf(x_range, 0, sigma1/np.sqrt(2)), 
         'r-', label='Theoretical')
plt.title('Gaussian Process Distribution (x1)')
plt.xlabel('Amplitude')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)
plt.show()



# Question 4
# Calculate and plot ACF
def theoretical_acf(tau, fmax):
    """Calculate theoretical autocorrelation function"""
    return (sigma1**2 / 2) * special.j0(2 * np.pi * fmax * tau)

max_lag = 1000
lags = np.arange(-max_lag, max_lag+1)
tau = lags * Ts

# Calculate ACF using numpy's correlate
acf_x1 = np.correlate(x1 - np.mean(x1), x1 - np.mean(x1), mode='full')
acf_x1 = acf_x1[len(acf_x1)//2 - max_lag:len(acf_x1)//2 + max_lag + 1]
acf_x1 = acf_x1 / acf_x1[max_lag]  # Normalize

plt.figure(figsize=(10, 6))
plt.plot(tau, acf_x1, label='Simulated')
plt.plot(tau, theoretical_acf(tau, fmax_Tx), 'r--', label='Theoretical')
plt.title('Autocorrelation Function (x1 x1) Comparison')
plt.xlabel('Time lag (τ)')
plt.ylabel('ACF')
plt.legend()
plt.grid(True)
plt.show()


# Nombre de sinusoïdes à tester
N_values = [4, 64]

# Paramètres pour l'ACF
max_lag = 1000
lags = np.arange(0, max_lag + 1)  # Temps positif seulement
tau = lags * Ts

plt.figure(figsize=(10, 6))

for i, N in enumerate(N_values):
    # Générer des processus gaussiens simulés
    x_simulated = generate_gaussian_process(t, N, sigma1, fmax_Tx)
    y_simulated = generate_gaussian_process(t, N, sigma1, fmax_Tx)

    # Calculer les ACFs simulées
    acf_x1 = np.correlate(x_simulated - np.mean(x_simulated), 
                         x_simulated - np.mean(x_simulated), mode='full')
    acf_x1 = acf_x1[len(acf_x1)//2:]  # Conserver seulement τ >= 0
    acf_x1 /= acf_x1[0]  # Normalisation
    
    acf_y1 = np.correlate(y_simulated - np.mean(y_simulated), 
                         y_simulated - np.mean(y_simulated), mode='full')
    acf_y1 = acf_y1[len(acf_y1)//2:]  # Conserver seulement τ >= 0
    acf_y1 /= acf_y1[0]  # Normalisation

    # Tracer les résultats
    plt.subplot(3, 2, i + 1)
    plt.plot(tau, acf_x1[:len(tau)], label='Simulated ACF x', linewidth=1.5)
    plt.plot(tau, acf_y1[:len(tau)], label='Simulated ACF y', linewidth=1.5)
    plt.plot(tau, theoretical_acf(tau, fmax_Tx), 'r--', label='Theoretical ACF', linewidth=1.5)
    plt.title(f'N = {N}')
    plt.xlabel('Time lag (τ)')
    plt.ylabel('ACF')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Question 5
# Calculer x et y
x = x1 * x2 - y1 * y2
y = x1 * y2 + x2 * y1

# Histogramme des données simulées
plt.figure(figsize=(10, 6))

# Histogramme pour x
plt.subplot(1, 2, 1)
plt.hist(x, bins=50, density=True, alpha=0.7, label='Simulated x')
x_range = np.linspace(min(x), max(x), 100)
plt.plot(x_range, laplace.pdf(x_range, loc=0, scale=sigma1 * sigma2), 'r-', label='Theoretical x')
plt.title('Difference between Simulated and Theoretical x')
plt.xlabel('x')
plt.ylabel('PDF')
plt.legend()

# Histogramme pour y
plt.subplot(1, 2, 2)
plt.hist(y, bins=50, density=True, alpha=0.7, label='Simulated y')
y_range = np.linspace(min(y), max(y), 100)
plt.plot(y_range, laplace.pdf(y_range, loc=0, scale=sigma1 * sigma2), 'r-', label='Theoretical y')
plt.title('Difference between Simulated and Theoretical y')
plt.xlabel('y')
plt.ylabel('PDF')
plt.legend()

plt.tight_layout()
plt.show()


# Question 6
# Plot Envelope Distribution
plt.figure(figsize=(10, 6))
plt.hist(R, bins=50, density=True, alpha=0.7, label='Simulated')
r_range = np.linspace(0, max(R), 100)
plt.plot(r_range, theoretical_envelope_pdf(r_range, sigma1, sigma2), 
         'r-', label='Theoretical')
plt.title('Envelope Distribution')
plt.xlabel('Envelope (R)')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)
plt.show()

# Plot Power Gain Distribution
plt.figure(figsize=(10, 6))
plt.hist(Omega, bins=50, density=True, alpha=0.7, label='Simulated')
w_range = np.linspace(0, max(Omega), 100)
plt.plot(w_range, theoretical_power_pdf(w_range, sigma1, sigma2), 
         'r-', label='Theoretical')
plt.title('Power Gain Distribution')
plt.xlabel('Power Gain (Ω)')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)
plt.show()

# Question 7
# Plot Phase Distribution
plt.figure(figsize=(10, 6))
plt.hist(phase, bins=50, density=True, alpha=0.7)
plt.title('V2V Channel Phase Distribution')
plt.xlabel('Phase (radians)')
plt.ylabel('PDF')
plt.grid(True)
plt.show()


# Question 9
# Calculate BER for DPSK
gamma_dB = np.arange(0, 31, 2)
gamma_linear = 10**(gamma_dB/10)

# Data rate
D = 1e6  # 1 Mb/s

# Calculate correlation coefficients
def calculate_ber(gamma, rho):
    """Calculate BER for DPSK"""
    return 0.5 / (1 + gamma*(1-rho))

# Slow fading
rho_slow = special.j0(2*np.pi*fmax_Tx/D)
Pb_slow = calculate_ber(gamma_linear, rho_slow)

# Fast fading
rho_fast = special.j0(2*np.pi*fmax_Tx/(D/100))
Pb_fast = calculate_ber(gamma_linear, rho_fast)

plt.figure(figsize=(10, 6))
plt.semilogy(gamma_dB, Pb_slow, 'b-', label='Slow Fading', linewidth=2)
plt.semilogy(gamma_dB, Pb_fast, 'r--', label='Fast Fading', linewidth=2)
plt.grid(True)
plt.xlabel('Average SNR (dB)')
plt.ylabel('BER')
plt.title('BER Performance of DPSK in V2V Channel')
plt.legend()