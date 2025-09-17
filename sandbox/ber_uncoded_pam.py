"""
Illustration of Forney Gap for Uncoded PAM

Reference: https://cioffi-group.stanford.edu/doc/book/chap1.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))

def ber_uncoded_pam(M, SNR_dB):
    """Bit Error Rate for uncoded M-PAM over AWGN channel
    This determines the actual BER for uncoded PAM transmission given a certain SNR.

    """

    SNR = 10**(SNR_dB / 10)
    return (2 * (M - 1) / M) * Q(np.sqrt((6 * SNR) / (M**2 - 1)))

def snr_shannon_limit(M):
    """Shannon limit for transmission of log2(M) bits per symbol over AWGN channel
    This dictates the minimum SNR for BER-free transmission *given infinite block length (ideal coding)*.

    b = 0.5 * log2(1 + SNR)
    SNR = 2^(2b) - 1 = M^2 - 1
    """
    return 10 * np.log10(M**2 - 1)

def forney_gap_approx_db(target_ber):
    """Approximate Forney gap for a target BER
    From Cioffi's book, the Forney gap can be approximated as:
    Forney gap (dB) ≈ 20 * log10(Qinv(target_ber/2)) - 4.7712
    where Qinv is the inverse Q-function.
    """
    from scipy.special import erfcinv
    Qinv = erfcinv(target_ber / 2)
    return 20 * np.log10(Qinv) - 4.7712


if __name__ == "__main__":
    snr_db = np.arange(0, 40, 1)

    pam_order = [2, 4, 8]

    plt.figure()
    # Plot BER curves for different PAM orders
    for M in pam_order:
        # Difference between required SNR of shannon limit (zero BER, ideal coding) 
        # and practically required SNR for a target BER is the Forney gap.

        ber = ber_uncoded_pam(M, snr_db)
        plt.semilogy(snr_db, ber, label=f'{M}-PAM')
        shannon_limit = snr_shannon_limit(M)
        plt.axvline(shannon_limit, color='gray', linestyle='--')
        plt.text(shannon_limit + 0.5, 1e-5, f'Shannon limit for {M}-PAM', rotation=90, verticalalignment='center')

    plt.ylim(1e-15, 1)
    plt.xlim(0, 40)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER for Uncoded PAM with Forney Gap Illustration')
    plt.legend()
    plt.show()

    # Plot Forney Gap for a BER from 1e-4 to 1e-15 for each PAM order
    target_ber = np.logspace(-4, -15, num=50)
    plt.figure()
    for M in pam_order:
        required_snr = []
        for ber in target_ber:
            # Find the SNR that gives the target BER
            snr_vals = np.arange(0, 100, 0.1)
            ber_vals = ber_uncoded_pam(M, snr_vals)
            idx = np.argmin(np.abs(ber_vals - ber))
            required_snr.append(snr_vals[idx])
        
        required_snr = np.array(required_snr)
        shannon_limit = snr_shannon_limit(M)
        forney_gap = required_snr - shannon_limit

        plt.semilogy(forney_gap, target_ber, label=f'{M}-PAM')

    plt.semilogy(forney_gap_approx_db(target_ber), target_ber, 'k--', label='Approximation')
        
    plt.ylim(1e-15, 1e-4)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylabel('Target Bit Error Rate (BER)')
    plt.xlabel('Forney Gap (dB)')
    plt.title('Forney Gap for Uncoded PAM')
    plt.legend()
    plt.show()



