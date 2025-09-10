# Derived from serdespy

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import skrf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def four_port_to_diff(network: skrf.Network, Zs: float, Zl: float):
    """

    Parameters
    ----------
    network : skrf Network
        4 port network object
        example: 
            s4p_file = 'path/to/touchstonefile.s4p'
            network = rf.Network(s4p_file)
    
    port_def: 2*2 array
        defines TX and RX side ports of network
        example:
            port_def = np.array([[TX1_index, RX1_index],[TX2_index, RX2_index]])
            
            PORT DEFINITIONS: 
                Port   1  ----->  TX Side      G11     RX Side  <-----  Port   2
                Port   3  ----->  TX Side      G12     RX Side  <-----  Port   4
                
            port_def = np.array([[0, 1],[2, 3]])

    Returns
    -------
    H : array
        tranfer function of differential channel
    
    f : array
        frequency vector
    """
    s_params = network.s
    f = network.f
    pts = f.size
    
    
    #change port def
    #ports = np.array([1,3,2,4])
    s_params_new = np.copy(s_params)
    
    s_params_new[:,1,:] = np.copy(s_params[:,2,:])
    s_params_new[:,2,:] = np.copy(s_params[:,1,:])
    
    s_params_new[:,:,1] = np.copy(s_params[:,:,2])
    s_params_new[:,:,2] = np.copy(s_params[:,:,1])
    
    s_params_new[:,1,2] = np.copy(s_params[:,1,2])
    s_params_new[:,2,1] = np.copy(s_params[:,2,1])
    
    s_params_new[:,1,1] = np.copy(s_params[:,2,2])
    s_params_new[:,2,2] = np.copy(s_params[:,1,1])
    
    
    #
    M = np.array([[1,-1,0,0],[0,0,1,-1],[1,1,0,0],[0,0,1,1]])
    invM = np.transpose(M)
    
    smm_params = np.zeros((4,4,pts), dtype = complex)
    
    for i in range(pts):
        smm_params[:,:,i] = (M@s_params_new[i,:,:]@invM)/2
    
    diff_s_params = smm_params[0:2,0:2,:]
    
    zl = Zl*np.ones((1,1,pts))
    zs = Zs*np.ones((1,1,pts))
    z0 = network.z0[0,0]*np.ones((1,1,pts))

    #reflection coefficients
    gammaL = (zl - z0) / (zl + z0)
    gammaL[zl == np.inf] = 1 
    
    gammaS = (zs - z0) / (zs + z0)
    gammaS[zs == np.inf] = 1
    
    gammaIn = (diff_s_params[0,0,:] + diff_s_params[0,1,:] * diff_s_params[1,0,:] * gammaL) / (1 - diff_s_params[1,1,:] * gammaL)
    
    H = diff_s_params[1,0,:] * (1 + gammaL) * (1 - gammaS) / (1 - diff_s_params[1,1,:] * gammaL) / (1 - gammaIn * gammaS) / 2
    
    H = H.reshape(pts,)

    # if option == 1:
    #     H = H/H[0]
    
    # else:
    #     h, t = freq2impulse(H,f)
    # h, t = freq2impulse(H,f)
        
    # return H, f, h, t

    H = H / H[0]

    return f, H


@dataclass
class ChannelTF:
    network: skrf.Network
    port_def: np.ndarray
    Zs: float
    Zl: float

    def thru_tf(self):
        f, H = four_port_to_diff(self.network, self.Zs, self.Zl)
        return f, H

if __name__ == "__main__":
    example_s4p = "../example_channel/Tp0_Tp5_28p5db_FQSFP_thru.s4p"
    channel_name = example_s4p.split("/")[-1].split(".s4p")[0]
    port_def = np.array([[0, 1], [2, 3]])
    Zs = 50
    Zl = 50

    channel = ChannelTF(skrf.Network(example_s4p), port_def, Zs, Zl)
    f, H = channel.thru_tf()

    f_nyquist = 26.56e9  # Nyquist frequency for 53.125 Gbps NRZ / 106.25 Gbps PAM4

    plt.figure()
    x_ghz = f / 1e9
    mag_db = 20 * np.log10(np.abs(H))
    plt.semilogx(x_ghz, mag_db, label="Channel TF")
    plt.axvline(x=f_nyquist / 1e9, color='grey', linestyle='--', label='Nyquist Frequency')

    # Mark channel loss at Nyquist with a horizontal line
    nyq_loss_db = float(np.interp(f_nyquist, f, mag_db))
    plt.axhline(y=nyq_loss_db, color='tab:red', linestyle=':', label=f'Loss @ Nyquist: {nyq_loss_db:.2f} dB')
    # Marker at the intersection point
    plt.plot([f_nyquist / 1e9], [nyq_loss_db], marker='o', color='tab:red', markersize=4, label='_nolegend_')

    # Denser and clearer log-scale xticks in GHz
    ax = plt.gca()
    # Determine range in GHz (positive only for log scale)
    x_min = np.nanmin(x_ghz[x_ghz > 0])
    x_max = np.nanmax(x_ghz)
    if np.isfinite(x_min) and np.isfinite(x_max) and x_min > 0:
        decade_min = int(np.floor(np.log10(x_min)))
        decade_max = int(np.ceil(np.log10(x_max)))
        major_ticks = []
        for p in range(decade_min, decade_max + 1):
            for m in (1.0, 2.0, 5.0):
                v = m * (10.0 ** p)
                if x_min <= v <= x_max:
                    major_ticks.append(v)
        if major_ticks:
            ax.set_xticks(major_ticks)
    # Minor ticks at 2..9 within each decade
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    # Compact major tick labels (values already in GHz)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{val:g}"))

    plt.ylabel("Magnitude (dB)")
    plt.xlabel("Frequency (GHz)")
    # plt.title("Channel Transfer Function")
    plt.title(f"Channel: {channel_name}")
    plt.grid(which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
