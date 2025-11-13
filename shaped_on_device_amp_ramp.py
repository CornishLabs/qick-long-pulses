from qick import QickSoc, QickProgram
import numpy as np
import matplotlib.pyplot as plt

def blackman_harris_4term(N: int, periodic: bool = False) -> np.ndarray:
    """4-term Blackman–Harris window samples, length N, in [0,1]."""
    if N < 2:
        return np.ones(N, dtype=float)
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    M = N if periodic else (N - 1)
    n = np.arange(N, dtype=float)
    return (a0
            - a1 * np.cos(2.0 * np.pi * n / M)
            + a2 * np.cos(4.0 * np.pi * n / M)
            - a3 * np.cos(6.0 * np.pi * n / M))

def blackman_harris_4term_derivative(
    N: int,
    periodic: bool = False,
    *,
    dt: float | None = None,
    T: float | None = None,
    return_dw_dn: bool = False,
) -> np.ndarray:
    """
    Analytic derivative of the 4-term Blackman–Harris window, sampled at n=0..N-1.

    By default returns dw/dt (time derivative). If `return_dw_dn=True`, returns dw/dn.
    Provide either `dt` (sample period) or `T` (total duration). If both are None,
    dw/dn is returned.

    periodic=False uses M=N-1 (symmetric window with endpoints ~0).
    periodic=True uses M=N (periodic for N-point DFT, last point excluded conceptually).
    """
    if N < 2:
        return np.zeros(N, dtype=float)

    a1, a2, a3 = 0.48829, 0.14128, 0.01168
    M = N if periodic else (N - 1)
    n = np.arange(N, dtype=float)
    k1 = 2.0 * np.pi / M
    k2 = 2.0 * k1
    k3 = 3.0 * k1

    # dw/dn per the analytic derivative above
    dwdn = (
        (k1 * a1) * np.sin(k1 * n)
        - (k2 * a2) * np.sin(k2 * n)
        + (k3 * a3) * np.sin(k3 * n)
    )

    if return_dw_dn:
        return dwdn

    # Convert to dw/dt if timing provided or infer from T
    if dt is None and T is not None:
        dt = (T / N) if periodic else (T / (N - 1))
    if dt is None:
        # No time scaling info → return dw/dn
        return dwdn

    return dwdn / dt
    

class RepeatOnTriggerProgram(QickProgram):
    """
    Piecewise-linear ramp where segments are defined by a list of amplitudes.
    Each adjacent pair (amps[i] -> amps[i+1]) is one segment that lasts
    ramp_time_us with n_steps points.

    DMEM layout (param_base):
      [0]              NSEG = len(amps) - 1
      [1 .. 1+NSEG-1]  start_amp[i] for i=0..NSEG-1  (DAC units)
      [1+NSEG .. ]     d_gain_q[i]   for i=0..NSEG-1  (Q-frac_bits per step)
    """

    def __init__(self, soccfg, cfg):
        super().__init__(soccfg)
        self.cfg = cfg
        self.make_program()

    def make_program(self):
        self.initialize()
        self.body()
        self.end()

    def initialize(self):
        ch = self.cfg["main_ch"]
        self.declare_gen(ch=ch, nqz=1)
        self.default_pulse_registers(ch=ch, style="const", phase=self.deg2reg(0, gen_ch=ch))
        
        ref_ch = int(self.cfg.get("ref_ch", 0))
        self.declare_gen(ch=ref_ch, nqz=1)
        self.default_pulse_registers(ch=ref_ch, style="const", phase=self.deg2reg(0, gen_ch=ref_ch))
        
        self.synci(200) # Give time for setting up some pulses

    def body(self):
        ch = self.cfg["main_ch"]
        ref_ch = int(self.cfg.get("ref_ch", 0))


        # Compile-time settings
        ramp_time_us  = float(self.cfg["ramp_time_us"])   # per-segment duration
        frequency_mhz = float(self.cfg["frequency_mhz"])
        n_steps       = int(self.cfg["n_steps"])          # steps per segment
        if n_steps <= 0:
            raise ValueError("n_steps must be >= 1")
            
                   
        ref_duration_us    = float(self.cfg.get("ref_duration_us", 100.0))   # 100 µs
        ref_frequency_mhz  = float(self.cfg.get("ref_frequency_mhz", frequency_mhz))
        ref_gain           = int(self.cfg.get("ref_gain", 20000))

        frac_bits     = int(self.cfg.get("frac_bits", 16)) # gain Q-format
        pfrac_bits    = int(self.cfg.get("pfrac_bits", frac_bits)) # phase Q-format
        param_base    = int(self.cfg.get("param_base", 0))

        # Pulse timing for each *step* in a segment
        step_len_us  = ramp_time_us / n_steps
        step_len_gen = int(self.us2cycles(step_len_us, gen_ch=ch))
        if step_len_gen < 6:
            step_len_gen = 6
        
        tproc_dt = max(1, int(round(step_len_us * self.tproccfg["f_time"])))
        
        ref_len_gen = int(self.us2cycles(ref_duration_us, gen_ch=ref_ch))


        # Special regs & wiring
        rp   = self.ch_page(ch)
        rF   = self.sreg(ch, "freq")
        rP   = self.sreg(ch, "phase")
        rG   = self.sreg(ch, "gain")
        rM   = self.sreg(ch, "mode")
        rT   = self.sreg(ch, "t")
        tproc_ch = self.soccfg["gens"][ch]["tproc_ch"]
        
        # --- special regs for reference channel ---
        rp_ref      = self.ch_page(ref_ch)
        rF_ref      = self.sreg(ref_ch, "freq")
        rP_ref      = self.sreg(ref_ch, "phase")
        rG_ref      = self.sreg(ref_ch, "gain")
        rM_ref      = self.sreg(ref_ch, "mode")
        rT_ref      = self.sreg(ref_ch, "t")
        tproc_ch_ref = self.soccfg["gens"][ref_ch]["tproc_ch"]

        # Mode word
        mc = self._gen_mgrs[ch].get_mode_code(
            length=step_len_gen, stdysel="zero", mode="oneshot", outsel="dds", phrst=0
        )
        
        # --- ADD: mode word for reference pulse (oneshot, zero steady) ---
        mc_ref = self._gen_mgrs[ref_ch].get_mode_code(
            length=ref_len_gen, stdysel="zero", mode="oneshot", outsel="dds", phrst=0
        )


        # Static regs
        self.safe_regwi(rp, rF, self.freq2reg(frequency_mhz, gen_ch=ch),  "freq")
        self.safe_regwi(rp, rP, self.deg2reg(0.0, gen_ch=ch),             "phase = 0")
        self.safe_regwi(rp, rM, mc,                                      f"mode (len={step_len_gen})")
        self.safe_regwi(rp, rT, 0,                                        "t = 0")
        
        
        # --- ADD: program ref channel and queue a single pulse at t=0 ---
        self.safe_regwi(rp_ref, rF_ref, self.freq2reg(ref_frequency_mhz, gen_ch=ref_ch), "ref freq")
        self.safe_regwi(rp_ref, rP_ref, self.deg2reg(0.0, gen_ch=ref_ch),                 "ref phase = 0")
        self.safe_regwi(rp_ref, rM_ref, mc_ref,                                           f"ref mode (len={ref_len_gen})")
        self.safe_regwi(rp_ref, rG_ref, ref_gain,                                         "ref gain")
        self.safe_regwi(rp_ref, rT_ref, 0,                                                "ref t = 0")

        # queue the 100 µs marker at absolute time 0 on the ref channel
        self.set(tproc_ch_ref, rp_ref, rF_ref, rP_ref, 0, rG_ref, rM_ref, rT_ref)

        # Registers
        rN      = 1   # inner loop counter: steps per segment
        rDT     = 2   # Δt (tProc cycles)
        rGAQ    = 3   # gain accumulator (Q)
        rDGAQ   = 4   # Δgain per step (Q)
        rBASE   = 5   # param_base
        rTMP    = 6   # temp
        rAADDR  = 7   # DMEM addr for start_amp[i]
        rDADDR  = 8   # DMEM addr for d_gain_q[i]
        rNSEG   = 9   # number of segments remaining
        
        rPAQ    = 10  # phase accumulator (Q in phase-register units)
        rDPAQ   = 11  # Δphase per step (Q in phase-register units)
        rPADDR  = 12  # DMEM addr for start_phase_q[i]
        # rBASE will be reused to hold DPADDR (addr of d_phase_q[i])

        self.regwi(rp, rDT,  tproc_dt, "Δt (tProc cycles)")
        self.regwi(rp, rBASE, param_base, "param_base")

        # rNSEG = DMEM[base+0]
        # Get the number of segments, put into rNSEG register
        self.memr(rp, rNSEG, rBASE) 

        # rAADDR = base + 1
        # Add one to the memory start loc register rBASE and put into rAADDR register
        # This is the start of the amp memory
        self.mathi(rp, rAADDR, rBASE, "+", 1)
        # rDADDR = rAADDR + rNSEG
        # Add rNSEG register conents to the rAADDR register and put into rDADDR register
        # This is the start of the amp step memory
        self.math(rp, rDADDR, rAADDR, "+", rNSEG)
        
        # Phase tables follow immediately after the amplitude tables:
        # PADDR = DADDR + NSEG, DPADDR = PADDR + NSEG
        self.math(rp, rPADDR, rDADDR, "+", rNSEG)   # start_phase_q base
        self.math(rp, rBASE,  rPADDR, "+", rNSEG)   # reuse rBASE as DPADDR
        
        # Repurpose rNSEG as a outer loop counter
        self.mathi(rp, rNSEG, rNSEG, "-", 1)

        # --- outer loop over segments ---
        self.label("SEGMENT")
        # Load start_amp[i] -> rTMP, convert to Q
        self.memr(rp, rTMP, rAADDR) # Read segment start amp into temp register
        self.bitwi(rp, rGAQ, rTMP, "<<", frac_bits) # Shift this into the top 16 bits (most significant) of the GAQ register
        self.memr(rp, rDGAQ, rDADDR) # Load d_gain_q[i]: Load this segments amp step into the rDGAQ (gain per step) register
        
        # Load start phase, and d_phase
        self.memr(rp,rPAQ, rPADDR)
        self.memr(rp,rDPAQ, rBASE) # (BASE currently holds DPADDR)
        
        self.regwi(rp, rN, n_steps-1, "steps this segment")

        # --- inner loop over steps in this segment ---
        self.label("LOOP")
        self.bitwi(rp, rG, rGAQ, ">>", frac_bits) # Put the gain accumulator (bit shifted down, crops fractional part) into the pulse gain register
        self.mathi(rp, rP, rPAQ, "+", 0) # Copy current phase accumulator value into phase register
        
        # Schedule pulse
        self.set(tproc_ch, rp, rF, rP, 0, rG, rM, rT) # put the pulse into the queue
        
        # Advance time, gain, and phase accumulators
        self.math(rp, rT,   rT,   "+", rDT) # Increase the pulse time register by rDT
        self.math(rp, rGAQ, rGAQ, "+", rDGAQ) # Increase the gain accumulator register by rDGAQ 
        self.math(rp, rPAQ, rPAQ, "+", rDPAQ) # Increase the phase accumulator register by rDGAQ 
        self.loopnz(rp, rN, "LOOP") # if rN==0: continue else: (rN->rN-1 & jump to loop)

        # advance to next segment (increment counters)
        self.mathi(rp, rAADDR, rAADDR, "+", 1)
        self.mathi(rp, rDADDR, rDADDR, "+", 1)
        self.mathi(rp, rPADDR, rPADDR, "+", 1)
        self.mathi(rp, rBASE,  rBASE,  "+", 1) # (DPADDR++)
        
        self.loopnz(rp, rNSEG, "SEGMENT") # if rNSEG==0: continue else: rN->rN-1

        # relax before next external trigger
        self.sync_all(self.us2cycles(self.cfg["relax_delay_us"]))


    def run_loop(self, soc, *, amps, phases=None):
        """
        amplanguages   : list of DAC amplitudes (e.g., [100, 300, 10000]).
                 Each adjacent pair is one segment of duration ramp_time_us with n_steps steps.
        phases : list of phases in radians, same length as amps. If None, zeros are used.
                 Linear interpolation per segment, exactly like amplitude.
        """
        amps = list(map(int, amps))
        if len(amps) < 2:
            raise ValueError("amps must have at least 2 values.")
            
        nseg = len(amps) - 1

        # Clip to DAC range
        amps = [max(0, min(32766, a)) for a in amps]
        
        if phases is None:
            phases = np.zeros(len(amps), dtype=float) # rad
        else:
            phases = np.asarray(phases, dtype=float) # rad
            if len(phases) != len(amps):
                raise ValueError("phases must be the same length as amps.")

        ch         = self.cfg["main_ch"]
        n_steps   = int(self.cfg["n_steps"])      # per segment
        frac_bits = int(self.cfg.get("frac_bits", 16))
        pfrac_bits = int(self.cfg.get("pfrac_bits", frac_bits))
        base      = int(self.cfg.get("param_base", 0))
        
        starts = amps[:-1]
        ends   = amps[1:]

        # Per-segment Q increment; choose d so the *last* emitted point hits 'end'
        if n_steps == 1:
            d_q = [0 for _ in range(nseg)]
        else:
            d_q = [int(round((e - s) * (1 << frac_bits) / (n_steps - 1))) for s, e in zip(starts, ends)]
            
        # ----- Phase path -----
        # Convert radians -> DDS phase-register units (integer full scale).
        phases_reg = [self.deg2reg((180/np.pi)*phase_rad, gen_ch=ch) for phase_rad in phases]
        p_starts_q = phases_reg[:-1]
        # dp_q = np.zeros(nseg, dtype=np.int32)
        
        # Use deg2reg(180) to deduce full-scale in a robust way: full_scale = 2 * reg(180°)
        # reg_full = 2 * self.deg2reg(180.0, gen_ch=ch)   # e.g., 65536 for 16-bit phase
        # reg_per_rad = reg_full / (2.0 * np.pi)

        # Unwrapped phase in *register units* (float), then build Q-format tables
        # p_reg = phases * reg_per_rad
        # p_starts_q = np.round(p_reg[:-1] * (1 << pfrac_bits)).astype(np.int64)
        
        

        # if n_steps == 1:
            # dp_q = np.zeros(nseg, dtype=np.int64)
        # else:
        
        # dp_q = np.round(((p_reg[1:] - p_reg[:-1]) * (1 << pfrac_bits)) / (n_steps - 1)).astype(np.int64)
        
        dps = (phases[1:] - phases[:-1])/(n_steps - 1)
        dp_q = [self.deg2reg((180/np.pi)*dp_rad, gen_ch=ch) for dp_rad in dps]

        # Pack DMEM: [NSEG] + starts[0..nseg-1] + d_q[0..nseg-1]
        words = np.concatenate([
            np.array([nseg], dtype=np.int32),
            np.asarray(starts, dtype=np.int32),
            np.asarray(d_q, dtype=np.int32),
            np.asarray(p_starts_q, dtype=np.int32),
            np.asarray(dp_q, dtype=np.int32),
            # dp_q.astype(np.int32),
        ])

        # Write DMEM and go
        soc.tproc.load_dmem(words, addr=base)

        # Do NOT reload DMEM now
        self.config_all(soc, load_envelopes=True, load_mem=False)

        soc.start_src(self.cfg.get("start_src", "external"))
        soc.start_tproc()
        return None


if __name__ == "__main__":
    soc = QickSoc(external_clk=False)
    soccfg = soc
    print(soccfg)
    ramp_time_us = 8.0
    cfg = {
        "main_ch": 1,
        "ramp_time_us": ramp_time_us,   # duration PER SEGMENT
        "frequency_mhz": 6.0,
        "relax_delay_us": 1.0,
        "n_steps": 30,          # steps PER SEGMENT
        "param_base": 0,
        "frac_bits": 16,
        "start_src": "external",
    }

    prog = RepeatOnTriggerProgram(soc, cfg)
    print(str(prog))
    
    # print(soccfg._get_ch_cfg(gen_ch=1)['b_phase'])
    
    N=23
    BH_amps_I = np.rint(20000*blackman_harris_4term(N))
    
    # DRAG pulses
    # Ω_Q = −β⁢/α (dΩ_I⁢(t)/dt)
    # α is related to the detuning
    # beta ~ 0.5 is tuned to get best fidelity
    
    BH_amps_Q = 4e5*blackman_harris_4term_derivative(N,dt=ramp_time_us) 
    # Prefactor chosen to look reasonable for now in the experiment it would be set based on the nearest detuned state.
    
    BH_amps = np.sqrt(BH_amps_I**2 + BH_amps_Q**2)
    #ϕ(t)=atan2(Q(t),I(t))
    BH_phase_rad = np.arctan2(BH_amps_Q, BH_amps_I) # radians [-\pi, \pi]
    
    fig,axs=plt.subplots(3)
    axs[0].plot(BH_amps_I,  c='g')
    axs[0].plot(BH_amps_Q,  c='b')
    axs[1].plot(BH_amps,    c='k')
    axs[2].plot(BH_phase_rad,    c='b')
    fig.savefig('IQplot.png')
    
    prog.run_loop(soc, amps=BH_amps, phases=BH_phase_rad)
