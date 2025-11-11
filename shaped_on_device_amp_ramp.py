from qick import QickSoc, QickProgram
import numpy as np

def blackman_harris_4term(N: int, periodic: bool = False) -> np.ndarray:
    """
    4-term Blackman–Harris window (Harris 1978), ~92 dB sidelobes.
    Returns values in [0, 1]. If periodic=True, use N samples for an N-point DFT;
    else symmetric with endpoints ~0.
    """
    if N < 2:
        return np.ones(N, dtype=float)
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    M = N if periodic else (N - 1)
    n = np.arange(N, dtype=float)
    w = (
        a0
        - a1 * np.cos(2.0 * np.pi * n / M)
        + a2 * np.cos(4.0 * np.pi * n / M)
        - a3 * np.cos(6.0 * np.pi * n / M)
    )
    return w

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
        self.synci(200)

    def body(self):
        ch = self.cfg["main_ch"]

        # Compile-time settings
        ramp_time_us  = float(self.cfg["ramp_time_us"])   # per-segment duration
        frequency_mhz = float(self.cfg["frequency_mhz"])
        n_steps       = int(self.cfg["n_steps"])          # steps per segment
        if n_steps <= 0:
            raise ValueError("n_steps must be >= 1")

        frac_bits     = int(self.cfg.get("frac_bits", 16))
        param_base    = int(self.cfg.get("param_base", 0))

        # Pulse timing for each *step* in a segment
        step_len_us  = ramp_time_us / n_steps
        step_len_gen = int(self.us2cycles(step_len_us, gen_ch=ch))
        if step_len_gen < 6:
            step_len_gen = 6
        tproc_dt = max(1, int(round(step_len_us * self.tproccfg["f_time"])))

        # Special regs & wiring
        rp   = self.ch_page(ch)
        rF   = self.sreg(ch, "freq")
        rP   = self.sreg(ch, "phase")
        rG   = self.sreg(ch, "gain")
        rM   = self.sreg(ch, "mode")
        rT   = self.sreg(ch, "t")
        tproc_ch = self.soccfg["gens"][ch]["tproc_ch"]

        # Mode word
        mc = self._gen_mgrs[ch].get_mode_code(
            length=step_len_gen, stdysel="zero", mode="oneshot", outsel="dds", phrst=0
        )

        # Static regs
        self.safe_regwi(rp, rF, self.freq2reg(frequency_mhz, gen_ch=ch), "freq")
        self.safe_regwi(rp, rP, self.deg2reg(0.0, gen_ch=ch),             "phase = 0")
        self.safe_regwi(rp, rM, mc,                                      f"mode (len={step_len_gen})")
        self.safe_regwi(rp, rT, 0,                                        "t = 0")

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
        
        # Repurpose rNSEG as a outer loop counter
        self.mathi(rp, rNSEG, rNSEG, "-", 1)

        # --- outer loop over segments ---
        self.label("SEGMENT")
        # Load start_amp[i] -> rTMP, convert to Q
        self.memr(rp, rTMP, rAADDR) # Read segment start amp into temp register
        self.bitwi(rp, rGAQ, rTMP, "<<", frac_bits) # Shift this into the top 16 bits (most significant) of the GAQ register
        # Load d_gain_q[i]
        self.memr(rp, rDGAQ, rDADDR) # Load this segments amp step into the rDGAQ (gain per step) register
        
        self.regwi(rp, rN, n_steps-1, "steps this segment")

        # --- inner loop over steps in this segment ---
        self.label("LOOP")
        self.bitwi(rp, rG, rGAQ, ">>", frac_bits) # Put the gain accumulator (bit shifted down, crops fractional part) into the pulse gain register
        self.set(tproc_ch, rp, rF, rP, 0, rG, rM, rT) # put the pulse into the queue
        self.math(rp, rT,   rT,   "+", rDT) # Increase the pulse time register by rDT
        self.math(rp, rGAQ, rGAQ, "+", rDGAQ) # Increase the gain accumulator register by rDGAQ 
        self.loopnz(rp, rN, "LOOP") # if rN==0: continue else: (rN->rN-1 & jump to loop)

        # advance to next segment (increment counters)
        self.mathi(rp, rAADDR, rAADDR, "+", 1)
        self.mathi(rp, rDADDR, rDADDR, "+", 1)
        self.loopnz(rp, rNSEG, "SEGMENT") # if rNSEG==0: continue else: rN->rN-1

        # relax before next external trigger
        self.sync_all(self.us2cycles(self.cfg["relax_delay_us"]))


    def run_loop(self, soc, *, amps):
        """
        amps: list of DAC amplitudes (e.g., [100, 300, 10000]).
              Each adjacent pair is one segment of duration ramp_time_us with n_steps steps.
        """
        amps = list(map(int, amps))
        if len(amps) < 2:
            raise ValueError("amps must have at least 2 values.")

        # Clip to DAC range
        amps = [max(0, min(32766, a)) for a in amps]

        n_steps   = int(self.cfg["n_steps"])      # per segment
        frac_bits = int(self.cfg.get("frac_bits", 16))
        base      = int(self.cfg.get("param_base", 0))

        nseg = len(amps) - 1
        starts = amps[:-1]
        ends   = amps[1:]

        # Per-segment Q increment; choose d so the *last* emitted point hits 'end'
        if n_steps == 1:
            d_q = [0 for _ in range(nseg)]
        else:
            d_q = [int(round((e - s) * (1 << frac_bits) / (n_steps - 1))) for s, e in zip(starts, ends)]

        # Pack DMEM: [NSEG] + starts[0..nseg-1] + d_q[0..nseg-1]
        words = np.array([nseg] + starts + d_q, dtype=np.int32)

        # Write DMEM and go
        soc.tproc.load_dmem(words, addr=base)

        # Do NOT reload DMEM now
        self.config_all(soc, load_envelopes=True, load_mem=False)

        soc.start_src(self.cfg.get("start_src", "external"))
        soc.start_tproc()
        return None


if __name__ == "__main__":
    soc = QickSoc(external_clk=False)
    cfg = {
        "main_ch": 1,
        "ramp_time_us": 500.0,   # duration PER SEGMENT
        "frequency_mhz": 12.0,
        "relax_delay_us": 1.0,
        "n_steps": 50,          # steps PER SEGMENT
        "param_base": 0,
        "frac_bits": 16,
        "start_src": "external",
    }

    prog = RepeatOnTriggerProgram(soc, cfg)
    print(str(prog))
    
    BH_amps = np.rint(10000*blackman_harris_4term(20))

    prog.run_loop(soc, amps=BH_amps)
