from qick import QickSoc, QickProgram

import numpy as np
class RepeatOnTriggerProgram(QickProgram):
    """
    Linear ramp where start/end are provided via tProc DMEM at run time.
    DMEM layout (param_base):
      [0] start_amp (int DAC units)
      [1] end_amp   (int DAC units)  -- not used by the loop, stored for bookkeeping
      [2] d_gain_q  (per-step increment in Q-frac_bits, signed)
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
        ramp_time_us  = float(self.cfg["ramp_time_us"])
        frequency_mhz = float(self.cfg["frequency_mhz"])
        n_steps       = int(self.cfg["n_steps"])
        if n_steps <= 0:
            raise ValueError("n_steps must be >= 1")

        frac_bits     = int(self.cfg.get("frac_bits", 16))   # set to 13 if your DMEM reads look /8
        param_base    = int(self.cfg.get("param_base", 0))   # DMEM base for params

        # Pulse timing
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

        # General-purpose regs on this page
        rN     = 1   # loop counter
        rDT    = 2   # Δt (tProc cycles)
        rGAQ   = 3   # gain accumulator (Q)
        rDGAQ  = 4   # Δgain per step (Q)
        rBASE  = 5   # base address into DMEM for parameters
        rTMP   = 6   # temp / address

        self.regwi(rp, rN,  n_steps,  "N steps")
        self.regwi(rp, rDT, tproc_dt, "Δt (tProc cycles)")
        self.regwi(rp, rBASE, param_base, "param_base")

        # Read start_amp (DAC int) from DMEM, convert to Q on-chip
        #   rTMP <- DMEM[param_base + 0] = start_amp
        self.memr(rp, rTMP, rBASE)
        self.bitwi(rp, rGAQ, rTMP, "<<", frac_bits)  # GAQ = start_amp << FRAC

        # Read d_gain_q from DMEM[param_base + 2]
        self.mathi(rp, rTMP, rBASE, "+", 2)
        self.memr(rp, rDGAQ, rTMP)

        # --- main loop: schedule, advance time, accumulate gain ---
        self.label("LOOP")
        # quantize GAQ -> rG (DAC units)
        self.bitwi(rp, rG, rGAQ, ">>", frac_bits)
        # schedule pulse at absolute time rT
        self.set(tproc_ch, rp, rF, rP, 0, rG, rM, rT)
        # advance time and accumulator
        self.math(rp, rT,   rT,   "+", rDT)
        self.math(rp, rGAQ, rGAQ, "+", rDGAQ)
        self.loopnz(rp, rN, "LOOP")

        # relax before next external trigger
        self.sync_all(self.us2cycles(self.cfg["relax_delay_us"]))

    # ----------------------------
    # Host-side runner
    # ----------------------------
    def run_loop(self, soc, *, start_amp: int, end_amp: int):
        """
        Load DMEM parameters and start the tProc.
        """
        n_steps   = int(self.cfg["n_steps"])
        frac_bits = int(self.cfg.get("frac_bits", 16))
        base      = int(self.cfg.get("param_base", 0))

        # Compute per-step delta in Q
        if n_steps == 1:
            d_gain_q = 0
        else:
            d_gain_q = int(round((end_amp - start_amp) * (1 << frac_bits) / (n_steps - 1)))

        # Write DMEM parameters
        # Words: [start_amp, end_amp, d_gain_q]
        
        param_words = np.array([start_amp, end_amp, d_gain_q], dtype=np.int32)
        soc.tproc.load_dmem(param_words, addr=base)

        # IMPORTANT: don't overwrite DMEM after we just wrote it
        self.config_all(soc, load_envelopes=True, load_mem=False)

        soc.start_src(self.cfg.get("start_src", "external"))
        soc.start_tproc()
        return None


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    soc = QickSoc(external_clk=False)
    cfg = {
        "main_ch": 1,
        "ramp_time_us": 100.0,
        "frequency_mhz": 12.0,
        "relax_delay_us": 1.0,
        "n_steps": 500,
        "param_base": 0,     # DMEM base for [start, end, d_gain_q]
        "frac_bits": 16,     # use 13 if your DMEM effectively drops 3 LSBs
    }

    prog = RepeatOnTriggerProgram(soc, cfg)
    print(str(prog))

    # Send amplitudes via DMEM just before running
    prog.run_loop(soc, start_amp=100, end_amp=10000)
