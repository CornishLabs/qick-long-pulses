from qick import QickSoc, QickProgram

from typing import Any, Dict

# Load bitstream with custom overlay
soc = QickSoc(external_clk=False)
soccfg = soc
print(str(soccfg))

class RepeatOnTriggerProgram(QickProgram):

    def __init__(self, soccfg, cfg):
        super().__init__(soccfg)
        self.cfg = cfg
        self.make_program()

    def make_program(self):
        self.initialize()
        self.body()
        self.end()
        
    def initialize(self):
        main_ch = self.cfg["main_ch"] # Gen Channel number
        self.declare_gen(ch=main_ch, nqz=1) # set the nyquist zone
        
        default_phase = self.deg2reg(0, gen_ch=main_ch)

        # see https://docs.qick.dev/latest/_autosummary/qick.asm_v1.html#qick.asm_v1.FullSpeedGenManager for types
        self.default_pulse_registers(ch=main_ch, style="const", phase=default_phase)
        
        # Move processor timeline reference ahead of realtime, giving processor some time to configure pulses
        self.synci(200)  
    
    def body(self):
        main_ch = self.cfg["main_ch"]

        # Extract config
        ramp_time_us  = float(self.cfg["ramp_time_us"])
        start_amp     = int(self.cfg["start_amp"])
        end_amp       = int(self.cfg["end_amp"])
        frequency_mhz = float(self.cfg["frequency_mhz"])
        n_steps       = int(self.cfg["n_steps"])
        if n_steps <= 0:
            raise ValueError("n_steps must be >= 1")

        # Calculate per-step pulse length
        step_len_us  = ramp_time_us / n_steps
        step_len_gen = int(self.us2cycles(step_len_us, gen_ch=main_ch))
        if step_len_gen < 6:
            step_len_gen = 6  # generator requires >=6 fabric cycles per pulse (number of instructions in loop)

        # Get time increment in tProc cycles per step
        tproc_dt = max(1, int(round(step_len_us * self.tproccfg["f_time"])))  # f_time in MHz

        # Define aliases for registers & channel wiring
        rp   = self.ch_page(main_ch)
        rF   = self.sreg(main_ch, "freq")
        rP   = self.sreg(main_ch, "phase")
        rG   = self.sreg(main_ch, "gain")
        rM   = self.sreg(main_ch, "mode")
        rT   = self.sreg(main_ch, "t")
        tproc_ch = self.soccfg["gens"][main_ch]["tproc_ch"]

        # mode word once (const, DDS)
        mc = self._gen_mgrs[main_ch].get_mode_code(
            length=step_len_gen, stdysel="zero", mode="oneshot", outsel="dds", phrst=0
        )

        # program static regs
        self.safe_regwi(rp, rF, self.freq2reg(frequency_mhz, gen_ch=main_ch), "freq")
        self.safe_regwi(rp, rP, self.deg2reg(0.0, gen_ch=main_ch),             "phase = 0")
        self.safe_regwi(rp, rM, mc,                                           f"mode (len={step_len_gen})")
        self.safe_regwi(rp, rT, 0,                                             "t = 0")

        # --- user registers on this page ---
        rN     = 1  # loop counter
        rDT    = 2  # Δt (tProc cycles)
        rGAQ   = 3  # gain accumulator (Q16)
        rDGAQ  = 4  # Δgain per step (Q16)

        self.regwi(rp, rN,  n_steps,  "N steps")
        self.regwi(rp, rDT, tproc_dt, "Δt (tProc cycles)")

        # Q16 fixed-point for gain accumulation
        FRAC = 16
        dga_q = 0 if n_steps == 1 else int(round((end_amp - start_amp) * (1 << FRAC) / (n_steps - 1)))
        self.regwi(rp, rGAQ, start_amp << FRAC, "gain_accum Q16")
        self.regwi(rp, rDGAQ, dga_q,           "Δgain Q16")

        # --- loop: quantize gain, schedule, then bump time & accumulator ---
        self.label("LOOP")
        # rG = rGAQ >> FRAC   (quantize to DAC units)
        self.bitwi(rp, rG, rGAQ, ">>", FRAC)
        # schedule one step at absolute time $rT
        self.set(tproc_ch, rp, rF, rP, 0, rG, rM, rT)
        # t += Δt; gain_accum += Δgain_q
        self.math(rp, rT,   rT,   "+", rDT)
        self.math(rp, rGAQ, rGAQ, "+", rDGAQ)
        self.loopnz(rp, rN, "LOOP")

        # settle before next external trigger
        self.sync_all(self.us2cycles(self.cfg["relax_delay_us"]))


        
    def run_loop(self, soc):
        """
        Parameters
        ----------
        soc : QickSoc
            Qick object
        """
        self.config_all(soc, load_envelopes=True, load_mem=True)
        # soc.start_src("internal") # "internal to pulse when program is ran, external to pulse on trigger (see start soc output)"
        soc.start_src("external")
        soc.start_tproc()
        return None
    
def make_new_prog():
    cfg = {
        "main_ch": 1,
        "ramp_time_us": 2000.0,
        "start_amp": 1000,
        "end_amp": 10000, 
        "frequency_mhz": 12, # Note, baluns are rated for 10MHz-10GHz
        "relax_delay_us": 1.0,
        "n_steps":10000,
    }
    prog = RepeatOnTriggerProgram(soccfg, cfg)
    print(str(prog))
    prog.run_loop(soc)
    
if __name__ == "__main__":
    make_new_prog()

    
    
