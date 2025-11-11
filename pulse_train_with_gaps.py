from qick import QickSoc, QickProgram

from typing import Any, Dict

REQUIRED_KEYS = [
    "main_ch", "relax_delay_us",
    "pulse_freqs_mhz", "pulse_lengths_us", "pulse_gains_dac"
]

def validate_cfg(cfg: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")

    # if cfg["pulse_style"] not in ("const", "flat_top", "arb"):
    #     raise ValueError("pulse_style must be one of: 'const', 'flat_top', 'arb'.")

    n_f, n_l, n_g = map(len, (cfg["pulse_freqs_mhz"], cfg["pulse_lengths_us"], cfg["pulse_gains_dac"]))
    if not (n_f == n_l == n_g):
        raise ValueError(
            f"pulse_* arrays must have the same length (got freqs={n_f}, lengths={n_l}, gains={n_g})"
        )

    if cfg["relax_delay_us"] < 0 :
        raise ValueError("relax_delay_us >= 0, are required.")

# Load bitstream with custom overlay
soc = QickSoc(external_clk=True)
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
        mon_ch = self.cfg["mon_ch"] # Gen Channel number
        self.declare_gen(ch=main_ch, nqz=1) # set the nyquist zone
        
        default_phase = self.deg2reg(0, gen_ch=main_ch)

        # see https://docs.qick.dev/latest/_autosummary/qick.asm_v1.html#qick.asm_v1.FullSpeedGenManager for types
        self.default_pulse_registers(ch=main_ch, style="const", phase=default_phase)
        if mon_ch is not None:
            self.default_pulse_registers(ch=mon_ch, style="const", phase=default_phase)
        
        # Move processor timeline reference ahead of realtime, giving processor some time to configure pulses
        self.synci(200)  
    
    def body(self):
        main_ch = self.cfg["main_ch"]
        mon_ch = self.cfg["mon_ch"]
        freqs, lengths, gains = (
            self.cfg["pulse_freqs_mhz"],
            self.cfg["pulse_lengths_us"],
            self.cfg["pulse_gains_dac"],
        )
        if not (len(freqs) == len(lengths) == len(gains)):
            raise ValueError("pulse_* arrays must have the same length")

        for f, L, g in zip(freqs, lengths, gains):
            if g== 0:
                self.sync_all(self.us2cycles(L)) # DONT specify gen ch
            else:
                while L>0:
                    self.set_pulse_registers(ch=main_ch,
                             freq=self.freq2reg(f, gen_ch=main_ch),              # MUST specify gen ch
                             length=self.us2cycles(min(L,100),gen_ch=main_ch),   # MUST specify gen ch
                             gain=g)
                    self.pulse(ch=main_ch, t='auto')
                    
                    if mon_ch is not None:
                        # Pulse the same thing on the monitor, with 1/200 th the frequency, and 1/10 th the amp.
                        # Note that the DACs aren't on the same tile, so this will have a 2ns jitter I believe on relative start edges
                        # between restarts.
                        self.set_pulse_registers(ch=mon_ch,
                                 freq=self.freq2reg(f/200, gen_ch=mon_ch),             # MUST specify gen ch
                                 length=self.us2cycles(min(L,100),gen_ch=mon_ch),      # MUST specify gen ch
                                 gain=int(g*20))
                        self.pulse(ch=mon_ch, t='auto')
                    L -= 100
                           
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
    
# This is the RFSoC normal output, so gen 0 goes to DAC_B (lab monitor), gen 1 goes to DAC_A (lab main output)
#         2 signal generator channels:
#         0:      axis_signal_gen_v6 - envelope memory 65536 samples (6.667 us)
#                 fs=9830.400 MHz, fabric=614.400 MHz, 32-bit DDS, range=9830.400 MHz
#                 DAC tile 0, blk 0 is DAC_B
#         1:      axis_signal_gen_v6 - envelope memory 65536 samples (6.667 us)
#                 fs=9830.400 MHz, fabric=614.400 MHz, 32-bit DDS, range=9830.400 MHz
#                 DAC tile 2, blk 0 is DAC_A

current_config={
    "main_ch": 1, 
    "mon_ch": None,
    "relax_delay_us":1.0, # Time at the end to settle before next trigger is allowed
    "pulse_freqs_mhz":  [10,10,10],
    "pulse_lengths_us": [3,3,3],
    "pulse_gains_dac":  [3000,4000,5000]
}
validate_cfg(current_config)

def make_new_prog():
    try:
        validate_cfg(current_config)
    except Exception as e:
        warning(f"Config invalid, not rebuilding program: {e}")
        return
    prog = RepeatOnTriggerProgram(soccfg, current_config)
    print(str(prog))
    prog.run_loop(soc)
    
if __name__ == "__main__":
    make_new_prog()

    