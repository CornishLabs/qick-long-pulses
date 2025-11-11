from qick import QickSoc, QickProgram

from typing import Any, Dict

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
        self.declare_gen(ch=main_ch, nqz=1) # set the nyquist zone
        
        default_phase = self.deg2reg(0, gen_ch=main_ch)

        # see https://docs.qick.dev/latest/_autosummary/qick.asm_v1.html#qick.asm_v1.FullSpeedGenManager for types
        self.default_pulse_registers(ch=main_ch, style="const", phase=default_phase)
        
        # Move processor timeline reference ahead of realtime, giving processor some time to configure pulses
        self.synci(200)  
    
    def body(self):
        main_ch = self.cfg["main_ch"]
        
        ramp_time_us = self.cfg["ramp_time_us"]
        start_amp = self.cfg["start_amp"]
        end_amp = self.cfg["end_amp"]
        frequency_mhz = self.cfg["frequency_mhz"]
        n_steps = self.cfg["n_steps"]
        
        # This linear interpolation is done in python making a really long programe
        # However we should be able to do the interpolation math on the device
        # with the assembly instructions...
        step_length_us = ramp_time_us/n_steps
        for i in range(n_steps):
            amp_here = int(start_amp + (i/(n_steps-1)) * (end_amp-start_amp))
            self.set_pulse_registers(ch=main_ch,
                             freq=self.freq2reg(frequency_mhz, gen_ch=main_ch),              # MUST specify gen ch
                             length=self.us2cycles(step_length_us,gen_ch=main_ch),   # MUST specify gen ch
                             gain=amp_here)
            self.pulse(ch=main_ch, t='auto')
                           
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
        "ramp_time_us": 40.0,
        "start_amp": 1000,
        "end_amp": 5000,
        "frequency_mhz": 10.0,
        "relax_delay_us": 1.0,
        "n_steps":100,
    }
    prog = RepeatOnTriggerProgram(soccfg, cfg)
    print(str(prog))
    prog.run_loop(soc)
    
if __name__ == "__main__":
    make_new_prog()

    
    
