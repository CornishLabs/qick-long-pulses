"""
This file produces three pulses with increasing amplitude, 
however it programs the SOC from the assembly program txt directly.
"""

from qick import QickSoc, QickProgram
from qick.parser import load_program

# Load bitstream with custom overlay
soc = QickSoc(external_clk=True)
print(soc)

# stop the tproc (if the tproc supports it)
soc.stop_tproc(lazy=False)

    