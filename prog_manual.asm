// --- Params (page 1 to match your build) ---
  regwi 1, $CH,      1            // generator channel
  regwi 1, $FREQ,    4369067      // from your compiled output
  regwi 1, $PHASE,   0
  regwi 1, $ADDR,    0

// Length & flags  -> CTRL = (FLAGS<<12) | LENGTH
  regwi 1, $LEN,     1843         // pulse length (generator units)
  regwi 1, $FLAGS,   0x90         // QICK const/phrst/outsel bits for your style
  bitwi 1, $FLAGS,   $FLAGS << 12
  bitw  1, $CTRL,    $FLAGS | $LEN

// tProc-time delta between pulses (what QICK used)
  regwi 1, $DT,      1228         // delta t in tProc cycles
  regwi 1, $T,       0            // running absolute time

// (Optional) give the tProc a head start like your Python
  synci 200

// --- Pulse 1: gain = 3000 at t=0 ---
  regwi 1, $GAIN,    3000
  set   1, 1, $FREQ, $PHASE, $ADDR, $GAIN, $CTRL, $T

// --- Pulse 2: gain = 4000 at t += DT ---
  math  1, $T,       $T + $DT
  regwi 1, $GAIN,    4000
  set   1, 1, $FREQ, $PHASE, $ADDR, $GAIN, $CTRL, $T

// --- Pulse 3: gain = 5000 at t += DT ---
  math  1, $T,       $T + $DT
  regwi 1, $GAIN,    5000
  set   1, 1, $FREQ, $PHASE, $ADDR, $GAIN, $CTRL, $T

// (Optional) nudge toff to the end like the compiler does
  synci 4094
  end
