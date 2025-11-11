regwi 1, $23, 0;                              //phase = 0
synci 200;
regwi 1, $22, 4369067;                        //freq = 4369067
regwi 1, $25, 3000;                           //gain = 3000
regwi 1, $26, 591667;                         //phrst| stdysel | mode | | outsel = 0b01001 | length = 1843 
regwi 1, $27, 0;                              //t = 0
set 1, 1, $22, $23, $0, $25, $26, $27;        //ch = 1, pulse @t = $27
regwi 1, $22, 4369067;                        //freq = 4369067
regwi 1, $25, 4000;                           //gain = 4000
regwi 1, $26, 591667;                         //phrst| stdysel | mode | | outsel = 0b01001 | length = 1843 
regwi 1, $27, 1228;                           //t = 1228
set 1, 1, $22, $23, $0, $25, $26, $27;        //ch = 1, pulse @t = $27
regwi 1, $22, 4369067;                        //freq = 4369067
regwi 1, $25, 8000;                           //gain = 5000
regwi 1, $26, 591667;                         //phrst| stdysel | mode | | outsel = 0b01001 | length = 1843 
regwi 1, $27, 2456;                           //t = 2456
set 1, 1, $22, $23, $0, $25, $26, $27;        //ch = 1, pulse @t = $27
synci 4094;
end ;