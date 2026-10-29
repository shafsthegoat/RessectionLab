'use strict';
// Reuse the hidden source/GPU/HUD/optics/save checker; never show a window.
process.env.RESSECTIONLAB_THERMAL_QUALIFICATION='1';
for(const suffix of ['CASE','OUTPUT','REPORT','REOPEN_EXPECT']){
  if(process.env['BIPOLAR_'+suffix])process.env['VASCULAR_'+suffix]=process.env['BIPOLAR_'+suffix];
}
require('./verify-vascular-rehearsal.cjs');
