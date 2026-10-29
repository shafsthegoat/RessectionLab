'use strict';
// Reuse hidden native continuum/GPU/HUD/optics and save/cold-reopen checks.
for(const suffix of ['CASE','OUTPUT','REOPEN_EXPECT']){
  if(process.env['CONSEQUENCE_'+suffix])process.env['CONTINUUM_'+suffix]=process.env['CONSEQUENCE_'+suffix];
}
require('./verify-continuum-rehearsal.cjs');
