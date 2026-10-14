'use strict';
// The native collection now uses the main workspace; keep the existing verifier entry point.
if(process.env.ANATOMY_VERIFY_OUTPUT)process.env.UNIFIED_VERIFY_OUTPUT=process.env.ANATOMY_VERIFY_OUTPUT;
require('./verify-unified-workspace.cjs');
