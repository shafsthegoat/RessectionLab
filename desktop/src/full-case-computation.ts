import type {SurgicalResponse} from './surgical-types';
type Scenario='primary'|'vascular';
/** One bounded engine request at a time; the next waits for display admission. */
export class FullCaseComputation {
  scenario:Scenario|null=null;
  private epoch=0;
  private initial=false;
  private inFlight=false;
  private lastResponse:SurgicalResponse|null=null;
  start(scenario:Scenario){this.pause();this.scenario=scenario;this.initial=true;}
  pause(){this.epoch++;this.scenario=null;this.inFlight=false;this.lastResponse=null;}
  claim(response:SurgicalResponse|null,pending:boolean,enabled:boolean) {
    if(!enabled){this.pause();return null;}
    if(!this.scenario||pending||this.inFlight||!response||(!this.initial&&response===this.lastResponse))return null;
    if(!this.initial&&(response.fullCase?.scenario!==this.scenario||!['ready','running'].includes(response.fullCase.status)||response.outcome?.status==='rejected')){this.pause();return null;}
    this.initial=false;this.inFlight=true;this.lastResponse=response;
    return {epoch:this.epoch,scenario:this.scenario,maxActions:2};
  }
  settled(epoch:number,ok:boolean){if(epoch===this.epoch){this.inFlight=false;if(!ok)this.pause();}}
}
