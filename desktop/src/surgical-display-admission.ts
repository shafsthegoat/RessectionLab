/** A renderer completion acknowledges one exact requested object in one source
 * generation. State IDs alone cannot distinguish a preview or a repeated seek. */
export class SurgicalDisplayAdmission<T extends object> {
  private requested:{source:string;generation:number;view:T}|null=null;
  private displayed:{source:string;generation:number;view:T}|null=null;
  expect(source:string,generation:number,view:T|null){
    if(!view){this.requested=null;return null;}
    if(this.requested?.source!==source||this.requested.generation!==generation||this.requested.view!==view)
      this.requested={source,generation,view};
    return this.requested;
  }
  isCurrent(ticket:ReturnType<SurgicalDisplayAdmission<T>['expect']>,view:T|null):boolean {
    return !!ticket&&ticket===this.requested&&ticket.view===view;
  }
  commit(ticket:ReturnType<SurgicalDisplayAdmission<T>['expect']>,view:T|null):boolean {
    if(!this.isCurrent(ticket,view)||!ticket)return false;
    if(this.displayed?.source===ticket.source&&this.displayed.generation===ticket.generation&&this.displayed.view===view)return false;
    this.displayed=ticket;return true;
  }
  invalidate(){this.requested=null;this.displayed=null;}
}
