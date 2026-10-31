'use strict';
// Navigation is allowed only after an explicit discard or a successful save.
class RehearsalUnsaved {
  constructor(ask) { this.ask = ask; this.dirty = false; }
  changed(request, result) {
    if (!request.source && result && result.changed !== false && !['inspect','preview','previewApproach','previewStrategy'].includes(request.command) && result.outcome?.status !== 'rejected') this.dirty = true;
  }
  saved() { this.dirty = false; }
  async allowNavigation() {
    return !this.dirty || await this.ask();
  }
  installed() { this.dirty = false; }
}
module.exports = { RehearsalUnsaved };
