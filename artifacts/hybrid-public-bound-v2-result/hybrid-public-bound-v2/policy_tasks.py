"""Same already-admitted public worlds; policy choice is the sole new delta."""
from hybrid_proposal import GeneratedSparseTask,RidgePriority
from fallback_policy import NoPositiveFallbackMixin
from bound_policy import PublicBoundFallbackMixin
from patient_regression import PilotAdmission
from patient_transfer import TransferAdmission
from batch_contract import RIDGE,need

class SourceTask(GeneratedSparseTask):
    def __init__(self,case,*,admission,ranker,max_steps=24,cancelled=None,costs=None):
        need(type(admission) in (PilotAdmission,TransferAdmission) and type(max_steps) is int
             and max_steps==24 and type(ranker) is RidgePriority and ranker.fingerprint==RIDGE['fingerprint'],
             'existing_source_admission_and_frozen_h24_model')
        self.admission=admission
        super().__init__(case,mode='ridge_top8',ranker=ranker,max_steps=24,cancelled=cancelled,costs=costs)
    def _require_source(self,case):self.admission.require_source(case)
    def _contract_record(self):return {**super()._contract_record(),'public_bound_source_admission':self.admission.fingerprint}
    def fresh(self):
        self._assert_frozen()
        return type(self)(self.case,admission=self.admission,ranker=self._ranker,
                         max_steps=self.max_steps,cancelled=self._cancelled,costs=self.costs)

class BaselineTask(NoPositiveFallbackMixin,SourceTask):pass
class BoundTask(PublicBoundFallbackMixin,SourceTask):pass
