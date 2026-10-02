"""Evaluation-grid detection; memorization precedes confirmed generalization."""
from dataclasses import dataclass, asdict

@dataclass
class GrokkingState:
    memorization_step: int | None = None
    memorization_time_seconds: float | None = None
    first_threshold_crossing_step: int | None = None
    grokking_step: int | None = None
    grokking_time_seconds: float | None = None
    consecutive: int = 0
    def update(self,step,elapsed,train_acc,test_acc,config):
        events=[]
        if step==0: return events
        previous_mem=self.memorization_step
        if self.memorization_step is None and train_acc >= config['memorization']['train_accuracy_threshold']:
            self.memorization_step=step; self.memorization_time_seconds=elapsed; events.append('memorization')
        g=config['grokking']
        qualifies=train_acc>=g['train_accuracy_threshold'] and test_acc>=g['test_accuracy_threshold'] and previous_mem is not None
        if qualifies:
            if self.first_threshold_crossing_step is None: self.first_threshold_crossing_step=step
            self.consecutive+=1
        else: self.consecutive=0
        if self.grokking_step is None and self.consecutive>=g['required_consecutive_evals']:
            self.grokking_step=step; self.grokking_time_seconds=elapsed; events.append('grokking')
        return events
    def to_dict(self): return asdict(self)
