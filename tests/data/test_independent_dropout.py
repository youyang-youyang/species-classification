import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.finetune import finetune_forward_fn

class Recorder:
    training=True
    def __call__(self, **kw):
        self.inputs=kw
        return {},None,torch.tensor(1.),{'L_total':1.}

class Tests(unittest.TestCase):
    def setUp(self):
        self.model=Recorder()
        self.batch=dict(bases_21=torch.ones(1,2,21),uniqueness_scores=torch.ones(1,2),phylo_vecs=torch.ones(1,64),labels={})
    def run_case(self,mode,draw):
        with patch('scripts.finetune.torch.rand',return_value=torch.tensor(draw)):
            return finetune_forward_fn(self.model,self.batch,'cpu',guidance_dropout=.5,guidance_dropout_mode=mode)[1]
    def test_independent_four_modes(self):
        for u,p in [(0.1,0.1),(.1,.9),(.9,.1),(.9,.9)]:
            metrics=self.run_case('independent',[u,p])
            self.assertEqual(self.model.inputs['uniqueness_scores'] is None,u<.5)
            self.assertEqual(self.model.inputs['phylo_vecs'] is None,p<.5)
            self.assertEqual(metrics['guidance_drop_rate'],float(u<.5 and p<.5))
            self.assertIsNotNone(self.model.inputs['bases_21'])
    def test_joint_compatibility(self):
        for value in [.1,.9]:
            self.run_case('joint',[value])
            self.assertEqual(self.model.inputs['uniqueness_scores'] is None,value<.5)
            self.assertEqual(self.model.inputs['phylo_vecs'] is None,value<.5)
    def test_validation_no_random_draw(self):
        self.model.training=False
        self.batch['uniqueness_scores']=None
        del self.batch['phylo_vecs']
        with patch('scripts.finetune.torch.rand',side_effect=AssertionError('Validation must not sample')):
            finetune_forward_fn(self.model,self.batch,'cpu',guidance_dropout=.5,guidance_dropout_mode='independent')
        self.assertIsNone(self.model.inputs['uniqueness_scores'])
        self.assertIsNone(self.model.inputs['phylo_vecs'])
    def test_extreme_probabilities(self):
        for probability in [0.,1.]:
            finetune_forward_fn(self.model,self.batch,'cpu',guidance_dropout=probability,guidance_dropout_mode='independent')
            self.assertEqual(self.model.inputs['uniqueness_scores'] is None,probability==1.)
            self.assertEqual(self.model.inputs['phylo_vecs'] is None,probability==1.)

if __name__=='__main__': unittest.main()
