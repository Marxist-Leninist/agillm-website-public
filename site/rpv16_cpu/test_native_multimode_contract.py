"""Run with: python3 -m unittest discover -s rpv16_cpu -p 'test_native_multimode_contract.py'."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from fastapi import FastAPI
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rpv16_multimode_v1 as mm

class NativeContractTests(unittest.TestCase):
    def test_mode_aliases(self):
        self.assertEqual(mm.mode_key({'mode':'sat','var':True}),'sat_variable')
        self.assertEqual(mm.mode_key({'mode':'sat','var':False}),'sat_fixed')
        self.assertEqual(mm.mode_key({'mode':'nat'}),'nat')

    def test_unsupported_mode_is_not_ar(self):
        with self.assertRaises(ValueError):mm.mode_key({'mode':'diffusion'})

    def test_causal_window(self):
        mask=mm.attention_mask(8,2)[0,0]
        self.assertTrue(mask[3,1]);self.assertFalse(mask[3,0]);self.assertFalse(mask[3,4])

    def test_sat_pair_mask(self):
        pair=[False]*8;pair[4]=True;mask=mm.attention_mask(8,2,pair)[0,0]
        self.assertTrue(mask[4,5]);self.assertFalse(mask[4,6]);self.assertFalse(mask[3,4])
        self.assertTrue(mask[4,2]);self.assertFalse(mask[4,1])

    def test_invalid_partition(self):
        for pair in ([True,True,False],[False,False,True],[False]):
            with self.assertRaises(ValueError):mm.attention_mask(3,2,pair)

    def test_policy_uses_both_strides(self):
        two=mm.select_stride(torch.tensor([0.,math.log(1.1)]),1.,.5)[0]
        one=mm.select_stride(torch.tensor([0.,math.log(5.)]),1.,.5)[0]
        self.assertEqual((one,two),(1,2))

    def test_merge_cost_changes_decision(self):
        prediction=torch.tensor([0.,math.log(1.6)])
        self.assertEqual(mm.select_stride(prediction,1.,0.)[0],2)
        self.assertEqual(mm.select_stride(prediction,1.,.5)[0],1)

    def test_nonfinite_policy_is_rejected(self):
        with self.assertRaises(ValueError):mm.select_stride(torch.tensor([0.,float('nan')]),1.,.5)

    def test_auxiliary_state_is_required(self):
        with self.assertRaises(ValueError):mm.check_aux({})

    def test_health_reports_actual_sampler_contract(self):
        engine=SimpleNamespace(available_modes=lambda:list(mm.CANONICAL),lock=SimpleNamespace(locked=lambda:False),
                               active_request_id=None,mode_error='',objective_counts={'ar':1,'sat':1,'nat':1},
                               aux={'regret_updates':torch.tensor(1)})
        ns={'RPV16CPU':object,'D':1280,'Q_HEADS':20,'KV_HEADS':5,'HEAD_DIM':64,
            'rms_norm':None,'app':FastAPI(),'health':lambda:{'model':{}},'engine':lambda:engine}
        mm.install(ns);health=ns['health']()
        expected={'ar':'ar','sat_fixed':'sat','sat_variable':'sat','nat':'nat'}
        self.assertEqual(set(health['canonical_modes']),set(expected))
        for key,sampler in expected.items():
            cap=health['capabilities'][key]
            self.assertTrue(cap['available']);self.assertEqual(cap['semantics']['runtime_sampler'],sampler)
        self.assertEqual(health['mode_limits']['nat']['max_new_tokens'],256)

if __name__=='__main__':unittest.main()
