"""Summary-analysis checks only; no model, fitting or inference."""
import pytest
from ben.analyse_t13b import contrast, positive_ci, unanimity, required_all


@pytest.mark.parametrize('votes,expected', [
    (['PASS']*3,'PASS'), (['FAIL']*3,'FAIL'),
    (['PASS','PASS','FAIL'],'INCONCLUSIVE'),
    (['FAIL','FAIL','PASS'],'INCONCLUSIVE'),
    (['FAIL','FAIL','INCONCLUSIVE'],'INCONCLUSIVE'),
])
def test_per_seed_deviation_requires_unanimity(votes, expected):
    assert unanimity(votes)==expected


def test_zero_bound_fails_and_missing_ci_is_not_a_pass():
    assert positive_ci(.1,[0,.2])=='FAIL'
    assert positive_ci(0,[.01,.2])=='FAIL'
    assert positive_ci(.1,None)=='INCONCLUSIVE'
    assert positive_ci(.1,[.01,.2])=='PASS'
    assert required_all(['PASS','INCONCLUSIVE','FAIL'])=='FAIL'


def test_missing_paired_ci_is_not_inferred_from_marginal_intervals():
    runs=[]
    for seed,gain in ((1,.2),(2,.3),(3,.4)):
        runs.append({'seed':seed,'modes':{'s2':{
            'out_of_fold':{'full':{'macro_map':.4+gain,'macro_map_ci95':[.6,.9]},
                           'none':{'macro_map':.4,'macro_map_ci95':[.35,.45]},
                           'shuffle':{'macro_map':.3,'macro_map_ci95':[.2,.35]}},
            'hypotheses':{'H1':{'per_arm':{'full':{'gain':gain,'gain_ci95':[gain-.05,gain+.05]}}}}
        }}})
    registered=contrast(runs,'s2','none')
    assert registered['verdict']=='PASS' and registered['mean']==pytest.approx(.3)
    for sample,gain in zip(registered['seeds'],(.2,.3,.4)):
        assert sample['ci95']==pytest.approx([gain-.05,gain+.05])
    shuffled=contrast(runs,'s2','shuffle')
    assert shuffled['verdict']=='INCONCLUSIVE'
    assert all(s['ci95'] is None for s in shuffled['seeds'])


def test_missing_seed_cannot_silently_pass():
    with pytest.raises(ValueError,match='three'):
        unanimity(['PASS','PASS'])
