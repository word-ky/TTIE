from ttie.movement_attribution import category,counterfactual,DEFINITIONS

def test_all_nine_axis_pairs_and_fixed_oracles():
    expected=[['correct_center','missed_move','missed_move'],['false_move','correct_move_direction','wrong_move_direction'],['false_move','wrong_move_direction','correct_move_direction']]
    for i,p in enumerate((.5,.4,.6)):
        for j,t in enumerate((.5,.4,.6)):
            assert category(p,t)==expected[i][j]
            a,b=counterfactual(p,t,[100.,-2.,-1.])
            assert a==(.5 if t==.5 else .6)
            assert (b==.5)==(p==.5)
            assert b==(t if p!=.5 and t!=.5 else p)
    assert counterfactual(.5,.6,[100.,2.,2.])[0]==.4
    assert counterfactual(.4,.5,[0.,2.,1.])[1]==.4

def test_four_literal_interpretations():
    assert DEFINITIONS['interpretation']=={'10':'necessity-dominant','01':'direction-dominant','11':'both individually sufficient / mixed','00':'neither sufficient / interaction-or-representation-limited'}
