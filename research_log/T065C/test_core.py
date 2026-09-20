import numpy as np
from research_log.T065C.core import build_bank,predict,choose


def test_exact_distance_tie_order_and_unweighted_probability():
    rows=[dict(index=i,step=k,features=[v]+[0.]*10,safe=s) for i,k,v,s in [(2,1,1,1),(1,2,-1,0),(1,0,1,1),(0,0,-1,0),(2,0,1,1),(3,0,2,1)]]
    bank=build_bank(rows,dict(mean=[0.]*11,scale=[1.]*11));p,n=predict([[0.]*11],bank)
    assert [(r['index'],r['step']) for r in n[0]]==[(0,0),(1,0),(1,2),(2,0),(2,1)]
    assert [r['distance'] for r in n[0]]==[1.]*5 and p==[.6]


def test_leave_one_image_out_excludes_all_states():
    rows=[dict(index=0,step=k,features=[k/100]+[0.]*10,safe=1) for k in range(28)]
    rows += [dict(index=1,step=k,features=[1.+k]+[0.]*10,safe=0) for k in range(5)]
    bank=build_bank(rows,dict(mean=[0.]*11,scale=[1.]*11))
    assert predict([[0.]*11],bank)[0]==[1.]
    p,n=predict([[0.]*11],bank,exclude_image=0)
    assert p==[0.] and all(v['index']==1 for v in n[0])
    assert choose([.8,.4,.6],2)==2 and choose([.8,.4,.2],2)==0


def test_reused_normalization_and_euclidean_metric():
    norm=dict(mean=[2.]*11,scale=[2.]*11)
    rows=[dict(index=i,step=0,features=[2.+i]*11,safe=i%2) for i in range(6)]
    bank=build_bank(rows,norm);p,n=predict([[2.]*11],bank)
    assert bank['mean']==norm['mean'] and bank['scale']==norm['scale']
    np.testing.assert_allclose([v['distance'] for v in n[0]],np.arange(5)*np.sqrt(11)/2,rtol=0,atol=1e-14)
    assert p==[.4]


def test_independent_neighbors_and_ties_match():
    from research_log.T065C.verify import independent_neighbors
    rows=[dict(index=i//3,step=i%3,features=[float(i%2)]+[0.]*10,safe=i%2) for i in range(15)]
    bank=build_bank(rows,dict(mean=[0.]*11,scale=[1.]*11));q=[[0.]*11,[.2]*11]
    a,n=predict(q,bank,exclude_image=1);b,m=independent_neighbors(q,bank,1)
    assert a==b
    for x,y in zip(n,m):
        assert [(v['index'],v['step']) for v in x]==[(v['index'],v['step']) for v in y]
        np.testing.assert_allclose([v['distance'] for v in x],[v['distance'] for v in y],rtol=0,atol=1e-14)
