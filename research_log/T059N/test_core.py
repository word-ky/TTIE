from research_log.T059N.core import select,LABELS,partition,LIMIT
def test_eligibility_and_earliest_tie():
 assert select([1,2],[0,0])['classification']==LABELS[0]
 r=select([1,.01,.02],[0,.03,.03]);assert r['selected_epoch']==2 and r['classification']==LABELS[2]
 assert select([0],[1])['classification']==LABELS[1]
def test_partition_group_order():
 ids=list(range(48));p=dict(image_ids={'train':ids[::-1]},group_indices={str(i):i for i in ids},row_indices={'train':ids},canonical=[dict(image_id=i,bank_index=i) for i in ids]);s=partition(p);assert s['image_ids']['selector']==list(range(5,48,6));assert len(s['image_ids']['fit'])==40
