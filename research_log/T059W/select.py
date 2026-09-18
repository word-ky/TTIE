import math
IMAGE_IDS=[36660,37670,38070,38825,39551,39951,41488,41990,42563,43435,44195,45070,45728,46463,47121,47801]
def select(actions, image_ids=IMAGE_IDS):
    rows=[]
    for image_id in image_ids:
        group=[a for a in actions if a['image_id']==image_id and a['state_index']==0]
        assert group, 'T059-W active-path coverage blocked; no scientific classification'
        assert all(math.isfinite(a['norm']) for a in group)
        a=min(group,key=lambda a:(-a['norm'],a['bank_index']))
        rows.append({k:a[k] for k in ['image_id','bank_index','state_index','index','position','norm','active_regions']})
    assert all(a['norm']>0 and a['active_regions']>0 for a in rows), 'T059-W active-path coverage blocked; no scientific classification'
    return rows
